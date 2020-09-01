import math
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .embednets import ItemEmbeddingNetwork, WordEmbeddingNetwork



class MMStatefulLSTM(nn.Module):
    """

    Args:
        nn ([type]): [description]
    """
    _HIDDEN_SIZE = 300

    def __init__(self, word_embeddings_path, word2id, pad_token, unk_token, seed, OOV_corrections, device='cpu'):
        torch.manual_seed(seed)
        super(MMStatefulLSTM, self).__init__()

        self.device = device
        self.memory_hidden_size = self._HIDDEN_SIZE
        n_encoders, n_heads = 1, 2
        
        #self.item_embeddings_layer = ItemEmbeddingNetwork(item_embeddings_path)
        self.word_embeddings_layer = WordEmbeddingNetwork(word_embeddings_path=word_embeddings_path, 
                                                        word2id=word2id, 
                                                        pad_token=pad_token, 
                                                        unk_token=unk_token)
        self.emb_dim = self.word_embeddings_layer.embedding_size
        self.sentence_encoder = SentenceEncoder(emb_dim=self.emb_dim, hidden_size=self.memory_hidden_size, bidirectional=True)

        self.memory_net = MemoryNet(emb_dim=self.emb_dim, memory_hidden_size=self.memory_hidden_size)

        #for h heads: d_k == d_v == emb_dim/h
        self.triton_attention = Triton(emb_dim=self.emb_dim,
                                        d_k=int(self.emb_dim/n_heads),
                                        d_v=int(self.emb_dim/n_heads),
                                        d_f=int(self.emb_dim/2),
                                        n_heads=n_heads,
                                        n_layers=n_encoders)

        self.layerNorm = nn.LayerNorm(self.memory_hidden_size)
        self.dropout = nn.Dropout(p=.1)
        self.out_layer = CandidateAttentiveOutput(in_features=self.memory_hidden_size, 
                                                hidden_size=int(self.memory_hidden_size/2))


    def forward(self, utterances, history, actions, attributes, focus_items, 
                candidates_pool, seq_lengths=None):
        """The visual context is a list of visual contexts (a batch). Each visual context is, in turn, a list
            of items. Each item is a list of (key, values) pairs, where key is a tensor containing the word ids
            for the field name and values is a list of values where each value is a tensor of word ids.

        type(visual_context[<sample_in_batch>][<item_n>][<field_n>]) -> tuple(tensor(key), [tensor(value)])

        Args:
            utterances ([type]): [description]
            history ([type]): [description]
            visual_context ([type]): [description]
            seq_lengths ([type], optional): [description]. Defaults to None.
            device (str, optional): [description]. Defaults to 'cpu'.

        Returns:
            [type]: [description]
        """
        #check batch size consistency (especially when using different gpus) and move list tensors to correct gpu
        assert utterances.shape[0] == len(history), 'Inconsistent batch size'
        assert utterances.shape[0] == len(actions), 'Inconsistent batch size'
        assert utterances.shape[0] == len(attributes), 'Inconsistent batch size'
        assert utterances.shape[0] == len(focus_items), 'Inconsistent batch size'
        assert utterances.shape[0] == candidates_pool.shape[0], 'Inconsistent batch size'
        curr_device = utterances.device
        for idx, _ in enumerate(history):
            if len(history[idx]):
                history[idx] = history[idx].to(curr_device)
        for idx, _ in enumerate(focus_items):
            focus_items[idx][0] = focus_items[idx][0].to(curr_device)
            focus_items[idx][1] = focus_items[idx][1].to(curr_device)

        # todo use attributes to enforce the user utterance?
        # u_t shape [BATCH_SIZE x 2MEMORY_HIDDEN_SIZE]
        embedded_utt_tensor = self.word_embeddings_layer(utterances)
        u_t = self.sentence_encoder(embedded_utt_tensor, seq_lengths)

        #h_t shape h_t[<sample_in_batch>].shape = [N_TURNSx300] or [] if no history
        embedded_hist_tensor = []
        for history_sample in history:
            if not len(history_sample):
                embedded_hist_tensor.append([])
            else:
                embedded_hist_tensor.append(self.word_embeddings_layer(history_sample))
        h_t = []
        for h_embedding in embedded_hist_tensor:
            if not len(h_embedding):
                h_t.append([])
            else:
                h_t.append(self.sentence_encoder(h_embedding))

        #embedded_vis_tensor shape v_t[<sample_in_batch>][0/1].shape = [N_FIELDSx300]
        #v_t shape [<sample_in_batch>]x300
        #v_t_tilde contains contextual embedding for each item in the batch. Shape [Bx300]
        v_t = self.encode_v_context(focus_items)
        v_t_tilde = torch.stack([self.triton_attention(utt, v[0], v[1]) for utt, v in zip(u_t, v_t)])

        # compute history context
        h_t_tilde = []
        for idx in range(u_t.shape[0]):
            if not len(h_t[idx]):
                h_t_tilde.append(u_t[idx])
            else:
                h_t_tilde.append(self.memory_net(u_t[idx], h_t[idx]))
        h_t_tilde = torch.stack(h_t_tilde)

        #encode candidates
        candidates_emb = self.word_embeddings_layer(candidates_pool)
        encoded_candidates = torch.stack([self.sentence_encoder(candidates) for candidates in candidates_emb])

        #try the fusion with a fnn also
        turns_repr = v_t_tilde + h_t_tilde
        out = self.out_layer(turns_repr, encoded_candidates)
        return out


    def encode_v_context(self, focus_images):
        v_batch = []
        for item in focus_images:
            k_ht = self.sentence_encoder(self.word_embeddings_layer(item[0]))
            v_ht = self.sentence_encoder(self.word_embeddings_layer(item[1]))
            v_batch.append([k_ht, v_ht])
        return v_batch


    def collate_fn(self, batch):
        """
        This method prepares the batch for the LSTM: padding + preparation for pack_padded_sequence

        Args:
            batch (tuple): tuple of element returned by the Dataset.__getitem__()

        Returns:
            dial_ids (list): list of dialogue ids
            turns (list): list of dialogue turn numbers
            seq_tensor (torch.LongTensor): tensor with BxMAX_SEQ_LEN containing padded sequences of user transcript sorted by descending effective lengths
            seq_lenghts: tensor with shape B containing the effective length of the correspondant transcript sequence
            actions (torch.Longtensor): tensor with B shape containing target actions
            attributes (torch.Longtensor): tensor with Bx33 shape containing attributes one-hot vectors, one for each sample.
        """
        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        transcripts = [torch.tensor(item[2]) for item in batch]
        history = [item[3] for item in batch]
        actions = [item[4] for item in batch]
        attributes = [item[5] for item in batch]
        focus_items = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        assert len(transcripts) == len(dial_ids), 'Batch sizes do not match'
        assert len(transcripts) == len(turns), 'Batch sizes do not match'
        assert len(transcripts) == len(history), 'Batch sizes do not match'
        assert len(transcripts) == len(actions), 'Batch sizes do not match'
        assert len(transcripts) == len(attributes), 'Batch sizes do not match'
        assert len(transcripts) == len(focus_items), 'Batch sizes do not match'
        assert len(transcripts) == len(responses_pool), 'Batch sizes do not match'

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        transcripts_lengths = torch.tensor(list(map(len, transcripts)), dtype=torch.long)
        transcripts_tensor = torch.zeros((len(transcripts), transcripts_lengths.max()), dtype=torch.long)
        for idx, (seq, seqlen) in enumerate(zip(transcripts, transcripts_lengths)):
            transcripts_tensor[idx, :seqlen] = seq.clone().detach()

        #pad the history
        padded_history = []
        for history_sample in history:
            if not len(history_sample):
                padded_history.append(history_sample)
                continue
            history_lens = torch.tensor(list(map(len, history_sample)), dtype=torch.long)
            history_tensor = torch.zeros((len(history_sample), history_lens.max()), dtype=torch.long)
            for idx, (seq, seqlen) in enumerate(zip(history_sample, history_lens)):
                history_tensor[idx, :seqlen] = seq.clone().detach()
            padded_history.append(history_tensor)

        #pad the response candidates
        batch_lens = torch.tensor([list(map(len, pool_sample)) for pool_sample in responses_pool], dtype=torch.long)
        pools_tensor = torch.zeros((len(responses_pool), len(responses_pool[0]), batch_lens.max()), dtype=torch.long)
        for batch_idx, (pool_lens, pool_sample) in enumerate(zip(batch_lens, responses_pool)):       
            for pool_idx, (seq, seqlen) in enumerate(zip(pool_sample, pool_lens)):
                pools_tensor[batch_idx, pool_idx, :seqlen] = seq.clone().detach()

        #pad focus items
        padded_focus = []
        keys = [[datum[0] for datum in item] for item in focus_items]
        vals = [[datum[1] for datum in item] for item in focus_items]
        batch_klens = [list(map(len, key)) for key in keys]
        batch_vlens = [list(map(len, val)) for val in vals]
        klens_max, vlens_max = 0, 0
        for klens, vlens in zip(batch_klens, batch_vlens):
            curr_kmax = max(klens)
            curr_vmax = max(vlens)
            klens_max = curr_kmax if curr_kmax > klens_max else klens_max
            vlens_max = curr_vmax if curr_vmax > vlens_max else vlens_max

        for batch_idx, item in enumerate(focus_items):
            curr_keys = torch.zeros((len(item), klens_max), dtype=torch.long)
            curr_vals = torch.zeros((len(item), vlens_max), dtype=torch.long)
            for item_idx, (k, v) in enumerate(item):
                curr_keys[item_idx, :batch_klens[batch_idx][item_idx]] = k.clone().detach()
                curr_vals[item_idx, :batch_vlens[batch_idx][item_idx]] = v.clone().detach()
            padded_focus.append([curr_keys, curr_vals])

        # sort instances by sequence length in descending order and order targets to keep the correspondance
        transcripts_lengths, perm_idx = transcripts_lengths.sort(0, descending=True)
        transcripts_tensor = transcripts_tensor[perm_idx]
        pools_tensor = pools_tensor[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        sorted_dial_history = []
        sorted_actions = []
        sorted_attributes = []
        sorted_focus_items = []
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])
            sorted_dial_history.append(padded_history[idx])
            sorted_actions.append(actions[idx])
            sorted_attributes.append(attributes[idx])
            sorted_focus_items.append(padded_focus[idx])

        batch_dict = {}
        batch_dict['utterances'] = transcripts_tensor
        batch_dict['history'] = sorted_dial_history
        batch_dict['actions'] = sorted_actions
        batch_dict['attributes'] = sorted_attributes
        batch_dict['focus_items'] = sorted_focus_items
        batch_dict['seq_lengths'] = transcripts_lengths

        return sorted_dial_ids, sorted_dial_turns, batch_dict, pools_tensor


    def __str__(self):
        return super().__str__()



class CandidateAttentiveOutput(nn.Module):

    def __init__(self, in_features, hidden_size):
        super(CandidateAttentiveOutput, self).__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.query_encoder = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.candidate_encoder = nn.Linear(in_features=in_features, out_features=hidden_size)


    def forward(self, turns_repr, candidates):
        #query shape [B x hidden_size]
        #cand_enc shape [B x N_CANDIDATES x hidden_size]
        query = self.query_encoder(turns_repr)
        cand_enc = self.candidate_encoder(candidates)
        attn_logits = torch.bmm(query[:, None], torch.transpose(cand_enc, 1, 2))/math.sqrt(self.hidden_size)
        return attn_logits.squeeze(1)
        
        """
        out = []
        for turn_repr, c in zip(turns_repr, candidates):
            query = self.query_encoder(turn_repr)
            c_enc = self.candidate_encoder(c)
            attn_logits = torch.matmul(query, torch.transpose(c_enc, 0, 1))/math.sqrt(self.hidden_size)
            out.append(attn_logits)
        return torch.stack(out)
        """



class SentenceEncoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, bidirectional=False):
        super(SentenceEncoder, self).__init__()

        self.encoder = nn.LSTM(emb_dim,
                            hidden_size, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        in_features = 2*hidden_size if bidirectional else hidden_size
        self.mlp = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.dropout = nn.Dropout(p=.1)
        self.layerNorm = nn.LayerNorm(hidden_size)


    def forward(self, seq, seq_lengths=None):

        if seq_lengths is not None:
            # pack padded sequence
            input_seq = pack_padded_sequence(seq, seq_lengths, batch_first=True) #seq_lengths.cpu().numpy()
        else:
            input_seq = seq

        #to call every forward if DataParallel is used. Otherwise only once inside __init__()
        self.encoder.flatten_parameters()
        out_enc, (h_t, c_t) = self.encoder(input_seq)
        bidirectional_h_t = torch.cat((h_t[0], h_t[-1]), dim=-1)
        """unpack not needed. We don't use the output
        if seq_lengths is not None:
            # unpack padded sequence
            output, input_sizes = pad_packed_sequence(out1, batch_first=True)
        """
        
        mlp_out = self.mlp(bidirectional_h_t)
        out = self.layerNorm(self.dropout(mlp_out))

        return out



class MemoryNet(nn.Module):

    def __init__(self, emb_dim, memory_hidden_size):
        super(MemoryNet, self).__init__()

        self.memory_hidden_size = memory_hidden_size
        self.query_encoder = nn.Linear(in_features=emb_dim, out_features=memory_hidden_size)
        self.memory_encoder = nn.Linear(in_features=emb_dim, out_features=memory_hidden_size)
        self.dropout = nn.Dropout(p=.1)
        self.layerNorm = nn.LayerNorm(memory_hidden_size)


    def forward(self, input, context, device='cpu'):
        query = self.query_encoder(input)
        memory = self.memory_encoder(context) + self.get_positional_embeddings(context.shape[0], self.memory_hidden_size).to(context.device)

        attn_logits = torch.matmul(query, torch.transpose(memory, 0, 1))/math.sqrt(self.memory_hidden_size)
        attn_scores = F.softmax(attn_logits, -1)
        weighted_memory = torch.sum(attn_scores[:, None] * memory, dim=0)

        out = self.layerNorm(query + self.dropout(weighted_memory))
        return out


    def get_positional_embeddings(self, n_position, emb_dim):
        """Create positional embeddings (from "Attention Is All You Need", Vaswani et al. 2017)

        Args:
            n_position (int): number of elements in the sequence
            emb_dim (int): size of embeddings

        Returns:
            toch.FloatTensor: a positional embedding with shape [N_POSITION x EMB_DIM] 
        """
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (k // 2) / emb_dim) for k in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)



# Triton, trident? Not self attention! Triplet as input q, k, v belonging to different conceptual sets
class Triton(nn.Module):
    
    def __init__(self, emb_dim, d_k, d_v, d_f, n_heads, n_layers):
        super(Triton, self).__init__()

        assert n_layers >= 1, 'Not acceptable number of layers: {}'.format(n_layers)
        self.n_layers = n_layers
        #self.encoders = nn.ModuleList([TritonEncoder(emb_dim, d_k, d_v, n_heads) for _ in range(n_layers)])
        #encoders = [TritonEncoder(emb_dim, d_k, d_v, d_f, n_heads) for _ in range(n_layers)]
        #self.encoders = nn.Sequential(*encoders)
        #todo change to allow multiple layers. Problem: sequential take only 1 input, so pack inputs to a tuple.
        self.encoders = TritonEncoder(emb_dim, d_k, d_v, d_f, n_heads)


    def forward(self, ut, kt, vt):
        out = self.encoders(ut, kt, vt)
        return out


    def __str__(self):
        return super().__str__()



class TritonEncoder(nn.Module):


    def __init__(self, emb_dim, d_k, d_v, d_f, n_heads):
        super(TritonEncoder, self).__init__()

        self.multihead_attn = TritonMultiHeadCrossAttention(emb_dim, d_k, d_v, n_heads)
        self.dropout = nn.Dropout(p=.1)
        self.layerNorm = nn.LayerNorm(emb_dim)
        self.fnn = nn.Sequential(nn.Linear(in_features=emb_dim, out_features=d_f),
                                nn.ReLU(),
                                nn.Linear(in_features=d_f, out_features=emb_dim))


    def forward(self, u_t, k_t, v_t):
        multihead_out = self.multihead_attn(u_t, k_t, v_t)
        # residual connection is performed after the dropout and before normalization in (Vaswani et al.)
        norm_out = self.layerNorm(self.dropout(multihead_out))
        enc_out = self.fnn(norm_out)
        out = self.layerNorm(self.dropout(multihead_out))
        return out
  
    
    def __str__(self):
        return super().__str__()



class TritonMultiHeadCrossAttention(nn.Module):
    
    def __init__(self, emb_dim, d_k, d_v, n_heads):
        super(TritonMultiHeadCrossAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([TritonCrossAttentionHead(emb_dim, d_k, d_v) for _ in range(n_heads)])


    def forward(self, u_t, k_t, v_t):
        attn_outs = []
        heads_out = torch.cat([attn_head(u_t, k_t, v_t) for attn_head in self.attn_heads])
        return heads_out


    def __str__(self):
        return super().__str__()



class TritonCrossAttentionHead(nn.Module):

    def __init__(self, emb_dim, d_k, d_v):
        super(TritonCrossAttentionHead, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(emb_dim, d_k)
        self.K = nn.Linear(emb_dim, d_k)
        self.V = nn.Linear(emb_dim, d_v)

    def forward(self, u_t, k_t, v_t):
        query = self.Q(u_t)
        key = self.K(k_t)
        value = self.V(v_t)

        attn_logits = torch.matmul(query, torch.transpose(key, 0, 1))/ math.sqrt(self.d_k)
        attn_scores = F.softmax(attn_logits, -1)
        out = torch.sum(attn_scores[:, None] * value, dim=0)
        return out



    def __str__(self):
        return super().__str__()