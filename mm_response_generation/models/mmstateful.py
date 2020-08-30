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

    def __init__(self, word_embeddings_path, word2id, pad_token, unk_token, seed, OOV_corrections):
        torch.manual_seed(seed)
        super(MMStatefulLSTM, self).__init__()

        self.memory_hidden_size = self._HIDDEN_SIZE
        n_encoders, n_heads = 1, 2

        # MAE
        # MN
        # ATTRIBUTES-ITEM attention
        # Candidate ENCODER
        # Matching network

        # NETWORK
        
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
        
        """
        self.item_dim_reduction = nn.Linear(in_features=self.item_embeddings_layer.embedding_dim,
                                            out_features=2*self.memory_hidden_size)

        self.utterance_memory_attn = nn.Sequential(nn.Linear(4*self.memory_hidden_size, 4*self.memory_hidden_size),
                                                    nn.Tanh(),
                                                    nn.Dropout(p=.5),
                                                    nn.Linear(4*self.memory_hidden_size, 1),
                                                    nn.Dropout(p=.5)) #todo introduce layerNorm
        self.linear_act_post_attn = nn.Sequential(nn.Linear(4*self.memory_hidden_size, 2*self.memory_hidden_size),
                                                    nn.Dropout(p=.5),
                                                    nn.ReLU())
        self.linear_args_post_attn = nn.Sequential(nn.Linear(4*self.memory_hidden_size, 2*self.memory_hidden_size),
                                                    nn.Dropout(p=.5),
                                                    nn.ReLU())

        #self.multiturn_actions_outlayer = nn.Linear(in_features=2*self.memory_hidden_size, out_features=self.num_actions)
        #self.multiturn_args_outlayer = nn.Linear(in_features=2*self.memory_hidden_size, out_features=self.num_attrs)
        
        #self.singleturn_actions_outlayer = nn.Linear(in_features=4*self.memory_hidden_size, out_features=self.num_actions)
        #self.singleturn_args_outlayer = nn.Linear(in_features=4*self.memory_hidden_size, out_features=self.num_attrs)
        """


    def forward(self, utterances, history, actions, attributes, visual_context, 
                candidates_pool, seq_lengths=None, device='cpu'):
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
                embedded_hist_tensor.append(self.word_embeddings_layer(history_sample.to(device)))
        h_t = []
        for h_embedding in embedded_hist_tensor:
            if not len(h_embedding):
                h_t.append([])
            else:
                h_t.append(self.sentence_encoder(h_embedding))

        #embedded_vis_tensor shape v_t[<sample_in_batch>][<item_n>][0/1].shape = [N_FIELDSx300]
        #v_t shape [<sample_in_batch>][<n_items>]x300
        embedded_vis_tensor = self.embed_v_context(visual_context, device)
        v_t = []
        for utt, v in zip(u_t, embedded_vis_tensor):
            curr_batch = torch.stack([self.triton_attention(utt, item[0], item[1]) for item in v])
            v_t.append(curr_batch)

        assert u_t.shape[0] == len(h_t), 'Inconsistent batch size'
        assert u_t.shape[0] == len(v_t), 'Inconsistent batch size'

        h_t_tilde = []
        for idx in range(u_t.shape[0]):
            if not len(h_t[idx]):
                h_t_tilde.append(u_t[idx])
            else:
                h_t_tilde.append(self.memory_net(u_t[idx], h_t[idx], device=device))
        h_t_tilde = torch.stack(h_t_tilde)
        pdb.set_trace()
            



        """
        # separate single from multi-turn
        st_samples = []
        st_visual = []
        st_pos = set()
        mt_samples = []
        mt_visual = []
        mt_history = []
        for batch_idx, history_item in enumerate(history):
            if len(history_item) == 0:
                st_pos.add(batch_idx)
                st_samples.append(u_t[batch_idx])
                st_visual.append(items_contextual[batch_idx])
            else:
                mt_samples.append(u_t[batch_idx])
                mt_history.append(history[batch_idx])
                mt_visual.append(items_contextual[batch_idx])

        """

        """
        if len(st_samples):
            # concat u_t with correspondent v_t
            single_u_t = torch.stack(single_turns)
            single_v_t = torch.stack(single_turns_v_focus)
            #pos = list(single_turns_pos)
            single_u_v_concat = torch.cat((single_u_t, single_v_t), dim=-1)
            # compute output for single turn dialogues
            act_out1 = self.singleturn_actions_outlayer(single_u_v_concat)
            args_out1 = self.singleturn_args_outlayer(single_u_v_concat)

        if len(mt_samples):
            multi_u_t = torch.stack(multi_turns)
            multi_v_t = torch.stack(multi_turns_v_focus)
            multi_v_t = self.item_dim_reduction(multi_v_t)

            # memory bank is a list of BATCH_SIZE tensors, each of them having shape N_TURNSx2MEMORY_HIDDEN_SIZE
            lang_memory = self.encode_history(multi_turns_history, device)        
            assert len(multi_turns) == len(lang_memory), 'Wrong memory size'

            #visual_memory = self.encode_visual_history(multi_turns_v_history, device) #todo visual memory
            #assert len(multi_turns) == len(visual_memory), 'Wrong memory size'

            # c_t shape [MULTI_TURNS_SET_SIZE x MEMORY_HIDDEN_SIZE]
            c_t = self.attention_over_memory(multi_u_t, lang_memory)
            mm_c_t = c_t * multi_v_t

            #? Hadamard product between c_t and u_t? It is simply "tensor1 * tensor2"
            ut_ct_concat = torch.cat((multi_u_t, mm_c_t), dim=-1)
            c_t_tilde1 = self.linear_act_post_attn(ut_ct_concat)

            ut_ct1_concat = torch.cat((multi_u_t, c_t_tilde1), dim=-1)
            c_t_tilde2 = self.linear_args_post_attn(ut_ct1_concat)

            act_out2 = self.multiturn_actions_outlayer(c_t_tilde1)
            args_out2 = self.multiturn_args_outlayer(c_t_tilde2)

        # recompose the output
        act_out = []
        args_out = []
        pos1 = 0
        pos2 = 0
        for idx in range(utterances.shape[0]):
            if idx in single_turns_pos:
                act_out.append(act_out1[pos1])
                args_out.append(args_out1[pos1])
                pos1 += 1
            else:
                act_out.append(act_out2[pos2])
                args_out.append(args_out2[pos2])
                pos2 += 1
        act_out = torch.stack(act_out)
        args_out = torch.stack(args_out)

        act_probs = F.softmax(act_out, dim=-1)
        args_probs = torch.sigmoid(args_out)
        return act_out, args_out, act_probs, args_probs
        """


    def embed_v_context(self, visual_context, device):
        
        v_batch = []
        for batch_sample in visual_context:
            curr_items = []
            for item in batch_sample:
                curr_item_fields = []
                curr_item_values = []
                for (field, values) in item:
                    curr_item_fields.append(self.word_embeddings_layer(field.to(device)).mean(0))
                    values_emb = []
                    for v in values:
                        #an unknown bug during preprocessing causes sometimes to have 0-d value tensors
                        correct_v = v.unsqueeze(0) if v.shape == torch.Size([]) else v
                        values_emb.append(self.word_embeddings_layer(correct_v.to(device)).mean(0)) #averaging values with multiple tokens
                    curr_item_values.append(torch.stack(values_emb).mean(0))
                curr_items.append((torch.stack(curr_item_fields), torch.stack(curr_item_values)))
            v_batch.append(curr_items)
        return v_batch


    def encode_visual(self, visual_context, device):
        focus = self.item_embeddings_layer(visual_context['focus'].to(device))
        history = []
        for history_item in visual_context['history']:
            if not len(history_item):
                history.append([])
            else:
                history.append(self.item_embeddings_layer(history_item.to(device)))
        return focus, history


    def encode_history(self, history, device):
        #todo turn embedding based on previous turns (hierarchical recurrent encoder - HRE)
        encoded_batch_history = []
        for dial in history:
            hiddens = []
            positional_embeddings = get_positional_embeddings(len(dial), 2*self.memory_hidden_size).to(device)
            assert positional_embeddings.shape[0] == len(dial)
            for turn, pos_emb in zip(dial, positional_embeddings):
                emb = self.word_embeddings_layer(turn.unsqueeze(0).to(device))
                # h_t.shape = [num_directions x 1 x HIDDEN_SIZE]
                out, (h_t, c_t) = self.utterance_encoder(emb)
                bidirectional_h_t = torch.cat((h_t[0], h_t[-1]), dim=-1)
                pos_bidirectional_h_t = bidirectional_h_t+pos_emb
                hiddens.append(pos_bidirectional_h_t.squeeze(0))
            assert len(hiddens) > 0, 'Impossible to encode history for single turn instance'
            encoded_batch_history.append(torch.stack(hiddens))
        return encoded_batch_history


    def encode_visual_history(self, history, device):
        encoded_batch_history = []
        for dial in history:
            hiddens = []
            for turn in dial:
                m_t = self.item_dim_reduction(turn.unsqueeze(0).to(device))
                hiddens.append(m_t.squeeze(0))
            assert len(hiddens) > 0, 'Impossible to encode history for single turn instance'
            encoded_batch_history.append(torch.stack(hiddens))
        return encoded_batch_history


    def attention_over_memory(self, u_t, memory_bank):
        # input for attention layer consists of pairs (utterance_j, memory_j_i), for each j, i
        attn_input_list = []
        for dial_idx, dial_mem in enumerate(memory_bank):
            num_turns = dial_mem.shape[0]
            #utterance_mem_concat shape N_TURNS x (utterance_size + memory_size)
            utterance_mem_concat = torch.cat((u_t[dial_idx].expand(num_turns, -1), dial_mem), dim=-1)
            attn_input_list.append(utterance_mem_concat)

        scores = []
        for idx, input_tensor in enumerate(attn_input_list):
            curr_out = self.utterance_memory_attn(input_tensor)
            scores.append(curr_out)

        attn_weights = []
        for score in scores:
            attn = F.softmax(score, dim=0)
            attn_weights.append(attn)

        assert len(attn_weights) == len(memory_bank), 'Memory size and attention weights do not match'
        weighted_sum_list = []
        for attn, mem in zip(attn_weights, memory_bank):
            weighted_mem = attn * mem
            weighted_sum = torch.sum(weighted_mem, dim=0)
            weighted_sum_list.append(weighted_sum)
        weighted_sum = torch.stack(weighted_sum_list)

        return weighted_sum


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
        visual_context = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        assert len(transcripts) == len(dial_ids), 'Batch sizes do not match'
        assert len(transcripts) == len(turns), 'Batch sizes do not match'
        assert len(transcripts) == len(history), 'Batch sizes do not match'
        assert len(transcripts) == len(actions), 'Batch sizes do not match'
        assert len(transcripts) == len(attributes), 'Batch sizes do not match'
        assert len(transcripts) == len(visual_context), 'Batch sizes do not match'
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

        # sort instances by sequence length in descending order and order targets to keep the correspondance
        transcripts_lengths, perm_idx = transcripts_lengths.sort(0, descending=True)
        transcripts_tensor = transcripts_tensor[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        sorted_dial_history = []
        sorted_responses = []
        sorted_actions = []
        sorted_attributes = []
        sorted_visual_context = []
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])
            sorted_dial_history.append(padded_history[idx])
            sorted_actions.append(actions[idx])
            sorted_attributes.append(attributes[idx])
            sorted_visual_context.append(visual_context[idx])
            sorted_responses.append(responses_pool[idx])

        batch_dict = {}
        batch_dict['utterances'] = transcripts_tensor
        batch_dict['history'] = sorted_dial_history
        batch_dict['actions'] = sorted_actions
        batch_dict['attributes'] = sorted_attributes
        batch_dict['visual_context'] = sorted_visual_context
        batch_dict['seq_lengths'] = transcripts_lengths

        return sorted_dial_ids, sorted_dial_turns, batch_dict, sorted_responses


    def __str__(self):
        return super().__str__()



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


    def forward(self, input, seq_lengths=None):

        if seq_lengths is not None:
            # pack padded sequence
            input_seq = pack_padded_sequence(input, seq_lengths.cpu().numpy(), batch_first=True)
        else:
            input_seq = input

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
        memory = self.memory_encoder(context) + self.get_positional_embeddings(context.shape[0], self.memory_hidden_size).to(device)

        attn_logits = torch.matmul(query, torch.transpose(memory, 0, 1))
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