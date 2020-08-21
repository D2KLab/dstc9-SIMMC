import pdb

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .embednets import WordEmbeddingNetwork, ItemEmbeddingNetwork

import numpy as np



def get_positional_embeddings(n_position, emb_dim):
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


class MMStatefulLSTM(nn.Module):
    """

    Args:
        nn ([type]): [description]
    """
    _HIDDEN_SIZE = 300

    def __init__(self, word_embeddings_path, word2id, item_embeddings_path, 
                                pad_token, unk_token, seed, OOV_corrections):
        torch.manual_seed(seed)
        super(MMStatefulLSTM, self).__init__()

        self.memory_hidden_size = self._HIDDEN_SIZE

        # NETWORK
        self.item_embeddings_layer = ItemEmbeddingNetwork(item_embeddings_path)
        self.word_embeddings_layer = WordEmbeddingNetwork(word_embeddings_path=word_embeddings_path, 
                                                            word2id=word2id, 
                                                            pad_token=pad_token, 
                                                            unk_token=unk_token)
        self.utterance_encoder = nn.LSTM(self.word_embeddings_layer.embedding_dim, 
                                        self.memory_hidden_size, 
                                        batch_first=True, 
                                        bidirectional=True)
        self.utterance_dropout = nn.Dropout(p=.5)

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


    def forward(self, utterances, history, visual_context, seq_lengths=None, device='cpu'):

        # u_t shape [BATCH_SIZE x 2MEMORY_HIDDEN_SIZE]
        u_t = self.encode_utterance(utterances, seq_lengths)
        focus, focus_history = self.encode_visual(visual_context, device)
        # u_t shape [BATCH_SIZE x 2MEMORY_HIDDEN_SIZE]
        focus_t = self.item_dim_reduction(focus)

        # separate single from multi-turn
        single_turns = []
        single_turns_v_focus = []
        single_turns_pos = set()
        multi_turns = []
        multi_turns_v_focus = []
        multi_turns_history = []
        multi_turns_v_history = []
        for dial_idx, history_item in enumerate(history):
            if len(history_item) == 0:
                single_turns_pos.add(dial_idx)
                single_turns.append(u_t[dial_idx])
                single_turns_v_focus.append(focus_t[dial_idx])
            else:
                multi_turns.append(u_t[dial_idx])
                multi_turns_history.append(history[dial_idx])
                multi_turns_v_focus.append(focus[dial_idx])
                multi_turns_v_history.append(focus_history[dial_idx])

        if len(single_turns):
            # concat u_t with correspondent v_t
            single_u_t = torch.stack(single_turns)
            single_v_t = torch.stack(single_turns_v_focus)
            #pos = list(single_turns_pos)
            single_u_v_concat = torch.cat((single_u_t, single_v_t), dim=-1)
            # compute output for single turn dialogues
            #act_out1 = self.singleturn_actions_outlayer(single_u_v_concat)
            #args_out1 = self.singleturn_args_outlayer(single_u_v_concat)

        if len(multi_turns):
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

            #act_out2 = self.multiturn_actions_outlayer(c_t_tilde1)
            #args_out2 = self.multiturn_args_outlayer(c_t_tilde2)

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


    def encode_utterance(self, batch, seq_lengths):
        embedded_seq_tensor = self.word_embeddings_layer(batch)
        if seq_lengths is not None:
            # pack padded sequence
            packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        
        out1, (h_t, c_t) = self.utterance_encoder(packed_input)
        bidirectional_h_t = torch.cat((h_t[0], h_t[-1]), dim=-1)
        bidirectional_h_t = self.utterance_dropout(bidirectional_h_t)

        """unpack not needed. We don't use the output
        if seq_lengths is not None:
            # unpack padded sequence
            output, input_sizes = pad_packed_sequence(out1, batch_first=True)
        """
        return bidirectional_h_t


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
        visual_context = {'focus': [], 'history': []}
        visual_context['focus'] = [item[6] for item in batch]
        visual_context['history'] = [item[7] for item in batch]
        responses_pool = [item[7] for item in batch]

        assert len(transcripts) == len(dial_ids), 'Batch sizes do not match'
        assert len(transcripts) == len(turns), 'Batch sizes do not match'
        assert len(transcripts) == len(history), 'Batch sizes do not match'
        assert len(transcripts) == len(actions), 'Batch sizes do not match'
        assert len(transcripts) == len(attributes), 'Batch sizes do not match'
        assert len(transcripts) == len(visual_context['focus']), 'Batch sizes do not match'
        assert len(transcripts) == len(visual_context['history']), 'Batch sizes do not match'
        assert len(transcripts) == len(responses_pool), 'Batch sizes do not match'

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        transcripts_lengths = torch.tensor(list(map(len, transcripts)), dtype=torch.long)
        transcripts_tensor = torch.zeros((len(transcripts), transcripts_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(transcripts, transcripts_lengths)):
            transcripts_tensor[idx, :seqlen] = seq.clone().detach()

        # sort instances by sequence length in descending order
        transcripts_lengths, perm_idx = transcripts_lengths.sort(0, descending=True)
        transcripts_tensor = transcripts_tensor[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        sorted_dial_history = []
        sorted_responses = []
        sorted_actions = []
        sorted_attributes = []
        sorted_visual_context = {'focus': [], 'history': []}
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])
            sorted_dial_history.append(history[idx])
            sorted_actions.append(actions[idx])
            sorted_attributes.append(attributes[idx])
            sorted_visual_context['focus'].append(visual_context['focus'][idx])
            sorted_visual_context['history'].append(visual_context['history'][idx])
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
