import pdb

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import WordEmbeddingBasedNetwork


class BlindStatefulLSTM(WordEmbeddingBasedNetwork):
    """

    Args:
        nn ([type]): [description]
    """
    _HIDDEN_SIZE = 300

    def __init__(self, embedding_path, word2id, num_actions, num_args, pad_token, unk_token, seed, OOV_corrections):
        
        torch.manual_seed(seed)
        super(BlindStatefulLSTM, self).__init__(embedding_path, word2id, pad_token, unk_token, OOV_corrections)

        self.memory_hidden_size = self._HIDDEN_SIZE

        #history encoder
        self.history_encoder = nn.LSTM(self.embedding_size, self.memory_hidden_size, batch_first=True, bidirectional=True)
        self.utterance_encoder = nn.LSTM(self.embedding_size, self.memory_hidden_size, batch_first=True, bidirectional=True)
        #utterance encoder
        #cross-history attention
        #utterance self-attention
        #filtered cross-history attention
        #context encoder
        #action out layer
        #args encoder
        #args out layer
        pass


    def forward(self, batch, history, seq_lengths=None, device='cpu'):

        # memory bank is a list of BATCH_SIZE tensors, each of them having shape N_TURNSx2MEMORY_HIDDEN_SIZE
        memory_bank = self.encode_history(history, device)
        # u_t shape [BATCH_SIZE x 2MEMORY_HIDDEN_SIZE]
        u_t = self.encode_utterance(batch, seq_lengths)

        assert u_t.shape[0] == len(memory_bank), 'Batch size and memory size does not match'
        pdb.set_trace()


    def encode_history(self, history, device):
        #todo turn embedding based on previous turns (hierarchical recurrent encoder - HRE)
        encoded_batch_history = []
        for dial in history:
            hiddens = []
            for turn in dial:
                emb = self.embedding_layer(turn.unsqueeze(0).to(device))
                # h_t.shape = [num_directions x 1 x HIDDEN_SIZE]
                out, (h_t, c_t) = self.history_encoder(emb)
                bidirectional_h_t = torch.cat((h_t[0], h_t[-1]), dim=-1)
                hiddens.append(bidirectional_h_t.squeeze(0))
            if len(hiddens) > 0:
                encoded_batch_history.append(torch.stack(hiddens))
            else:
                encoded_batch_history.append([])
        return encoded_batch_history


    def encode_utterance(self, batch, seq_lengths):
        embedded_seq_tensor = self.embedding_layer(batch)
        if seq_lengths is not None:
            # pack padded sequence
            packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        
        out1, (h_t, c_t) = self.utterance_encoder(packed_input)
        bidirectional_h_t = torch.cat((h_t[0], h_t[-1]), dim=-1)

        """unpack not needed. We don't use the output
        if seq_lengths is not None:
            # unpack padded sequence
            output, input_sizes = pad_packed_sequence(out1, batch_first=True)
        """
        return bidirectional_h_t


    def collate_fn(self, batch):
        """This method prepares the batch for the LSTM: padding + preparation for pack_padded_sequence

        Args:
            batch (tuple): tuple of element returned by the Dataset.__getitem__()

        Returns:
            dial_ids (list): list of dialogue ids
            turns (list): list of dialogue turn numbers
            seq_tensor (torch.LongTensor): tensor with BxMAX_SEQ_LEN containing padded sequences of user transcript sorted by descending effective lengths
            seq_lenghts: tensor with shape B containing the effective length of the correspondant transcript sequence
            actions (torch.Longtensor): tensor with B shape containing target actions
            arguments (torch.Longtensor): tensor with Bx33 shape containing arguments one-hot vectors, one for each sample.
        """
        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        history = [item[3] for item in batch]
        actions = torch.tensor([item[4] for item in batch])
        arguments = torch.tensor([item[5] for item in batch])

        # words to ids for the history
        history_seq_ids = []
        for turn, item in zip(turns, history):
            assert len(item) == turn, 'Number of turns does not match history length'
            curr_turn_ids = []
            for t in range(turn):
                concat_sentences = item[t][0] + ' ' + item[t][1] #? separator token
                ids = torch.tensor([self.word2id[word] for word in concat_sentences.split()])
                curr_turn_ids.append(ids)
            history_seq_ids.append(curr_turn_ids)

        # words to ids for the current utterance
        utterance_seq_ids = []
        for item in batch:
            curr_seq = []
            for word in item[2].split():
                if word in self.word2id:
                    curr_seq.append(self.word2id[word])
                else:
                    curr_seq.append(self.word2id[self.unk_token])
            utterance_seq_ids.append(curr_seq)
        
        seq_lengths = torch.tensor(list(map(len, utterance_seq_ids)), dtype=torch.long)
        seq_tensor = torch.zeros((len(utterance_seq_ids), seq_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(utterance_seq_ids, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)

        # sort instances by sequence length in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        seq_tensor = seq_tensor[perm_idx]
        actions = actions[perm_idx]
        arguments = arguments[perm_idx]

        # seq_lengths is used to create a pack_padded_sequence
        return dial_ids, turns, seq_tensor, seq_lengths, history_seq_ids, actions, arguments


    def __str__(self):
        return super().__str__()
