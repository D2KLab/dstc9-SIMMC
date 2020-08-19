import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .embednets import WordEmbeddingNetwork
        

class BlindStatelessLSTM(nn.Module):
    """Implementation of a blind and stateless LSTM for action prediction. It approximates the probability distribution:

            P(a_t | U_t)
    
        Where a_t is the action and U_t the user utterance.

    Args:
        torch (WordEmbeddingBasedNetwork): inherits from WordEmbeddingBasedNetwork

    Attributes:
        self.corrections (dict): Mapping from dataset word to its corrections (the corrections is included in the vocabulary)
    """

    _HIDDEN_SIZE = 600


    def __init__(self, word_embeddings_path, word2id, pad_token, unk_token, 
                    seed, OOV_corrections=False, freeze_embeddings=False):
        """
        Glove download: https://nlp.stanford.edu/projects/glove/

        Args:
            embedding_path ([type]): [description]
        """

        torch.manual_seed(seed)
        super(BlindStatelessLSTM, self).__init__()

        self.hidden_size = self._HIDDEN_SIZE

        self.word_embeddings_layer = WordEmbeddingNetwork(word_embeddings_path=word_embeddings_path, 
                                                            word2id=word2id, 
                                                            pad_token=pad_token, 
                                                            unk_token=unk_token,
                                                            OOV_corrections=OOV_corrections,
                                                            freeze=freeze_embeddings)

        self.lstm = nn.LSTM(self.word_embeddings_layer.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.matching_layer = nn.Linear(in_features=2*self.hidden_size, out_features=1)


    def forward(self, utterances, actions, attributes, candidates_pool, seq_lengths=None, device='cpu'):
        """Forward pass for BlindStatelessLSTM

        Args:
            batch (torch.LongTensor): Tensor containing the batch (shape=BxMAX_SEQ_LEN)
            seq_lengths (torch.LongTensor, optional): Effective lengths (no pad) of each sequence in the batch. If given the pack_padded__sequence are used.
                                                        The shape is B. Defaults to None.
        """
        embedded_seq_tensor = self.word_embeddings_layer(utterances)
        if seq_lengths is not None:
            # pack padded sequence
            packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        # h_t has shape NUM_DIRxBxHIDDEN_SIZE
        out, (h_t, c_t) = self.lstm(packed_input)

        """unpack not needed. We don't use the output
        if seq_lengths is not None:
            # unpack padded sequence
            output, input_sizes = pad_packed_sequence(out1, batch_first=True)
        """
        utterance_hiddens = self.dropout(h_t.squeeze(0))

        # tensors_pool has shape BATCH_SIZEx100xHIDDEN_SIZE
        tensors_pool = self.encode_pool(candidates_pool, device)

        # build pairs (utterance, candidate[i]) for computing similarity
        assert utterance_hiddens.shape[0] == tensors_pool.shape[0], 'Problem with first dimension'
        matching_pairs = []
        for utterance, candidate in zip(utterance_hiddens, tensors_pool):
            matching_pairs.append(torch.cat((utterance.expand(candidate.shape[0], -1), candidate), dim=-1))
        matching_pairs = torch.stack(matching_pairs)

        # matching_pairs has shape Bx100x2*HIDDEN_SIZE
        matching_logits = []
        for pair in matching_pairs:
            matching_logits.append(self.matching_layer(pair))
        matching_logits = torch.stack(matching_logits).squeeze(-1)
        matching_scores = torch.sigmoid(matching_logits)

        return matching_logits, matching_scores


    def encode_pool(self, candidates_pool, device):
        tensors_pool = []
        for pool in candidates_pool:
            curr_hiddens = []
            for candidate in pool:
                embeddings = self.word_embeddings_layer(candidate.to(device))
                _, (h_t, _) = self.lstm(embeddings.unsqueeze(0))
                curr_hiddens.append(h_t.squeeze(0).squeeze(0))
            tensors_pool.append(torch.stack(curr_hiddens))
        tensors_pool = torch.stack(tensors_pool)

        return tensors_pool
     

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
        transcripts =  [item[2] for item in batch]
        actions = [item[5] for item in batch]
        attributes = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        # words to ids for the current utterance
        seq_ids = []
        word2id = self.word_embeddings_layer.word2id
        unk_token = self.word_embeddings_layer.unk_token
        for transcript in transcripts:
            curr_seq = []
            for word in transcript.split():
                if word in word2id:
                    curr_seq.append(word2id[word])
                else:
                    curr_seq.append(word2id[unk_token])
            seq_ids.append(torch.tensor(curr_seq, dtype=torch.long))

        # convert response candidates to word ids
        resp_ids = []
        for resps in responses_pool:
            curr_candidate = []
            for resp in resps:
                curr_seq = []
                for word in resp.split():
                    if word in word2id:
                        curr_seq.append(word2id[word])
                    else:
                        curr_seq.append(word2id[unk_token])
                curr_candidate.append(torch.tensor(curr_seq, dtype=torch.long))
            resp_ids.append(curr_candidate)

        #convert actions and attributes to word ids
        act_ids = []
        for act in actions:
            curr_seq = []
            for word in act.split():
                if word in word2id:
                    curr_seq.append(word2id[word])
                else:
                    curr_seq.append(word2id[unk_token])
            act_ids.append(torch.tensor(curr_seq, dtype=torch.long))
        attr_ids = []
        for attrs in attributes:
            curr_attributes = []
            for attr in attrs:
                curr_seq = []
                for word in attr.split():
                    if word in word2id:
                        curr_seq.append(word2id[word])
                    else:
                        curr_seq.append(word2id[unk_token])
                curr_attributes.append(torch.tensor(curr_seq, dtype=torch.long))
            attr_ids.append(curr_attributes)

        assert len(seq_ids) == len(dial_ids), 'Batch sizes do not match'
        assert len(seq_ids) == len(turns), 'Batch sizes do not match'
        assert len(seq_ids) == len(resp_ids), 'Batch sizes do not match'
        assert len(seq_ids) == len(act_ids), 'Batch sizes do not match'
        assert len(seq_ids) == len(attr_ids), 'Batch sizes do not match'

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        seq_lengths = torch.tensor(list(map(len, seq_ids)), dtype=torch.long)
        seq_tensor = torch.zeros((len(seq_ids), seq_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(seq_ids, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)

        # sort instances by sequence length in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        sorted_actions = []
        sorted_attributes = []
        sorted_responses = []
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])
            sorted_actions.append(act_ids[idx])
            sorted_attributes.append(attr_ids[idx])
            sorted_responses.append(resp_ids[idx])
        batch_dict = {}
        batch_dict['utterances'] = seq_tensor
        batch_dict['seq_lengths'] = seq_lengths
        batch_dict['actions'] = sorted_actions
        batch_dict['attributes'] = sorted_attributes

        # seq_lengths is used to create a pack_padded_sequence
        return sorted_dial_ids, sorted_dial_turns, batch_dict, sorted_responses


    def __str__(self):
        return super().__str__()
