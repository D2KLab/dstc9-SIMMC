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


    def __init__(self, word_embeddings_path, word2id, num_actions, num_attrs, 
                        pad_token, unk_token, seed, OOV_corrections=False, freeze_embeddings=False):
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
        self.actions_linear = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
        self.attrs_linear = nn.Linear(in_features=self.hidden_size, out_features=num_attrs)


    def forward(self, utterances, seq_lengths=None, device='cpu'):
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
        
        out1, (h_t, c_t) = self.lstm(packed_input)

        """unpack not needed. We don't use the output
        if seq_lengths is not None:
            # unpack padded sequence
            output, input_sizes = pad_packed_sequence(out1, batch_first=True)
        """
        h_t = self.dropout(h_t)
        # h_t has shape NUM_DIRxBxHIDDEN_SIZE
        actions_logits = self.actions_linear(h_t[0])
        attrs_logits = self.attrs_linear(h_t[0])

        #out2.shape = BxNUM_LABELS
        actions_probs = F.softmax(actions_logits, dim=-1)
        attrs_probs = torch.sigmoid(attrs_logits)
        return actions_logits, attrs_logits, actions_probs, attrs_probs


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
        transcripts = [torch.tensor(item[2]) for item in batch]
        actions = torch.tensor([item[4] for item in batch])
        attributes = torch.stack([item[5] for item in batch])

        assert len(transcripts) == len(dial_ids), 'Batch sizes do not match'
        assert len(transcripts) == len(turns), 'Batch sizes do not match'
        assert len(transcripts) == actions.shape[0], 'Batch sizes do not match'
        assert len(transcripts) == attributes.shape[0], 'Batch sizes do not match'
        
        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        transcripts_lengths = torch.tensor(list(map(len, transcripts)), dtype=torch.long)
        transcripts_tensor = torch.zeros((len(transcripts), transcripts_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(transcripts, transcripts_lengths)):
            transcripts_tensor[idx, :seqlen] = seq.clone().detach()

        # sort instances by sequence length in descending order
        transcripts_lengths, perm_idx = transcripts_lengths.sort(0, descending=True)
        transcripts_tensor = transcripts_tensor[perm_idx]
        actions = actions[perm_idx]
        attributes = attributes[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])

        batch_dict = {}
        batch_dict['utterances'] = transcripts_tensor
        batch_dict['seq_lengths'] = transcripts_lengths

        # seq_lengths is used to create a pack_padded_sequence
        return sorted_dial_ids, sorted_dial_turns, batch_dict, actions, attributes


    def __str__(self):
        return super().__str__()
