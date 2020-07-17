import copy
import pdb

import numpy as np
import torch
from spellchecker import SpellChecker
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class BlindStatelessLSTM(nn.Module):
    """Implementation of a blind and stateless LSTM for action prediction. It approximates the probability distribution:

            P(a_t | U_t)
    
        Where a_t is the action and U_t the user utterance.

    Args:
        torch (torch.nn.Module): inherits from torch.nn.Module

    Attributes:
        self.corrections (dict): Mapping from dataset word to its corrections (the corrections is included in the vocabulary)
    """

    def __init__(self, embedding_path, dataset_vocabulary, hidden_size, num_labels, pad_token, device, OOV_corrections=False):
        """
        Glove download: https://nlp.stanford.edu/projects/glove/

        Args:
            embedding_path ([type]): [description]
        """

        super(BlindStatelessLSTM, self).__init__()
        #torch.manual_seed(seed) #TODO unique seed to replicate the experiment

        self.padding = pad_token
        self.corrected_flag = OOV_corrections
        self.hidden_size = hidden_size
        self.embedding_file = embedding_path.split('/')[-1]
        self.load_embeddings_from_file(embedding_path)
        embedding_weights = self.get_embeddings_weights(dataset_vocabulary, OOV_corrections)

        num_embeddings, embedding_dim = embedding_weights.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.load_state_dict({'weight': embedding_weights})

        self.lstm = nn.LSTM(self.embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_labels)


    def forward(self, batch, seq_lengths=None):
        """Forward pass for BlindStatelessLSTM

        Args:
            batch (torch.LongTensor): Tensor containing the batch (shape=BxMAX_SEQ_LEN)
            seq_lengths (torch.LongTensor, optional): Effective lengths (no pad) of each sequence in the batch. If given the pack_padded__sequence are used.
                                                        The shape is B. Defaults to None.
        """

        embedded_seq_tensor = self.embedding_layer(batch)
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
        out2 = self.linear(h_t[0]) # h_t has shape NUM_DIRxBxHIDDEN_SIZE
        #out2.shape = BxNUM_LABELS
        #todo dropout=.5
        predictions = F.softmax(out2, dim=-1)
        return out2, predictions
   

    def load_embeddings_from_file(self, embedding_path):
        self.glove = {}
        
        with open(embedding_path) as fp:
            for l in fp:
                line_tokens = l.split()
                #pdb.set_trace()
                word = line_tokens[0]
                if word in self.glove:
                    raise Exception('Repeated words in {} embeddings file'.format(embedding_path))
                vector = np.asarray(line_tokens[1:], "float32")
                self.glove[word] = vector
        self.embedding_size = vector.size


    def get_embeddings_weights(self, dataset_vocabulary, OOV_corrections):

        self.word2id = {}
        if OOV_corrections:
            dataset_vocabulary = self.correct_spelling(dataset_vocabulary)
        matrix_len = len(dataset_vocabulary)+1 #take into account the padding token
        weights_matrix = np.zeros((matrix_len, self.embedding_size))

        # set pad token for index 0
        weights_matrix[0] = np.zeros(shape=(self.embedding_size, ))
        self.word2id[self.padding] = 0
        for idx, word in enumerate(dataset_vocabulary):
            if word in self.glove:
                 weights_matrix[idx+1] = self.glove[word]
            else:
                weights_matrix[idx+1] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
            self.word2id[word] = idx+1

        return torch.tensor(weights_matrix, dtype=torch.float32)


    def correct_spelling(self, dataset_vocabulary):
        oov = []
        self.corrections = {}
        checker = SpellChecker()

        vocab_copy = copy.deepcopy(dataset_vocabulary)
        for word in vocab_copy:
            if word not in self.glove:
                oov.append(word) 
                corrected_w = checker.correction(word)
                if corrected_w in self.glove:
                    # the word with typos is assigned to the same id of the correspondant word after the correction
                    try:
                        self.word2id[word] = self.word2id[corrected_w] #TODO fix: word2id is still empty at this point
                    except:
                        pdb.set_trace()
                    self.corrections[word] = corrected_w
                    dataset_vocabulary.remove(word)
        #print(oov)
        #print(corrections.values())
        return dataset_vocabulary


    def collate_fn(self, batch):
        """This method prepares the batch for the LSTM: padding + preparation for pack_padded_sequence

        Args:
            batch (tuple): tuple of element returned by the Dataset.__getitem__()

        Returns:
            seq_tensor (torch.LongTensor): tensor with BxMAX_SEQ_LEN containing padded sequences of user transcript sorted by descending effective lengths
            targets (torch.Longtensor): tensor with B shape containing target actions
            seq_lenghts: tensor with shape B containing the effective length of the correspondant transcript sequence 
        """
        
        targets = torch.tensor([item[1] for item in batch])

        seq_ids = []
        for item in batch:
            curr_seq = []
            for word in item[0].split():
                curr_seq.append(self.word2id[word])
            seq_ids.append(curr_seq)
        
        seq_lengths = torch.tensor(list(map(len, seq_ids)), dtype=torch.long)
        seq_tensor = torch.zeros((len(seq_ids), seq_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(seq_ids, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)

        # sort instances by sequence length in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        seq_tensor = seq_tensor[perm_idx]
        targets = targets[perm_idx]

        # seq_lengths is used to create a pack_padded_sequence
        return seq_tensor, targets, seq_lengths


    def __str__(self):
        return super().__str__()
