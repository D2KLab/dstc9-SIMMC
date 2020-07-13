import pdb
import copy

from spellchecker import SpellChecker
import torch
from torch import nn
import numpy as np


class BlindStatelessLSTM(nn.Module):
    """Implementation of a blind and stateless LSTM for action prediction. It approximates the probability distribution:

            P(a_t | U_t)
    
        Where a_t is the action and U_t the user utterance.

    Args:
        torch (torch.nn.Module): inherits from torch.nn.Module
    """

    def __init__(self, embedding_path, dataset_vocabulary, OOV_corrections=False):
        """
        Glove download: https://nlp.stanford.edu/projects/glove/

        Args:
            embedding_path ([type]): [description]
        """

        self.embedding_file = embedding_path.split('/')[-1]
        self.load_embeddings_from_file(embedding_path)
        embedding_weights = self.get_embeddings_weights(dataset_vocabulary, OOV_corrections)

        num_embeddings, embedding_dim = embedding_weights.shape
        embeddings_layer = nn.Embedding(num_embeddings, embedding_dim)
        embeddings_layer.load_state_dict({'weight': embedding_weights})
        pdb.set_trace()


        nn.Sequential(
                    nn.Embedding(),
                    nn.LSTM(10, 10)
                )
        pdb.set_trace()
        
        

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
        not_found = []

        if OOV_corrections:
            dataset_vocabulary = self.correct_spelling(dataset_vocabulary)
        
        matrix_len = len(dataset_vocabulary)
        weights_matrix = np.zeros((matrix_len, 50))

        for idx, word in enumerate(dataset_vocabulary):
            if word in self.glove:
                 weights_matrix[idx] = self.glove[word]
            else:
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(self.embedding_size, ))

        return weights_matrix


    def correct_spelling(self, dataset_vocabulary):
        oov = []
        corrections = []
        checker = SpellChecker()

        vocab_copy = copy.deepcopy(dataset_vocabulary)
        for word in vocab_copy:
            if word not in self.glove:
                oov.append(word) 
                corrected_w = checker.correction(word)
                if corrected_w in self.glove:
                    corrections.append(corrected_w)
                    dataset_vocabulary.remove(word)
        #print(oov)
        #print(corrections)
        return dataset_vocabulary


