import pdb

import numpy as np
import torch
import torch.nn as nn
from spellchecker import SpellChecker


class ItemEmbeddingNetwork(nn.Module):
    """Base class for word embedding layer initialization and weights loading

    Args:
        nn (torch.nn.Module): inherits from torch.nn.Module
    """
    def __init__(self, item_embeddings_path, freeze=False):

        super(ItemEmbeddingNetwork, self).__init__()

        raw_data = np.load(item_embeddings_path, allow_pickle=True)
        raw_data = dict(raw_data.item())

        self.item2id = {}
        for idx, item in enumerate(raw_data['item_ids']):
            self.item2id[item] = idx

        self.embedding_dim = raw_data['embedding_size']
        fields_embeddings = np.stack([item_embs[0] for item_embs in raw_data['embeddings']])
        values_embeddings = np.stack([item_embs[1] for item_embs in raw_data['embeddings']])
        fields_embedding_weights = torch.tensor(fields_embeddings)
        values_embedding_weights = torch.tensor(values_embeddings)

        pdb.set_trace()
        assert fields_embedding_weights.shape[0] == values_embedding_weights.shape[0], 'Number of fields and values embedding does not match'
        assert fields_embedding_weights.shape[-1] == values_embedding_weights.shape[-1] and fields_embedding_weights.shape[-1] == self.embedding_dim,\
                                                                                    'Real embedding dimension does not match the declared one'
        num_embeddings = fields_embedding_weights.shape[0]
        self.fields_embedding_layer = nn.Embedding(num_embeddings, self.embedding_dim)
        self.fields_embedding_layer.load_state_dict({'weight': fields_embedding_weights})
        self.values_embedding_layer = nn.Embedding(num_embeddings, self.embedding_dim)
        self.values_embedding_layer.load_state_dict({'weight': values_embedding_weights})

        if freeze:
            for p in self.fields_embedding_layer.parameters():
                p.requires_grad = False
            for p in self.values_embedding_layer.parameters():
                p.requires_grad = False        


    def forward(self, fields_ids, values_ids):
        return self.fields_embedding_layer(fields_ids), self.values_embedding_layer(values_ids)



class WordEmbeddingNetwork(nn.Module):
    """Base class for word embedding layer initialization and weights loading

    Args:
        nn (torch.nn.Module): inherits from torch.nn.Module
    """
    def __init__(self, word_embeddings_path, word2id, pad_token, unk_token, OOV_corrections=False, freeze=False):

        super(WordEmbeddingNetwork, self).__init__()
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.corrected_flag = OOV_corrections
        self.word2id = word2id
        self.embedding_file = word_embeddings_path.split('/')[-1]
        self.load_embeddings_from_file(word_embeddings_path)

        embedding_weights = self.get_embeddings_weights(OOV_corrections)

        num_embeddings, self.embedding_dim = embedding_weights.shape
        self.embedding_layer = nn.Embedding(num_embeddings, self.embedding_dim)
        self.embedding_layer.load_state_dict({'weight': embedding_weights})
        if freeze:
            for p in self.embedding_layer.parameters():
                p.requires_grad = False


    def forward(self, input):
        return self.embedding_layer(input)


    def load_embeddings_from_file(self, embeddings_path):
        self.glove = {}
        
        with open(embeddings_path) as fp:
            for l in fp:
                line_tokens = l.split()
                word = line_tokens[0]
                if word in self.glove:
                    raise Exception('Repeated words in {} embeddings file'.format(embeddings_path))
                vector = np.asarray(line_tokens[1:], "float32")
                self.glove[word] = vector
        self.embedding_size = vector.size


    def get_embeddings_weights(self, OOV_corrections):

        #if OOV_corrections:
        #    dataset_vocabulary = self.correct_spelling(dataset_vocabulary)
        matrix_len = len(self.word2id)
        weights_matrix = np.zeros((matrix_len, self.embedding_size))

        # set pad and unknow ids
        pad_id = self.word2id[self.pad_token]
        unk_id = self.word2id[self.unk_token]
        weights_matrix[pad_id] = np.zeros(shape=(self.embedding_size, ))
        weights_matrix[unk_id] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
        
        for idx, word in enumerate(self.word2id):
            if word in self.glove:
                 weights_matrix[idx] = self.glove[word]

        return torch.tensor(weights_matrix, dtype=torch.float32)


    def correct_spelling(self, dataset_vocabulary):
        #todo fix: now dataset_vocabulary is a map, not a set (get the .keys())
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
