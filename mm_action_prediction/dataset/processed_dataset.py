import numpy as np
import torch
from torch.utils.data import Dataset
import pdb


class FastDataset(Dataset):
    """Dataset with preprocessed data for response generation subtask
    
    self.data.keys() = dict_keys(['dial_ids', 'turns', 'utterances', 'histories', 'actions', 
                                'attributes', 'visual_contexts', 'seq_lengths', 'candidates'])
    """
    def __init__(self, dat_path):

        super(FastDataset, self).__init__()
        self.data = torch.load(dat_path)
        self.dataset_name = 'SIMMC'
        self.task = 'action_prediction'
        self.num_actions = self.data['num_actions']
        self.num_attributes = self.data['num_attributes']
        self.act_support = self.data['actions_support']
        self.attr_support = self.data['attributes_support']


    def __getitem__(self, index):

        return self.data['dial_ids'][index], self.data['turns'][index], self.data['utterances'][index],\
                self.data['histories'][index], self.data['actions'][index], self.data['attributes'][index],\
                self.data['visual_contexts']['focus'][index], self.data['visual_contexts']['history'][index],

    def __len__(self):
        return len(self.data['utterances'])


    def ___str__(self):
        return '{}_subtask({})'.format(self.dataset_name, self.task)
