import pdb
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class FastDataset(Dataset):
    """Dataset with preprocessed data for response generation subtask
    
    self.data.keys() = dict_keys(['dial_ids', 'turns', 'utterances', 'histories', 'actions', 
                                'attributes', 'visual_contexts', 'seq_lengths', 'candidates'])
    """
    def __init__(self, dat_path, distractors_sampling=-1):

        super(FastDataset, self).__init__()
        self.data = torch.load(dat_path)
        self.dataset_name = 'SIMMC'
        self.task = 'response_retrienval'
        self.distractors_sampling = distractors_sampling


    def __getitem__(self, index):

        candidates = []
        if self.distractors_sampling >= 0:
            samples = random.sample(range(1, 100), self.distractors_sampling)
            # the first is always the ground truth
            candidates.append(self.data['candidates'][index][0])
            for sample in samples:
                candidates.append(self.data['candidates'][index][sample])
            assert len(candidates) == 1 + self.distractors_sampling, 'Invalid size of candidate list after sampling'
        else:
            candidates = self.data['candidates'][index]

        return self.data['dial_ids'][index], self.data['turns'][index], self.data['utterances'][index],\
                self.data['histories'][index], self.data['actions'][index], self.data['attributes'][index],\
                self.data['visual_contexts']['focus'][index], self.data['visual_contexts']['history'][index],\
                candidates


    def create_id2turns(self):
        """used to create the eval dict during evaluation phase
        """
        self.id2turns = {}
        for dial_id in self.data['dial_ids']:
            if dial_id in self.id2turns:
                self.id2turns[dial_id] += 1
            else:
                self.id2turns[dial_id] = 1


    def __len__(self):
        return len(self.data['utterances'])


    def __str__(self):
        return '{}_subtask({})'.format(self.dataset_name, self.task)