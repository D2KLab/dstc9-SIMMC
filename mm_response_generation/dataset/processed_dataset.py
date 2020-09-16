import pdb
import random

import numpy as np
import torch
from torch.utils.data import Dataset


#_DATA_PERC = 50

class FastDataset(Dataset):
    """Dataset with preprocessed data for response generation subtask
    
    self.data.keys() = dict_keys(['dial_ids', 'turns', 'utterances', 'histories', 'actions', 
                                'attributes', 'visual_contexts', 'seq_lengths', 'candidates'])
    """
    def __init__(self, dat_path, metadata_ids_path, retrieval=False, distractors_sampling=-1):

        super(FastDataset, self).__init__()
        self.data = torch.load(dat_path)
        self.retrieval = retrieval
        if not retrieval:
            self.data['data_dict'].pop('candidates', None)
        self.metadata = torch.load(metadata_ids_path)
        self.dataset_name = 'SIMMC'
        self.task = 'response_retrieval'
        self.distractors_sampling = distractors_sampling

    def __getitem__(self, index):

        """
        candidates = []
        if self.retrieval and self.distractors_sampling >= 0:
            samples = random.sample(range(1, 100), self.distractors_sampling)
            # the first is always the ground truth
            candidates.append(self.data['data_dict']['candidates'][index][0])
            for sample in samples:
                candidates.append(self.data['data_dict']['candidates'][index][sample])
            assert len(candidates) == 1 + self.distractors_sampling, 'Invalid size of candidate list after sampling'
        else:
            candidates = self.data['data_dict']['candidates'][index]
        """
        focus_id = self.data['data_dict']['focus'][index]
        focus_pos = self.metadata['id2pos'][focus_id]
        if self.data['turns'][index] != 0:
            assert self.data['turns'][index] == self.data['data_dict']['history'][index]['input_ids'].shape[0], 'Number of turns and history length do not correpond'
        ret_tuple = (self.data['dial_ids'][index],
                    self.data['turns'][index],
                    self.data['data_dict']['utterances']['input_ids'][index],
                    self.data['data_dict']['utterances']['attention_mask'][index],
                    self.data['data_dict']['utterances']['token_type_ids'][index],
                    self.data['data_dict']['responses']['input_ids'][index],
                    self.data['data_dict']['responses']['attention_mask'][index],
                    self.data['data_dict']['responses']['token_type_ids'][index],
                    #self.data['data_dict']['history'][index],
                    #self.data['data_dict']['actions'][index],
                    #self.data['data_dict']['attributes'][index],
                    self.metadata['items_tensors']['input_ids'][focus_pos],
                    self.metadata['items_tensors']['attention_mask'][focus_pos],
                    self.metadata['items_tensors']['token_type_ids'][focus_pos])
        if self.retrieval:
            ret_tuple += (self.data['data_dict']['candidates']['input_ids'][index],
                        self.data['data_dict']['candidates']['attention_mask'][index],
                        self.data['data_dict']['candidates']['token_type_ids'][index])

        return ret_tuple


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
        #frac = int(len(self.data['utterances']) * (_DATA_PERC/100))
        #return frac
        return self.data['data_dict']['utterances']['input_ids'].shape[0]


    def __str__(self):
        return '{}_subtask({})'.format(self.dataset_name, self.task)