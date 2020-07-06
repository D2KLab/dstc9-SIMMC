import json
import pdb

from torch.utils.data import Dataset




class SIMMCDataset(Dataset):

    def __init__(self, path, verbose=True):
        """Dataset constructor. The dataset has the following shapes

            self.id2dialog[<dialogue_id>].keys() = ['dialogue', 'dialogue_coref_map', 'dialogue_idx', 'domains', 'dialogue_task_id']

            self.id2dialog[<dialogue_id>][<dialogue_turn>].keys() = ['belief_state', 'domain', 'state_graph_0', 'state_graph_1', 'state_graph_2', 
                                                                    'system_transcript', 'system_transcript_annotated', 'system_turn_label', 
                                                                    'transcript', 'transcript_annotated', 'turn_idx', 'turn_label', 
                                                                    'visual_objects', 'raw_assistant_keystrokes']

        Args:
            path (str): path to dataset
        """
        fp = open(path)
        raw_json = json.load(fp)

        self.split = raw_json['split']
        self.version = raw_json['version']
        self.year = raw_json['year']
        self.domain = raw_json['domain']
        self.verbose = verbose
        if self.verbose:
            print('Creating index of dataset {}'.format(str(self)))

        raw_data = raw_json['dialogue_data']
        self.create_index(raw_data)
        if self.verbose:
            print('Index created')


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        dial_id = self.ids[index] 
        return self.id2dialog[dial_id]['dialogue'], self.id2dialog[dial_id]['dialogue_coref_map']


    def create_index(self, raw_data):
        self.ids = []
        self.id2dialog = {}
        for dialog in raw_data:
            self.ids.append(dialog['dialogue_idx'])
            try:
                dialog_obj = {
                            'dialogue': dialog['dialogue'], 
                            'dialogue_coref_map': dialog['dialogue_coref_map'], 
                            'dialogue_idx': dialog['dialogue_idx'], 
                            'domains': dialog['domains'], 
                            'dialogue_task_id': dialog['dialogue_task_id']}
            except:
                print('id: {} ; is dialogue_task_id missing: {}'.format(dialog['dialogue_idx'], not 'dialogue_task_id' in dialog))
            self.id2dialog[dialog['dialogue_idx']] = dialog_obj

    def __str__(self):
        return '{}_{}_{}_v{}'.format(self.domain, self.split, self.year, self.version)


class SIMMCDatasetForActionPrediction(SIMMCDataset):

    def __init__(self, path, verbose=True):
        self.task = 'action_prediction'
        super(SIMMCDatasetForActionPrediction, self).__init__(path, verbose)

    def __getitem__(self, index):
        dialogue, coref_map = super().__getitem__(index)
        #pdb.set_trace()
        #for count, dial in enumerate(dialogue):
        #    print('U{}: {} -- [{}]\nA{}: {}'.format(count, dial['transcript'], dial['belief_state'][0], count, dial['system_transcript']))
        #pdb.set_trace()
        return dialogue, coref_map

    def __len__(self):
        return super().__len__()

    def __str__(self):
        return '{}_{}'.format(super().__str__(), self.task)
