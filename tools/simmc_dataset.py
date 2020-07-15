import json
import pdb
import re
import string

from nltk.tokenize import WordPunctTokenizer
from torch.utils.data import Dataset

"""
The dialog intents have the shapes:
DA:<DIALOG_ACT>:<ACTIVITY>:<OBJECT> or DA:<DIALOG_ACT>:<ACTIVITY>:<OBJECT>.<attribute> 

Examples:
DA:INFORM:GET:CLOTHING.embellishment

The <DIALOG_ACT> values are shared between fashion and furniture dataset. <ACTIVITY> values are dataset specific (see paper fig.3).
"""


DIALOG_ACT = {'ASK', 'CONFIRM', 'INFORM', 'PROMPT', 'REQUEST'}
ACTIVITY = {'ADD_TO_CART', 'CHECK', 'COMPARE', 'COUNT', 'DISPREFER', 'GET', 'PREFER', 'REFINE'}
_DIALOGS_TO_SKIP = {321, 3969, 3406, 4847, 3414} #actions do not match turns for these dialogues


class SIMMCDataset(Dataset):
    """Dataset wrapper for SIMMC Fashion
    """

    def __init__(self, data_path, metadata_path, verbose=True):
        """Dataset constructor. The dataset has the following shapes

            self.id2dialog[<dialogue_id>].keys() = ['dialogue', 'dialogue_coref_map', 'dialogue_idx', 'domains', 'dialogue_task_id']

            self.id2dialog[<dialogue_id>]['dialogue'][<dialogue_turn>].keys() = ['belief_state', 'domain', 'state_graph_0', 'state_graph_1', 'state_graph_2', 
                                                                    'system_transcript', 'system_transcript_annotated', 'system_turn_label', 
                                                                    'transcript', 'transcript_annotated', 'turn_idx', 'turn_label', 
                                                                    'visual_objects', 'raw_assistant_keystrokes']
        Args:
            path (str): path to dataset json file
            metadata_path (str): path to metadata json file file
        """
        data_fp = open(data_path)
        raw_data = json.load(data_fp)

        metadata_fp = open(metadata_path)
        self.metadata = json.load(metadata_fp)

        self.split = raw_data['split']
        self.version = raw_data['version']
        self.year = raw_data['year']
        self.domain = raw_data['domain']
        self.verbose = verbose
        if self.verbose:
            print('Creating dataset index ...')

        raw_data = raw_data['dialogue_data']
        self.create_index(raw_data)
        if self.verbose:
            print(' ... index created')
        self.create_vocabulary()


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        dial_id = self.ids[index] 
        return self.id2dialog[dial_id]['dialogue'], self.id2dialog[dial_id]['dialogue_coref_map']


    def create_index(self, raw_data):
        self.ids = []
        self.id2dialog = {}
        for dialog in raw_data:
            if dialog['dialogue_idx'] in _DIALOGS_TO_SKIP:
                continue
            self.ids.append(dialog['dialogue_idx'])
            try:
                dialog_obj = {
                            'dialogue': dialog['dialogue'], 
                            'dialogue_coref_map': dialog['dialogue_coref_map'], 
                            'dialogue_idx': dialog['dialogue_idx'], 
                            'domains': dialog['domains'], 
                            'dialogue_task_id': dialog['dialogue_task_id']}
            except:
                if self.verbose:
                    print('id: {} ; is dialogue_task_id missing: {}'.format(dialog['dialogue_idx'], not 'dialogue_task_id' in dialog))
            self.id2dialog[dialog['dialogue_idx']] = dialog_obj


    def create_vocabulary(self):
        self.vocabulary = set()
        tokenizer = WordPunctTokenizer() 

        for dial_id in self.ids:
            for dial_turn in self.id2dialog[dial_id]['dialogue']:
                user_tokens = tokenizer.tokenize(dial_turn['transcript'])
                for tok in user_tokens:
                    self.vocabulary.add(tok.lower())
                agent_tokens = tokenizer.tokenize(dial_turn['system_transcript'])
                for tok in agent_tokens:
                    self.vocabulary.add(tok.lower())


    def clean_token(self, token):
        """Text lowercased, punctuation removal (e.g., from "meta-data" to "meta" and "data") and numbers conversion to 0.
        These strategies are used to keep the vocabulary more compact (avoiding different embeddings for all the numbers from 0 to 1000)

        Args:
            token (str): A single token to clean

        Returns:
            str: cleaned token
        """
        tmp_token = token.lower()
        
        try:
            if tmp_token[0] in string.punctuation and len(tmp_token) > 1:
                tmp_token = tmp_token[1:]
            if tmp_token[-1] in string.punctuation and len(tmp_token) > 1:
                tmp_token = tmp_token[:-1]
            tmp_token = tmp_token.split('.-/')
            for idx, t in enumerate(tmp_token):
                if t.isnumeric():
                    tmp_token[idx] = '0'  
        except:
            pdb.set_trace()
        
        return tmp_token


    def get_vocabulary(self):
        return self.vocabulary


    def getmetadata(self, obj_id):
        """Return metadata for the object with the specified id

        Args:
            obj_id (str): id of the object

        Returns:
            dict: returns a dict with the following shape
            {'metadata': 
                {'availability': [], 
                'availableSizes': "['L', 'XXL']", 
                'brand': '212 Local', 
                'color': ['black'], 
                'customerRating': '2.06', 
                'embellishments': ['layered'], 
                'hemLength': ['knee_length'], 
                'pattern': [], 
                'price': '$269',
                'size': [], 
                'skirtStyle': ['asymmetrical', 'fit_and_flare', 'loose'], 
                'type': 'skirt'
                }, 
            'url': 'GByeggJtfhLUq9UGAAAAAABqViN1btAUAAAB'
            }
        """
        return self.metadata[obj_id]


    def __str__(self):
        return '{}_{}_{}_v{}'.format(self.domain, self.split, self.year, self.version)



class SIMMCDatasetForActionPrediction(SIMMCDataset):
    """Dataset wrapper for SIMMC Fashion for api call prediction subtask
    """

    _ACT2LABEL = {'None': 0,'SearchDatabase': 1, 'SearchMemory': 2, 'SpecifyInfo': 3, 'AddToCart': 4}

    def __init__(self, data_path, metadata_path, actions_path, verbose=True):
        
        super(SIMMCDatasetForActionPrediction, self).__init__(data_path=data_path, metadata_path=metadata_path, verbose=verbose)
        self.task = 'api_call_prediction'
        self.load_actions(actions_path)
        self.create_target_tensors()
        

    def __getitem__(self, index):
        #TODO finish this method. Write a better parent __getitem__() and maybe structure differently the data. Maybe indexing them by dialogue turn

        dialogue, _ = super().__getitem__(index)
        actions = self.id2act[self.ids[index]]
        assert len(dialogue) == len(actions), 'Lengths of dialogue and actions arrays do not match'
        pdb.set_trace()
        #pdb.set_trace()
        #for count, dial in enumerate(dialogue):
        #    print('U{}: {} -- [{}]\nA{}: {}'.format(count, dial['transcript'], dial['belief_state'][0], count, dial['system_transcript']))
        #pdb.set_trace()
        return dialogue, actions


    def __len__(self):
        return super().__len__()


    def __str__(self):
        return '{}_subtask({})'.format(super().__str__(), self.task)

    
    def load_actions(self, actions_path):
        self.id2act = {}
        with open(actions_path) as fp:
            raw_actions = json.load(fp)
        for action in raw_actions:
            if action['dialog_id'] in _DIALOGS_TO_SKIP:
                continue
            self.id2act[action['dialog_id']] = action['actions']


    def create_target_tensors(self):
        """Creates the target tensors that for each dialog and for each turn contains the action and the arguments (tag 0,1 for each word).
        """
        pass
