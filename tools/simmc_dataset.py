import json
import pdb
import re
import string

import torch
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
        """Dataset constructor. The dataset has the following shape

            (list) self.ids[idx] = <dialogue_id>

            (dict) self.id2dialog[<dialogue_id>].keys() = ['dialogue', 'dialogue_coref_map', 'dialogue_idx', 'domains', 'dialogue_task_id']

            (dict) self.id2dialog[<dialogue_id>]['dialogue'][<dialogue_turn>].keys() = ['belief_state', 'domain', 'state_graph_0', 'state_graph_1', 'state_graph_2', 
                                                                    'system_transcript', 'system_transcript_annotated', 'system_turn_label', 
                                                                    'transcript', 'transcript_annotated', 'turn_idx', 'turn_label', 
                                                                    'visual_objects', 'raw_assistant_keystrokes']

            (list) self.transcript[idx] = 'dialogueid_turn' (e.g., '3094_3', '3094_0')

            (dict) self.processed_turns[<dialogue_id>][turn] = {'transcript': <tokenized_transcript>, 'system_transcript': <tokenized_system_transcript>}

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
        #dial_id = self.ids[index]
        #return self.id2dialog[dial_id]['dialogue'], self.id2dialog[dial_id]['dialogue_coref_map']
        dial_id, turn = self.transcripts[index].split('_')
        dial_id = int(dial_id)
        turn = int(turn)

        if isinstance(self, SIMMCDatasetForActionPrediction,):
            return self.processed_turns[dial_id][turn]['transcript'], \
                    self.id2act[dial_id][turn]['action']
                    #self.id2act[dial_id][turn]['action_supervision']['attributes'] handle 'action_supervision': None


    def create_index(self, raw_data):
        self.ids = []
        self.id2dialog = {}
        self.transcripts = []
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
                transcripts = ['{}_{}'.format(dialog['dialogue_idx'], turn) for turn, _ in enumerate(dialog['dialogue'])]

            except:
                if self.verbose:
                    print('id: {} ; is dialogue_task_id missing: {}'.format(dialog['dialogue_idx'], not 'dialogue_task_id' in dialog))
            self.id2dialog[dialog['dialogue_idx']] = dialog_obj
            self.transcripts.extend(transcripts)


    def create_vocabulary(self):
        self.vocabulary = set()
        self.processed_turns = {}
        tokenizer = WordPunctTokenizer() 

        for dial_id in self.ids:
            self.processed_turns[dial_id] = []
            for dial_turn in self.id2dialog[dial_id]['dialogue']:
                self.processed_turns[dial_id].append({'transcript': '', 'system_transcript': ''})
                processed_user = ''
                processed_agent = ''

                user_tokens = tokenizer.tokenize(dial_turn['transcript'])
                for tok in user_tokens:
                    cleaned_tok = self.token_clean(tok)
                    self.vocabulary.add(cleaned_tok)
                    processed_user += cleaned_tok + ' '
                agent_tokens = tokenizer.tokenize(dial_turn['system_transcript'])
                for tok in agent_tokens:
                    cleaned_tok = self.token_clean(tok)
                    self.vocabulary.add(cleaned_tok)
                    processed_agent += cleaned_tok + ' '
                self.processed_turns[dial_id][dial_turn['turn_idx']]['transcript'] = processed_user[:-1] #remove final space
                self.processed_turns[dial_id][dial_turn['turn_idx']]['system_transcript'] = processed_agent[:-1] #remove final space


    def token_clean(self, token):
        if token.isnumeric():
            return '0'
        else:
            return token.lower()


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
    _ARG2LABEL = {'embellishment', 'skirtStyle', 'availableSizes', 'dressStyle', 'material', 'clothingStyle', 'jacketStyle', 'sleeveLength', 'soldBy', 'price', 'ageRange', 'hemLength', 
                    'size', 'warmthRating', 'sweaterStyle', 'forGender', 'madeIn', 'info', 'customerRating', 'hemStyle', 'hasPart', 'pattern', 'clothingCategory', 'forOccasion', 'waistStyle', 
                    'sleeveStyle', 'amountInStock', 'waterResistance', 'necklineStyle', 'skirtLength', 'color', 'brand', 'sequential'}


    def __init__(self, data_path, metadata_path, actions_path, verbose=True):
        
        super(SIMMCDatasetForActionPrediction, self).__init__(data_path=data_path, metadata_path=metadata_path, verbose=verbose)
        self.task = 'api_call_prediction'
        self.load_actions(actions_path)
        

    def __getitem__(self, index):
        #TODO think about labelling the arguments in an efficient way. To pack them -> one hot vector
        transcript, action = super().__getitem__(index)
        return transcript, self._ACT2LABEL[action]


    def __len__(self):
        return super().__len__()


    def __str__(self):
        return '{}_subtask({})'.format(super().__str__(), self.task)

    
    def load_actions(self, actions_path):
        #TODO sort id2act based on 'turn_idx' field
        self.id2act = {}
        with open(actions_path) as fp:
            raw_actions = json.load(fp)
        for action in raw_actions:
            if action['dialog_id'] in _DIALOGS_TO_SKIP:
                continue
            self.id2act[action['dialog_id']] = action['actions']
