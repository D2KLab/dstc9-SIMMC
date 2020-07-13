import json
import pdb
import string
import re

from torch.utils.data import Dataset
from nltk.tokenize import WordPunctTokenizer 


"""
The dialog acts and intents have the shapes:
DA:<DIALOG_ACT>:<ACTIVITY>:<OBJECT> or DA:<DIALOG_ACT>:<ACTIVITY>:<OBJECT>.<attribute> 

Examples:
DA:INFORM:GET:CLOTHING.embellishment

The <DIALOG_ACT> values are shared between fashion and furniture dataset. <ACTIVITY> values are dataset specific (see paper fig.3).
"""

# shared between furniture and fashion dataset
DIALOG_ACT = {'ASK', 'CONFIRM', 'INFORM', 'PROMPT', 'REQUEST'}



class SIMMCDataset(Dataset):


    ACTIVITY = {'ADD_TO_CART', 'CHECK', 'COMPARE', 'COUNT', 'DISPREFER', 'GET', 'PREFER', 'REFINE'}

    def __init__(self, data_path, metadata_path, verbose=True):
        """Dataset constructor. The dataset has the following shapes

            self.id2dialog[<dialogue_id>].keys() = ['dialogue', 'dialogue_coref_map', 'dialogue_idx', 'domains', 'dialogue_task_id']

            self.id2dialog[<dialogue_id>][<dialogue_turn>].keys() = ['belief_state', 'domain', 'state_graph_0', 'state_graph_1', 'state_graph_2', 
                                                                    'system_transcript', 'system_transcript_annotated', 'system_turn_label', 
                                                                    'transcript', 'transcript_annotated', 'turn_idx', 'turn_label', 
                                                                    'visual_objects', 'raw_assistant_keystrokes']

        Args:
            path (str): path to dataset
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
            print('Creating index of dataset {}'.format(str(self)))

        raw_data = raw_data['dialogue_data']
        self.create_index(raw_data)
        if self.verbose:
            print('Index created')
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


    def create_vocabulary(self):
        self.vocabulary = set()
        tokenizer = WordPunctTokenizer() 

        for dial_id in self.ids:
            for dial_turn in self.id2dialog[dial_id]['dialogue']:
                user_tokens = tokenizer.tokenize(dial_turn['transcript'])
                """
                pdb.set_trace()
                for tok in user_tokens:
                    clean_toks = self.clean_token(tok)
                    for t in clean_toks:
                """
                for tok in user_tokens:
                    self.vocabulary.add(tok.lower())
                agent_tokens = tokenizer.tokenize(dial_turn['system_transcript'])
                for tok in agent_tokens:
                    self.vocabulary.add(tok.lower())


    def clean_token(self, token):
        #remove puncuation
        #pdb.set_trace()
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
