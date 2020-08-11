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

            (list) self.transcripts[idx] = 'dialogueid_turn' (e.g., '3094_3', '3094_0')

            (dict) self.processed_turns[<dialogue_id>][turn] = {'transcript': <tokenized_transcript>, 'system_transcript': <tokenized_system_transcript>}

        Args:
            path (str): path to dataset json file
            metadata_path (str): path to metadata json file
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
            print('Skipped dialogs: {}'.format(self.skipped_dialogs))
            print(' ... index created')
        self.create_vocabulary()


    def extract_visual_object(self, dial_id, turn, placeh2id):
        """Returns the visual object id and placeholder for a given turn and dialogue

        Args:
            dial_id (int): dialogue id
            turn (int): turn's number
            placeh2id (dict): dictionary from placeholder to object id

        Returns:
            tuple: (object_id, placeholder)
        """
        visual_object = self.id2dialog[dial_id]['dialogue'][turn]['visual_objects']
        if not len(visual_object):
            return None, None
        assert len(visual_object) <= 1, 'More than one visual object in turn {} of dialogue {}'.format(turn, dial_id)
        for obj in visual_object:
            visual_obj_placeholder = int(obj.split('_')[-1])
        return placeh2id[visual_obj_placeholder], visual_obj_placeholder


    def __len__(self):
        return len(self.transcripts)


    def __getitem__(self, index):
        # todo for the DSTC keep in mind that all the price values were set to 0 
        # todo      --> need to have a mapping to the original prices for the slot detection
        dial_id, turn = self.transcripts[index].split('_')
        dial_id = int(dial_id)
        turn = int(turn)

        current_transcript = self.processed_turns[dial_id][turn]['transcript']

        # extract visual objects
        coref_map = self.id2dialog[dial_id]['dialogue_coref_map']
        # inverted coref map: placeholder -> item_id
        inverted_coref_map = {}
        for key, item in coref_map.items():
            inverted_coref_map[item] = key
        focus_obj = self.extract_visual_object(dial_id, turn, inverted_coref_map)
        visual_context_dict = {'focus': focus_obj, 'history': []}

        # extract dialogue history
        history = []
        for t in range(turn):
            # extract visual history
            visual_object, _ = self.extract_visual_object(dial_id, t, inverted_coref_map)
            if visual_object is not None:
                visual_context_dict['history'].append(visual_object)
            # extract textual history
            qa = [self.processed_turns[dial_id][t]['transcript'], 
                            self.processed_turns[dial_id][t]['system_transcript']]
            history.append(qa)

        if isinstance(self, SIMMCDatasetForActionPrediction,):
            attributes = []
            if self.id2act[dial_id][turn]['action_supervision'] is not None:
                attributes = self.id2act[dial_id][turn]['action_supervision']['attributes']
            return dial_id, turn,\
                    current_transcript, history,\
                    visual_context_dict,\
                    self.id2act[dial_id][turn]['action'], attributes


    def create_index(self, raw_data):
        self.ids = []
        self.id2dialog = {}
        self.transcripts = []
        self.skipped_dialogs = set()
        for dialog in raw_data:

            if 'dialogue_task_id' in dialog:
                self.ids.append(dialog['dialogue_idx'])
                dialog_obj = {
                            'dialogue': dialog['dialogue'], 
                            'dialogue_coref_map': dialog['dialogue_coref_map'], 
                            'dialogue_idx': dialog['dialogue_idx'], 
                            'domains': dialog['domains'], 
                            'dialogue_task_id': dialog['dialogue_task_id']}
                transcripts = ['{}_{}'.format(dialog['dialogue_idx'], turn) for turn, _ in enumerate(dialog['dialogue'])]
                self.id2dialog[dialog['dialogue_idx']] = dialog_obj
                self.transcripts.extend(transcripts)
            else:
                if self.verbose:
                    #print('id: {} ; is dialogue_task_id missing: {}'.format(dialog['dialogue_idx'], not 'dialogue_task_id' in dialog))
                    self.skipped_dialogs.add(dialog['dialogue_idx'])


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
    _LABEL2ACT = ['None','SearchDatabase', 'SearchMemory', 'SpecifyInfo', 'AddToCart']
    _ATTR2LABEL = {'embellishment': 0, 'skirtStyle': 1, 'availableSizes': 2, 'dressStyle': 3, 'material': 4, 'clothingStyle': 5, 'jacketStyle': 6, 
                    'sleeveLength': 7, 'soldBy': 8, 'price': 9, 'ageRange': 10, 'hemLength': 11, 'size': 12, 'warmthRating': 13, 'sweaterStyle': 14, 
                    'forGender': 15, 'madeIn': 16, 'info': 17, 'customerRating': 18, 'hemStyle': 19, 'hasPart': 20, 'pattern': 21, 'clothingCategory': 22, 
                    'forOccasion': 23, 'waistStyle': 24, 'sleeveStyle': 25, 'amountInStock': 26, 'waterResistance': 27, 'necklineStyle': 28, 'skirtLength': 29, 
                    'color': 30, 'brand': 31, 'sequential': 32}
    _ATTRS = ['embellishment', 'skirtStyle', 'availableSizes', 'dressStyle', 'material', 'clothingStyle', 'jacketStyle', 
                    'sleeveLength', 'soldBy', 'price', 'ageRange', 'hemLength', 'size', 'warmthRating', 'sweaterStyle', 
                    'forGender', 'madeIn', 'info', 'customerRating', 'hemStyle', 'hasPart', 'pattern', 'clothingCategory', 
                    'forOccasion', 'waistStyle', 'sleeveStyle', 'amountInStock', 'waterResistance', 'necklineStyle', 'skirtLength', 
                    'color', 'brand', 'sequential']


    def __init__(self, data_path, metadata_path, actions_path, verbose=True):
        
        super(SIMMCDatasetForActionPrediction, self).__init__(data_path=data_path, metadata_path=metadata_path, verbose=verbose)
        self.task = 'api_call_prediction'
        self.load_actions(actions_path)
        

    def __getitem__(self, index):

        dial_id, turn, transcript, history, visual_context, action, attributes = super().__getitem__(index)

        one_hot_attrs = [0]*(len(self._ATTR2LABEL))
        for attr in attributes:
            assert attr in self._ATTR2LABEL, 'Unkown attribute \'{}\''.format(attr)
            assert one_hot_attrs[self._ATTR2LABEL[attr]] == 0, 'Attribute \'{}\' is present multiple times'.format(attr)
            one_hot_attrs[self._ATTR2LABEL[attr]] = 1

        return dial_id, turn, transcript, history, visual_context, self._ACT2LABEL[action], one_hot_attrs


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
            if action['dialog_id'] in self.skipped_dialogs:
                continue
            self.id2act[action['dialog_id']] = action['actions']
        
        #check if we have actions for all the turns
        for dial_id in self.ids:
            assert len(self.id2dialog[dial_id]['dialogue']) == len(self.id2act[dial_id]),\
                'Actions number does not match dialogue turns in dialogue {}'.format(dial_id)

        #compute frequency for actions
        act_freq = [0]*5
        freq_sum = 0
        for dial_id in self.ids:
            for act in self.id2act[dial_id]:
                act_freq[self._ACT2LABEL[act['action']]] += 1
                freq_sum += 1
        self.act_support = {'per_class_frequency': act_freq, 'tot_samples': freq_sum}

        #compute frequency for attributes
        attr_freq = [0] * 33
        freq_sum = 0
        for dial_id in self.ids:
            for act in self.id2act[dial_id]:
                if act['action_supervision'] != None:
                    for attr in act['action_supervision']['attributes']:
                        attr_freq[self._ATTR2LABEL[attr]] += 1
                        freq_sum += 1
        self.attr_support = {'per_class_frequency': attr_freq, 'tot_samples': freq_sum}

        """freq_sum = 0
        for idx, freq in enumerate(attr_freq):
            print('{}: {}'.format(self._ATTRS[idx], freq))
            freq_sum += freq
        print('Total support sum: {}'.format(freq_sum))
        """
        
