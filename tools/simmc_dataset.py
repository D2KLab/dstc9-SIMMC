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

    (list) self.ids[idx] = <dialogue_id>

    (dict) self.id2dialog[<dialogue_id>].keys() = ['dialogue', 'dialogue_coref_map', 'dialogue_idx', 'domains', 'dialogue_task_id']

    (dict) self.id2dialog[<dialogue_id>]['dialogue'][<dialogue_turn>].keys() = ['belief_state', 'domain', 'state_graph_0', 'state_graph_1', 'state_graph_2', 
                                                            'system_transcript', 'system_transcript_annotated', 'system_turn_label', 
                                                            'transcript', 'transcript_annotated', 'turn_idx', 'turn_label', 
                                                            'visual_objects', 'raw_assistant_keystrokes']

    (list) self.transcripts[idx] = 'dialogueid_turn' (e.g., '3094_3', '3094_0')

    (dict) self.task_mapping[<task_id>].keys() = ['task_id', 'image_ids', 'focus_image', 'memory_images', 'database_images']

    (dict) self.processed_turns[<dialogue_id>][turn] = {'transcript': <tokenized_transcript>, 'system_transcript': <tokenized_system_transcript>}
    """

    def __init__(self, data_path, metadata_path, verbose=True):
        """Dataset constructor.
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

        self.create_index(raw_data)
        if self.verbose:
            print('Skipped dialogs: {}'.format(self.skipped_dialogs))
            print(' ... index created')


    def __len__(self):
        return len(self.transcripts)


    def __getitem__(self, index):

        dial_id, turn = self.transcripts[index].split('_')
        dial_id = int(dial_id)
        turn = int(turn)

        user_req = self.id2dialog[dial_id]['dialogue'][turn]['transcript']
        wizard_resp = self.id2dialog[dial_id]['dialogue'][turn]['system_transcript']

        # extract dialogue history
        turn_str = '{} [SEP] {}'
        history = [turn_str.format(self.id2dialog[dial_id]['dialogue'][t]['transcript'],
                                    self.id2dialog[dial_id]['dialogue'][t]['transcript'])
                    for t in range(turn)]

        # dispatch data across different dataset instantiation
        if isinstance(self, SIMMCDatasetForActionPrediction,) or isinstance(self, SIMMCDatasetForResponseGeneration,):
            focus_item = self.id2focus[dial_id][turn]
            attributes = []
            if self.id2act[dial_id][turn]['action_supervision'] is not None:
                attributes = self.id2act[dial_id][turn]['action_supervision']['attributes']
            return_tuple = (dial_id, turn, user_req, wizard_resp, history, focus_item, self.id2act[dial_id][turn]['action'], attributes)
            if isinstance(self, SIMMCDatasetForResponseGeneration,):
                return_tuple += (self.id2candidates[dial_id][turn]['retrieval_candidates'],)

            return return_tuple


    def extract_visual_context(self, dial_id):
        task_id = self.id2dialog[dial_id]['dialogue_task_id']
        init_focus = self.task_mapping[task_id]['focus_image']
        focus_items = [init_focus]
        for act_annotation in self.id2act[dial_id]:
            #force object permanence
            if act_annotation['action_supervision'] is None or 'focus' not in act_annotation['action_supervision']:
                focus_items.append(focus_items[-1])
            else:
                focus_items.append(act_annotation['action_supervision']['focus'])
        return focus_items


    def create_index(self, raw_data):
        self.ids = []
        self.id2dialog = {}
        self.transcripts = []
        self.skipped_dialogs = set()
        for dialog in raw_data['dialogue_data']:

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

        self.task_mapping = {}
        for task in raw_data['task_mapping']:
            self.task_mapping[task['task_id']] = task


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



class SIMMCDatasetForResponseGeneration(SIMMCDataset):

    # conversion from attribute and action annotations format to english string
    _ATTRS = {'embellishment', 'skirtStyle', 'availableSizes', 'dressStyle', 'material', 'clothingStyle', 'jacketStyle', 
                    'sleeveLength', 'soldBy', 'price', 'ageRange', 'hemLength', 'size', 'warmthRating', 'sweaterStyle', 
                    'forGender', 'madeIn', 'info', 'customerRating', 'hemStyle', 'hasPart', 'pattern', 'clothingCategory', 
                    'forOccasion', 'waistStyle', 'sleeveStyle', 'amountInStock', 'waterResistance', 'necklineStyle', 'skirtLength', 
                    'color', 'brand', 'sequential'}
    _ATTR2STR = {'skirtstyle': 'skirt style', 'availablesizes': 'available sizes', 'dressstyle': 'dress style', 'clothingstyle': 'clothing style', 
                'jacketstyle': 'jacket style', 'sleevelength': 'sleeve length', 'soldby': 'sold by', 'agerange': 'age range', 'hemlength': 'hem length', 
                'warmthrating': 'warmth rating', 'sweaterstyle': 'sweater style', 'forgender': 'for gender', 'madein': 'made in', 'customerrating': 'customer rating',
                'hemstyle': 'hem style', 'haspart': 'has part', 'clothingcategory': 'clothing category', 'foroccasion': 'for occasion', 'waiststyle': 'waist style', 
                'sleevestyle': 'sleeve style', 'amountinstock': 'amount in stock', 'waterresistance': 'water resistance', 'necklinestyle': 'neckline style', 
                'skirtlength': 'skirt length'}
    _ACT2STR = {'none': 'none', 'searchdatabase': 'search database', 'searchmemory': 'search memory', 'specifyinfo': 'specify info', 'addtocart': 'add to cart'}


    def __init__(self, data_path, metadata_path, actions_path, candidates_path, verbose=True):
        super(SIMMCDatasetForResponseGeneration, self).__init__(data_path=data_path, metadata_path=metadata_path, verbose=verbose)

        self.task = 'response_generation'
        self.load_actions(actions_path)
        self.load_candidates(candidates_path)
        self.id2focus = {}
        for id in self.ids:
            #for response generation the context is shifted right (response based on the item chosen by the wizard)
            self.id2focus[id] = self.extract_visual_context(id)[1:]
            assert len(self.id2dialog[id]['dialogue']) == len(self.id2focus[id]), 'Focus items do not match dialogue {} length'.format(id)

        self.processed_metadata = {}
        self.process_metadata_items()


    def process_metadata_items(self):
        """This method process the data inside metadata fields and make each field values a list 
            (avoiding mixing up single values and lists)

        Args:
            tokenizer ([type]): [description]
        """
        for item_id, item in self.metadata.items():
            assert item_id not in self.processed_metadata, 'Item {} presents twice'.format(item_id)
            self.processed_metadata[item_id] = {}
            for field, field_vals in item['metadata'].items():
                curr_field = ''
                # availability field is always empty
                if field == 'availability' or field == 'url':
                    continue
                values = field_vals
                if field == 'availableSizes' and not isinstance(values, list,):
                    values = self.repair_size_list(values)

                #field_tokens = tokenizer.tokenize(field)
                field_tokens = re.split('_|\s', field)
                for tok in field_tokens:
                    cleaned_tok = self._ATTR2STR[tok.lower()] if tok.lower() in self._ATTR2STR else tok.lower()
                    curr_field += cleaned_tok + ' '
                curr_field = curr_field[:-1]
                
                curr_val = ''
                proc_values = []
                if isinstance(values, list,):
                    for val in values:
                        curr_val = ''
                        #value_tokens = tokenizer.tokenize(val)
                        value_tokens = re.split('_|\s', val)
                        proc_values.append(' '.join(value_tokens))
                else:
                    value_tokens = re.split('_|\s', values)
                    proc_values.append(' '.join(value_tokens))

                #metadata JSON files contains different samples having hemLenght field twice.
                #   In this case just discard the one with no values.
                if curr_field == 'hem length' and curr_field in self.processed_metadata[item_id]:
                    if not len(self.processed_metadata[item_id][curr_field]):
                        self.processed_metadata[item_id][curr_field] = proc_values
                        continue
                assert curr_field not in self.processed_metadata[item_id], 'Field {} presents twice in item {}. Please remove one of them (preferably the empty one)'.format(curr_field, item_id)
                self.processed_metadata[item_id][curr_field] = proc_values


    def repair_size_list(self, str_val):
        """fixes availableSizes when it is a stringified list (e.g., "[' xl ', ' m ']"

        Args:
            str_val ([type]): [description]
        """
        return [word for word in str_val[2:-2].split('\', \'')]


    def __getitem__(self, index):

        dial_id, turn, user_req, wizard_resp, history, focus, action, attributes, candidates_ids = super().__getitem__(index)

        #convert actions and attributes to english strings
        action = action.lower() if action.lower() not in self._ACT2STR else self._ACT2STR[action.lower()]
        attributes = [attr.lower() if attr.lower() not in self._ATTR2STR else self._ATTR2STR[attr.lower()] for attr in attributes]
        retrieval_candidates = [self.candidates[candidate_id] for candidate_id in candidates_ids]

        return dial_id, turn, user_req, wizard_resp, history, focus, action, attributes, retrieval_candidates


    def __len__(self):
        return super().__len__()


    def __str__(self):
        return '{}_subtask({})'.format(super().__str__(), self.task)


    def load_candidates(self, candidates_path):
        self.candidates = []
        self.id2candidates = {}
        with open(candidates_path) as fp:
            raw_candidates = json.load(fp)
        for candidate in raw_candidates['system_transcript_pool']:
            self.candidates.append(candidate)
        for candidates_per_dial in raw_candidates['retrieval_candidates']:
            self.id2candidates[candidates_per_dial['dialogue_idx']] = candidates_per_dial['retrieval_candidates']
        
        #check if all the candidate ids correspond to a valid candidate in the candidate pool
        for (_, candidates_per_dial) in self.id2candidates.items():
            for candidates_per_turn in candidates_per_dial:
                for candidate_id in candidates_per_turn['retrieval_candidates']:
                    assert candidate_id < len(self.candidates), 'Candidate with id {} not present in candidate pool'.format(candidate_id)


    def load_actions(self, actions_path):
        #TODO sort id2act based on 'turn_idx' field
        self.id2act = {}
        self.id2actfocus = {}
        with open(actions_path) as fp:
            raw_actions = json.load(fp)
        for action in raw_actions:
            if action['dialog_id'] in self.skipped_dialogs:
                continue
            assert len(action['actions']) == len(action['focus_images']), 'focus_images has different length than number of actions'
            self.id2act[action['dialog_id']] = action['actions']
            self.id2actfocus[action['dialog_id']] = action['focus_images']
        
        #check if we have actions for all the turns
        for dial_id in self.ids:
            assert len(self.id2dialog[dial_id]['dialogue']) == len(self.id2act[dial_id]),\
                'Actions number does not match dialogue turns in dialogue {}'.format(dial_id)



class SIMMCDatasetForActionPrediction(SIMMCDataset):
    """Dataset wrapper for SIMMC Fashion for api call prediction subtask
    """

    _ACT2LABEL = {'None': 0,'SearchDatabase': 1, 'SearchMemory': 2, 'SpecifyInfo': 3, 'AddToCart': 4}
    _LABEL2ACT = ['None','SearchDatabase', 'SearchMemory', 'SpecifyInfo', 'AddToCart']
    """
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
    """
    _ATTR2LABEL = {'embellishment': 0, 'availableSizes': 1, 'price': 2, 'info': 3, 'customerRating': 4, 
                    'pattern': 5, 'color': 6, 'brand': 7, 'other': 8}
    _ATTRS = ['embellishment', 'availableSizes', 'price', 'info', 'customerRating', 'pattern', 'color', 'brand', 'other']

    


    def __init__(self, data_path, metadata_path, actions_path, verbose=True):
        
        super(SIMMCDatasetForActionPrediction, self).__init__(data_path=data_path, metadata_path=metadata_path, verbose=verbose)
        self.task = 'api_call_prediction'
        self.load_actions(actions_path)
        self.id2focus = {}
        for id in self.ids:
            #for action prediction do not use the item context after the last turn
            self.id2focus[id] = self.extract_visual_context(id)[:-1]
            assert len(self.id2dialog[id]['dialogue']) == len(self.id2focus[id]), 'Focus items do not match dialogue {} length'.format(id)
        

    def __getitem__(self, index):

        dial_id, turn, transcript, history, visual_context, action, attributes = super().__getitem__(index)
        one_hot_attrs = [0]*(len(self._ATTR2LABEL))
        for attr in attributes:
            #assert attr in self._ATTR2LABEL, 'Unkown attribute \'{}\''.format(attr)
            curr_attr = attr if attr in self._ATTR2LABEL else 'other'
            #assert one_hot_attrs[self._ATTR2LABEL[curr_attr]] == 0, 'Attribute \'{}\' is present multiple times'.format(attr)
            one_hot_attrs[self._ATTR2LABEL[curr_attr]] = 1
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
        act_freq = [0]*len(self._LABEL2ACT)
        freq_sum = 0
        for dial_id in self.ids:
            for act in self.id2act[dial_id]:
                act_freq[self._ACT2LABEL[act['action']]] += 1
                freq_sum += 1
        self.act_support = {'per_class_frequency': act_freq, 'tot_samples': freq_sum}

        #compute frequency for attributes
        attr_freq = [0] * len(self._ATTRS)
        freq_sum = 0
        for dial_id in self.ids:
            for act in self.id2act[dial_id]:
                if act['action_supervision'] != None:
                    for attr in act['action_supervision']['attributes']:
                        if attr in self._ATTR2LABEL:
                            attr_freq[self._ATTR2LABEL[attr]] += 1
                        else:
                            attr_freq[self._ATTR2LABEL['other']] += 1
                        freq_sum += 1
        self.attr_support = {'per_class_frequency': attr_freq, 'tot_samples': freq_sum}

        """
        #print actions distribution
        print('_______________________')
        print('[ACTIONS DISTRIBUTION]:')
        tot_samples = self.act_support['tot_samples']
        for idx, freq in enumerate(self.act_support['per_class_frequency']):
            print('{}: \t\t[{}%]: {}'.format(self._LABEL2ACT[idx], round(100*freq/tot_samples), freq))
        print('Total support sum: {}'.format(tot_samples))
        print('_______________________')
        #print attributes distribution
        print('[ATTRIBUTES DISTRIBUTION]:')
        tot_samples = self.attr_support['tot_samples']
        for idx, freq in enumerate(self.attr_support['per_class_frequency']):
            print('{}: \t\t[{}%]: {}'.format(self._ATTRS[idx], round(100*freq/tot_samples), freq))
        print('Total support sum: {}'.format(tot_samples))
        print('_______________________')
        pdb.set_trace()
        """

        
