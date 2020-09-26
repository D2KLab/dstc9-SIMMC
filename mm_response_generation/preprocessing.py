import argparse
import datetime
import math
import os
import pdb
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('.')

from config import special_toks
from tools.simmc_dataset import SIMMCDatasetForResponseGeneration
from transformers import BertTokenizer



class Collate():

    ACT2STR = SIMMCDatasetForResponseGeneration._ACT2STR
    UNK_WORDS = set()

    def __init__(self, word2id, unk_token):
        self.word2id = word2id
        self.unk_token = unk_token


    def metadata2ids(self, processed_metadata, word2id, unk_token):
        unknown_words = set()
        metadata_ids = {}

        for item_id, item in processed_metadata.items():
            metadata_ids[int(item_id)] = []
            for field, values in item.items():
                curr_field = []
                for word in field.split():
                    if word not in word2id:
                        unknown_words.add(word)
                    curr_field.append(word2id[word] if word in word2id else unk_token)
                curr_values = []
                for value in values:
                    curr_value = []
                    for word in value.split():
                        if word not in word2id:
                            unknown_words.add(word)
                        curr_value.append(word2id[word] if word in word2id else unk_token)
                    curr_values.append(torch.tensor(curr_value))
                if len(curr_values):
                    curr_values = torch.cat(curr_values)
                else:
                    #insert none for field for which we do not have values
                    curr_values = torch.tensor([word2id['none']], dtype=torch.long)
            
                metadata_ids[int(item_id)].append((torch.tensor(curr_field, dtype=torch.long), curr_values))

        print('UNKNOWN METADATA WORDS: {}'.format(len(unknown_words)))
        return metadata_ids


    def collate_fn(self, batch):
        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        utterances = [item[2] for item in batch]
        history = [item[3] for item in batch]
        focus = [item[4] for item in batch]
        actions = [item[5] for item in batch]
        attributes = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        # words to ids for the current utterance
        utterance_seq_ids = []
        for utt in utterances:
            curr_seq = []
            for word in utt.split():
                word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                if word not in self.word2id:
                    self.UNK_WORDS.add(word)
                curr_seq.append(word_id)
            utterance_seq_ids.append(curr_seq)

        # words to ids for the history
        history_seq_ids = []
        for turn, item in zip(turns, history):
            assert len(item) == turn, 'Number of turns does not match history length'
            curr_turn_ids = []
            for t in range(turn):
                concat_sentences = item[t][0] + ' ' + item[t][1] #? separator token
                curr_seq = []
                for word in concat_sentences.split():
                    word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                    if word not in self.word2id:
                        self.UNK_WORDS.add(word)
                    curr_seq.append(word_id)
                curr_turn_ids.append(torch.tensor(curr_seq))
            history_seq_ids.append(curr_turn_ids)

        # convert response candidates to word ids
        resp_ids = []
        for resps in responses_pool:
            curr_candidate = []
            for resp in resps:
                curr_seq = []
                for word in resp.split():
                    word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                    if word not in self.word2id:
                        self.UNK_WORDS.add(word)
                    curr_seq.append(word_id)
                curr_candidate.append(torch.tensor(curr_seq, dtype=torch.long))
            resp_ids.append(curr_candidate)

        #convert actions and attributes to word ids
        act_ids = []
        for act in actions:
            curr_seq = []
            # todo collapse searchdatabase and searchmemory to one single action called search
            act_tokens = act.split() if 'search' not in act else ['search']
            for word in act_tokens:
                word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                if word not in self.word2id:
                    self.UNK_WORDS.add(word)
                curr_seq.append(word_id)
            act_ids.append(torch.tensor(curr_seq, dtype=torch.long))
        
        attr_ids = []
        for attrs in attributes:
            curr_attributes = []
            for attr in attrs:
                curr_seq = []
                for word in attr.split():
                    word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                    if word not in self.word2id:
                        self.UNK_WORDS.add(word)
                    curr_seq.append(word_id)
                curr_attributes.append(torch.tensor(curr_seq, dtype=torch.long))
            attr_ids.append(curr_attributes)

        assert len(utterance_seq_ids) == 1, 'Only unitary batch sizes allowed'
        assert len(utterance_seq_ids) == len(dial_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(turns), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(history_seq_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(resp_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(attr_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(focus)

        batch_dict = {}
        batch_dict['utterances'] = utterance_seq_ids
        batch_dict['history'] = history_seq_ids
        batch_dict['actions'] = act_ids
        batch_dict['attributes'] = attr_ids
        batch_dict['focus'] = focus[0] #only one focus per turn

        return dial_ids, turns, batch_dict, resp_ids



class BertCollate():
    def __init__(self, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.tokenizer_vocab = self.tokenizer.vocab
        self.bert2genid = {}
        self.bert2genid[self.tokenizer.convert_tokens_to_ids('[PAD]')] = 0
        self.bert2genid[self.tokenizer.convert_tokens_to_ids('[SEP]')] = 1
        self.bert2genid[self.tokenizer.convert_tokens_to_ids('[UNK]')] = 2
        self.avail_id = 3
        self.id_occur = [1, 1, 1]


    def add_tensor_ids_to_vocab(self, tensor_ids):
        ids = [id for id in tensor_ids.view(-1).tolist()]
        for id in ids:
            # skip the [CLS]. Never in the generated output
            if id == 101:
                continue
            if id not in self.bert2genid:
                self.bert2genid[id] = self.avail_id
                self.avail_id += 1
                self.id_occur.append(1)
            else:
                self.id_occur[self.bert2genid[id]] += 1


    def get_vocab_and_inv_frequencies(self):
        #avoid frequency computation for padding
        tot_sum = sum(self.id_occur[1:])
        word_inv_freqs = [tot_sum/occur for occur in self.id_occur[1:]]
        #insert 0 inverse frequency for padding
        word_inv_freqs.insert(0, 0)
        assert len(self.bert2genid) == len(word_inv_freqs)
        return self.bert2genid, word_inv_freqs


    def metadata2ids(self, processed_metadata):
        """Each item is represented by the plain string of all its attributes
                    'key1: val1, val2. key2: val1. ...'
        """

        id2pos = {}
        items_strings = []
        for idx, (item_id, item) in enumerate(processed_metadata.items()):
            id2pos[int(item_id)] = idx
            curr_item_strings = []
            for field, values in item.items():
                if len(values):
                    curr_str = '{}: {}'.format(field, ', '.join(values))
                else:
                    curr_str = '{}: {}'.format(field, 'none')
                curr_item_strings.append(curr_str)
            items_strings.append('. '.join(curr_item_strings))
        items_tensors = self.tokenizer(items_strings, padding='longest', return_tensors='pt')
        self.add_tensor_ids_to_vocab(items_tensors['input_ids'])

        res_dict = {'id2pos': id2pos, 'items_tensors': items_tensors}

        return res_dict


    def collate_fn(self, batch):

        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        utterances = [item[2] for item in batch]
        wizard_resp = [item[3] for item in batch]
        history = [item[4] for item in batch]
        focus = [item[5] for item in batch]
        actions = [item[6] for item in batch]
        attributes = [item[7][0] for item in batch]
        retr_candidates = [item[8] for item in batch]

        #each results has three keys: 'input_ids', 'token_type_ids', 'attention_mask'
        utterances_tensors = self.tokenizer(utterances, padding='longest', return_tensors='pt')
        self.add_tensor_ids_to_vocab(utterances_tensors['input_ids'])
        responses_tensors = self.tokenizer(wizard_resp, padding='longest', return_tensors='pt')
        self.add_tensor_ids_to_vocab(responses_tensors['input_ids'])
        history_seq_ids = []
        for turn, item in zip(turns, history):
            assert len(item) == turn, 'Number of turns does not match history length'
            if not len(item):
                no_history = {'input_ids': torch.zeros(utterances_tensors['input_ids'].shape[1]),
                            'token_type_ids': torch.zeros(utterances_tensors['input_ids'].shape[1]),
                            'attention_mask': torch.zeros(utterances_tensors['input_ids'].shape[1])}
                history_seq_ids.append(no_history)
                continue
            history_seq_ids.append(self.tokenizer(item, padding='longest', return_tensors='pt'))
        actions_tensors = self.tokenizer(actions, padding='longest', return_tensors='pt')
        all_candidates = [candidate for pool in retr_candidates for candidate in pool]
        candidates_tensors = self.tokenizer(all_candidates, padding='longest', return_tensors='pt')
        candidates_tensors = {'input_ids': candidates_tensors['input_ids'].view(len(dial_ids), 100, -1),
                            'token_type_ids': candidates_tensors['token_type_ids'].view(len(dial_ids), 100, -1),
                            'attention_mask': candidates_tensors['attention_mask'].view(len(dial_ids), 100, -1)}

        assert utterances_tensors['input_ids'].shape[0] == len(dial_ids), 'Batch sizes do not match'
        assert utterances_tensors['input_ids'].shape[0] == len(turns), 'Batch sizes do not match'
        assert utterances_tensors['input_ids'].shape[0] == responses_tensors['input_ids'].shape[0], 'Batch sizes do not match'
        assert utterances_tensors['input_ids'].shape[0] == len(history_seq_ids), 'Batch sizes do not match'
        assert utterances_tensors['input_ids'].shape[0] == actions_tensors['input_ids'].shape[0], 'Batch sizes do not match'
        assert utterances_tensors['input_ids'].shape[0] == len(attributes)
        assert utterances_tensors['input_ids'].shape[0] == candidates_tensors['input_ids'].shape[0]
        assert utterances_tensors['input_ids'].shape[0] == len(focus), 'Batch sizes do not match'

        data_dict = {}
        data_dict['utterances'] = utterances_tensors
        data_dict['responses'] = responses_tensors
        data_dict['history'] = history_seq_ids
        data_dict['actions'] = actions_tensors
        data_dict['attributes'] = attributes
        data_dict['focus'] = focus
        data_dict['candidates'] = candidates_tensors

        return dial_ids, turns, data_dict



def save_data_on_file(loader, save_path):

    dial_ids, turns, data_dict = iter(loader).next()
    
    
    torch.save(
        {
            'dial_ids': dial_ids,
            'turns': turns,
            'data_dict': data_dict,
        }, 
        save_path
    )



def preprocess(train_dataset, dev_dataset, test_dataset, args):

    save_path = '{}/{}'
    collate = BertCollate('bert-base-uncased')
    metadata_ids = collate.metadata2ids(train_dataset.processed_metadata)
    torch.save(metadata_ids, save_path.format(args.save_path, 'metadata_ids.dat'))

    # prepare DataLoader
    params = {'batch_size': len(train_dataset),
            'shuffle': False,
            'num_workers': 0}
    assert params['batch_size'] == len(train_dataset) and not params['shuffle'], 'Keep batch size to max and shuffle to False to avoid problems during training'
    trainloader = DataLoader(train_dataset, **params, collate_fn=collate.collate_fn)
    devloader = DataLoader(dev_dataset, **params, collate_fn=collate.collate_fn)
    testloader = DataLoader(test_dataset, **params, collate_fn=collate.collate_fn)

    start_t = time.time()

    save_data_on_file(loader=trainloader, save_path=save_path.format(args.save_path, 'train_response_retrieval_data.dat'))
    #save vocab and inverse word frequencies only for training data
    vocab, inv_freqs = collate.get_vocab_and_inv_frequencies()
    torch.save({'vocab': vocab, 'inv_freqs': torch.tensor(inv_freqs)}, save_path.format(args.save_path, 'generative_vocab.dat'))
    save_data_on_file(loader=devloader, save_path=save_path.format(args.save_path, 'dev_response_retrieval_data.dat'))
    save_data_on_file(loader=testloader, save_path=save_path.format(args.save_path, 'devtest_response_retrieval_data.dat'))

    #print('UNKNOWN DATASET WORDS: {}'.format(len(collate.UNK_WORDS)))

    end_t = time.time()
    h_count = (end_t-start_t) /60 /60
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('preprocessing time: {}h:{}m:{}s'.format(round(h_count), round(m_count), round(s_count)))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simmc_folder",
        type=str,
        required=True,
        help="Path to simmc fashion dataset folder")
    parser.add_argument(
        "--actions_folder",
        type=str,
        required=True,
        help="Path to simmc fashion actions folder")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save processed files")

    args = parser.parse_args()
    
    dataset_path = '{}/fashion_{}_dials.json'
    actions_path = '{}/fashion_{}_dials_api_calls.json'
    candidates_path = '{}/fashion_{}_dials_retrieval_candidates.json'

    train_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.simmc_folder, 'train'), 
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.actions_folder, 'train'), 
                                                    candidates_path=candidates_path.format(args.simmc_folder, 'train'))
    dev_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.simmc_folder, 'dev'),
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.actions_folder, 'dev'), 
                                                    candidates_path=candidates_path.format(args.simmc_folder, 'dev'))
    test_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.simmc_folder, 'devtest'), 
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.actions_folder, 'devtest'), 
                                                    candidates_path=candidates_path.format(args.simmc_folder, 'devtest'))

    preprocess(train_dataset, dev_dataset, test_dataset, args)
