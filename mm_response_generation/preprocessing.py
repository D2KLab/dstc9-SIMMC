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

from config import TrainConfig
from tools.simmc_dataset import SIMMCDatasetForResponseGeneration



class Collate():

    def __init__(self, word2id, item2id, unk_token):
        self.word2id = word2id
        self.item2id = item2id
        self.unk_token = unk_token

    def collate_fn(self, batch):
        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        history = [item[3] for item in batch]
        visual_context = [item[4] for item in batch]
        actions = [item[5] for item in batch]
        attributes = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        # words to ids for the current utterance
        utterance_seq_ids = []
        for item in batch:
            curr_seq = []
            for word in item[2].split():
                word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
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
                    curr_seq.append(word_id)
                curr_candidate.append(torch.tensor(curr_seq, dtype=torch.long))
            resp_ids.append(curr_candidate)

        #convert actions and attributes to word ids
        act_ids = []
        for act in actions:
            curr_seq = []
            for word in act.split():
                word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                curr_seq.append(word_id)
            act_ids.append(torch.tensor(curr_seq, dtype=torch.long))

        attr_ids = []
        for attrs in attributes:
            curr_attributes = []
            for attr in attrs:
                curr_seq = []
                for word in attr.split():
                    word_id = self.word2id[word] if word in self.word2id else self.word2id[self.unk_token]
                    curr_seq.append(word_id)
                curr_attributes.append(torch.tensor(curr_seq, dtype=torch.long))
            attr_ids.append(curr_attributes)

        # item to id for the visual context
        visual_ids = {'focus': [], 'history': []}
        for v in visual_context:
            curr_focus = self.item2id[v['focus']]
            curr_history = []
            for vv in v['history']:
                v_id = self.item2id[vv]
                curr_history.append(torch.tensor(v_id))
            visual_ids['focus'].append(torch.tensor(curr_focus))
            if len(curr_history):
                curr_history = torch.stack(curr_history)
            visual_ids['history'].append(curr_history)
        visual_ids['focus'] = torch.stack(visual_ids['focus'])

        assert len(utterance_seq_ids) == 1, 'Only unitary batch sizes allowed'
        assert len(utterance_seq_ids) == len(dial_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(turns), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(history_seq_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(resp_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(act_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(attr_ids), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(visual_ids['focus']), 'Batch sizes do not match'
        assert len(utterance_seq_ids) == len(visual_ids['history']), 'Batch sizes do not match'

        batch_dict = {}
        batch_dict['utterances'] = utterance_seq_ids
        batch_dict['history'] = history_seq_ids
        batch_dict['actions'] = act_ids
        batch_dict['attributes'] = attr_ids
        batch_dict['visual_context'] = visual_ids

        return dial_ids, turns, batch_dict, resp_ids


def save_data_on_file(iterator, save_path):
    dial_id_list = []
    turn_list = []
    utterance_list = []
    history_list = []
    actions_list = []
    attributes_list = []
    visual_context_list = {'focus': [], 'history': []}
    candidate_list = []
    for dial_ids, turns, batch, candidates_pool in iterator:
        dial_id_list.append(dial_ids[0])
        turn_list.append(turns[0])
        utterance_list.append(batch['utterances'][0])
        history_list.append(batch['history'][0])
        actions_list.append(batch['actions'][0])
        attributes_list.append(batch['attributes'][0])
        visual_context_list['focus'].append(batch['visual_context']['focus'][0])
        visual_context_list['history'].append(batch['visual_context']['history'][0])
        candidate_list.append(candidates_pool[0])
    
    torch.save(
        {
            'dial_ids': dial_id_list,
            'turns': turn_list,
            'utterances': utterance_list,
            'histories': history_list,
            'actions': actions_list,
            'attributes': attributes_list,
            'visual_contexts': visual_context_list,
            'candidates': candidate_list 
        },
        save_path
    )


def preprocess(train_dataset, dev_dataset, test_dataset, args):

    # prepare model's vocabulary
    train_vocabulary = train_dataset.get_vocabulary()
    dev_vocabulary = dev_dataset.get_vocabulary()
    test_vocabulary = test_dataset.get_vocabulary()

    vocabulary = train_vocabulary.union(dev_vocabulary)
    vocabulary = vocabulary.union(test_vocabulary)

    word2id = {}
    word2id[TrainConfig._PAD_TOKEN] = 0
    word2id[TrainConfig._UNK_TOKEN] = 1
    for idx, word in enumerate(vocabulary):
        word2id[word] = idx+2
    np.save(os.path.join('/'.join(args.train_folder.split('/')[:-1]), 'vocabulary.npy'), word2id)
    print('VOCABULARY SIZE: {}'.format(len(vocabulary)))

    raw_data = np.load(args.metadata_embeddings, allow_pickle=True)
    raw_data = dict(raw_data.item())
    item2id = {}
    for idx, item in enumerate(raw_data['item_ids']):
        item2id[item] = idx
    collate = Collate(word2id=word2id, item2id=item2id, unk_token=TrainConfig._UNK_TOKEN)
    # prepare DataLoader
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 0}
    trainloader = DataLoader(train_dataset, **params, collate_fn=collate.collate_fn)
    devloader = DataLoader(dev_dataset, **params, collate_fn=collate.collate_fn)
    testloader = DataLoader(test_dataset, **params, collate_fn=collate.collate_fn)

    start_t = time.time()

    save_data_on_file(iterator=trainloader, save_path=os.path.join(args.train_folder, 'response_retrieval_data.dat'))
    save_data_on_file(iterator=devloader, save_path=os.path.join(args.dev_folder, 'response_retrieval_data.dat'))
    save_data_on_file(iterator=testloader, save_path=os.path.join(args.test_folder, 'response_retrieval_data.dat'))

    end_t = time.time()
    h_count = (end_t-start_t) /60 /60
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('preprocessing time: {}h:{}m:{}s'.format(round(h_count), round(m_count), round(s_count)))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_folder",
        type=str,
        required=True,
        help="Path to training dataset JSON file")
    parser.add_argument(
        "--dev_folder",
        type=str,
        required=False,
        default=None,
        help="Path to training dataset JSON file")
    parser.add_argument(
        "--test_folder",
        type=str,
        required=False,
        default=None,
        help="Path to training dataset JSON file")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings file")
    parser.add_argument(
        "--metadata_embeddings",
        type=str,
        required=True,
        help="Path to metadata embeddings file")

    args = parser.parse_args()
    
    dataset_path = '{}/fashion_{}_dials.json'
    actions_path = '{}/fashion_{}_dials_api_calls.json'
    candidates_path = '{}/fashion_{}_dials_retrieval_candidates.json'

    train_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.train_folder, 'train'), 
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.train_folder, 'train'), 
                                                    candidates_path=candidates_path.format(args.train_folder, 'train'))
    dev_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.dev_folder, 'dev'),
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.dev_folder, 'dev'), 
                                                    candidates_path=candidates_path.format(args.dev_folder, 'dev'))
    test_dataset = SIMMCDatasetForResponseGeneration(data_path=dataset_path.format(args.test_folder, 'devtest'), 
                                                    metadata_path=args.metadata, 
                                                    actions_path=actions_path.format(args.test_folder, 'devtest'), 
                                                    candidates_path=candidates_path.format(args.test_folder, 'devtest'))

    preprocess(train_dataset, dev_dataset, test_dataset, args)