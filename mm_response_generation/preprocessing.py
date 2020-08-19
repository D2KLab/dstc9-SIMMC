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

from config import TrainConfig
from dataset import SIMMCDatasetForResponseGeneration
from models import MMStatefulLSTM




def save_data_on_file(iterator, save_path):
    dial_id_list = []
    turn_list = []
    utterance_list = []
    history_list = []
    actions_list = []
    attributes_list = []
    visual_context_list = {'focus': [], 'history': []}
    seq_lengths_list = []
    candidate_list = []
    for dial_ids, turns, batch, candidates_pool in iterator:
        dial_id_list.append(dial_ids[0])
        turn_list.append(turns)
        utterance_list.append(batch['utterances'][0])
        history_list.append(batch['history'][0])
        actions_list.append(batch['actions'][0])
        attributes_list.append(batch['attributes'][0])
        visual_context_list['focus'].append(batch['visual_context']['focus'][0])
        visual_context_list['history'].append(batch['visual_context']['history'][0])
        seq_lengths_list.append(batch['seq_lengths'][0])
        candidate_list.append(candidates_pool[0])
    
    np.save(
        save_path,
        {
            'dial_ids': dial_id_list,
            'turns': turn_list,
            'utterances': utterance_list,
            'histories': history_list,
            'actions': actions_list,
            'attributes': attributes_list,
            'visual_contexts': visual_context_list,
            'seq_lengths': seq_lengths_list,
            'candidates': candidate_list 
        }
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
    torch.save(word2id, os.path.join('/'.join(args.train_folder.split('/')[:-1]), 'vocabulary.pkl'))
    print('VOCABULARY SIZE: {}'.format(len(vocabulary)))

    # the model is used to only preprocess the data to be fed to him (word2id and pad packed sequence)
    model = MMStatefulLSTM(word_embeddings_path=args.embeddings, 
                            word2id=word2id,
                            item_embeddings_path=args.metadata_embeddings,
                            pad_token=TrainConfig._PAD_TOKEN,
                            unk_token=TrainConfig._UNK_TOKEN,
                            seed=TrainConfig._SEED,
                            OOV_corrections=False)

    # prepare DataLoader
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 0}
    trainloader = DataLoader(train_dataset, **params, collate_fn=model.collate_fn)
    devloader = DataLoader(dev_dataset, **params, collate_fn=model.collate_fn)
    testloader = DataLoader(test_dataset, **params, collate_fn=model.collate_fn)

    start_t = time.time()

    save_data_on_file(iterator=trainloader, save_path=os.path.join(args.train_folder, 'data.npy'))
    save_data_on_file(iterator=devloader, save_path=os.path.join(args.dev_folder, 'data.npy'))
    save_data_on_file(iterator=testloader, save_path=os.path.join(args.test_folder, 'data.npy'))

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
