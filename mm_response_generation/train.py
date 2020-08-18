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
from models import BlindStatelessLSTM
#from models import BlindStatelessLSTM, BlindStatefulLSTM, MMStatefulLSTM
from tools import Logger, plotting_loss

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,5"  # specify which GPU(s) to be used


def instantiate_model(args, word2id):
    if args.model == 'blindstateless':
        return BlindStatelessLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False,
                                freeze_embeddings=True)
    elif args.model == 'blindstateful':
        return BlindStatefulLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False)
    elif args.model == 'mmstateful':
        return MMStatefulLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                item_embeddings_path=args.metadata_embeddings,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False)
    else:
        raise Exception('Model not present!')


def plotting(epochs, losses_trend, checkpoint_dir):
    epoch_list = np.arange(1, epochs+1)
    losses = [(losses_trend['train'], 'blue', 'train'), 
                (losses_trend['dev'], 'red', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'global_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Global loss trend', x_label='epochs', y_label='loss')


def forward_step(model, batch, candidates_pool, response_criterion, device):

    batch['utterances'] = batch['utterances'].to(device)

    matching_logits, matching_scores = model(**batch,
                                            candidates_pool=candidates_pool,
                                            device=device)

    matching_targets = torch.zeros(batch['utterances'].shape[0], 100).to(device)
    # the true response is always the first in the list of candidates
    matching_targets[:, 0] = 1.0
    response_loss = response_criterion(matching_logits, matching_targets)

    return response_loss


def train(train_dataset, dev_dataset, args, device):

    # prepare checkpoint folder
    curr_date = datetime.datetime.now().isoformat().split('.')[0]
    checkpoint_dir = os.path.join(TrainConfig._CHECKPOINT_FOLDER, curr_date)
    #os.makedirs(checkpoint_dir, exist_ok=True) #todo uncomment before first training
    # prepare logger to redirect both on file and stdout
    #sys.stdout = Logger(os.path.join(checkpoint_dir, 'train.log')) #todo uncomment before training
    print('device used: {}'.format(str(device)))
    print('batch used: {}'.format(args.batch_size))
    print('lr used: {}'.format(TrainConfig._LEARNING_RATE))
    print('weight decay: {}'.format(TrainConfig._WEIGHT_DECAY))

    print('TRAINING DATASET: {}'.format(train_dataset))
    print('VALIDATION DATASET: {}'.format(dev_dataset))

    # prepare model's vocabulary
    vocabulary_train = train_dataset.get_vocabulary()
    vocabulary_dev = dev_dataset.get_vocabulary()
    vocabulary = vocabulary_train.union(vocabulary_dev)
    word2id = {}
    word2id[TrainConfig._PAD_TOKEN] = 0
    word2id[TrainConfig._UNK_TOKEN] = 1
    for idx, word in enumerate(vocabulary):
        word2id[word] = idx+2
    #torch.save(word2id, os.path.join(checkpoint_dir, 'vocabulary.pkl')) #todo uncomment before first training
    print('VOCABULARY SIZE: {}'.format(len(vocabulary)))

    # prepare model
    model = instantiate_model(args, word2id=word2id)
    model.to(device)
    print('MODEL NAME: {}'.format(args.model))
    print('NETWORK: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': args.batch_size,
            'shuffle': False, #todo set to True
            'num_workers': 0}
    trainloader = DataLoader(train_dataset, **params, collate_fn=model.collate_fn)

    devloader = DataLoader(dev_dataset, **params, collate_fn=model.collate_fn)

    #prepare losses and optimizer
    response_criterion = torch.nn.BCEWithLogitsLoss().to(device) #pos_weight=torch.tensor(10.)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=TrainConfig._LEARNING_RATE, weight_decay=TrainConfig._WEIGHT_DECAY)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(range(10, args.epochs, 10)), gamma = 0.8)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=5, threshold=1e-3, cooldown=4, verbose=True)

    #prepare containers for statistics
    losses_trend = {'train': [], 
                    'dev': []}

    best_loss = math.inf
    start_t = time.time()
    #todo continue to adapt code from here
    for epoch in range(args.epochs):
        model.train()
        curr_epoch_losses = []
        # sorted_dial_ids, sorted_dial_turns, batch_dict, sorted_responses, sorted_distractors
        for curr_step, (dial_ids, turns, batch, candidates_pool) in enumerate(trainloader):
            pdb.set_trace()
            response_loss = forward_step(model, 
                                        batch=batch,
                                        candidates_pool=candidates_pool,
                                        response_criterion=response_criterion,
                                        device=device)
            #backward
            optimizer.zero_grad()
            response_loss.backward()
            optimizer.step()
            pdb.set_trace()

            curr_epoch_losses.append(response_loss.item())
        losses_trend['train'].append(np.mean(curr_epoch_losses))

        model.eval()
        curr_epoch_losses = []
        with torch.no_grad(): 
            for curr_step, (dial_ids, turns, batch, candidates_pool) in enumerate(devloader):

                response_loss = forward_step(model, 
                                            batch=batch,
                                            candidates_pool=candidates_pool,
                                            response_criterion=response_criterion,
                                            device=device)
                curr_epoch_losses.append(response_loss.item())

        losses_trend['dev'].append(np.mean(curr_epoch_losses))
        pdb.set_trace()
        # save checkpoint if best model
        if losses_trend['dev'][-1] < best_loss:
            best_loss = losses_trend['dev'][-1]
            torch.save(model.cpu().state_dict(),\
                       os.path.join(checkpoint_dir, 'state_dict.pt'))
            model.to(device)
        
        print('EPOCH #{} :: train_loss = {:.4f} ; dev_loss = {:.4f} ; (lr={})'.format(epoch+1, 
                                                                                    losses_trend['train'][-1], 
                                                                                    losses_trend['dev'][-1],
                                                                                    optimizer.param_groups[0]['lr']))
        scheduler1.step()
        scheduler2.step(losses_trend['dev'][-1])

    end_t = time.time()
    h_count = (end_t-start_t) /60 /60
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('training time: {}h:{}m:{}s'.format(round(h_count), round(m_count), round(s_count)))

    plotting(epochs=args.epochs, losses_trend=losses_trend, checkpoint_dir=checkpoint_dir)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=['blindstateless', 'blindstateful', 'mmstateful'],
        required=True,
        help="Type of the model (options: 'blindstateless', 'blindstateful', 'mmstateful')")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training dataset JSON file")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file")
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        help="Path to validation JSON file")
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
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to training candidates response annotations file")
    parser.add_argument(
        "--eval_candidates",
        type=str,
        required=True,
        help="Path to validation candidates response annotations file")    
    parser.add_argument(
        "--actions",
        type=str,
        required=True,
        help="Path to training action annotations file")
    parser.add_argument(
        "--eval_actions",
        type=str,
        required=True,
        help="Path to validation action annotations file")
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="Batch size")
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        help="Number of epochs")
    parser.add_argument(
        "--cuda",
        default=None,
        required=False,
        type=int,
        help="id of device to use")

    args = parser.parse_args()
    train_dataset = SIMMCDatasetForResponseGeneration(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions, candidates_path=args.candidates)
    dev_dataset = SIMMCDatasetForResponseGeneration(data_path=args.eval, metadata_path=args.metadata, actions_path=args.eval_actions, candidates_path=args.eval_candidates)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    train(train_dataset, dev_dataset, args, device)