import datetime
import argparse
import math
import pdb
import time
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import BlindStatelessLSTM
from tools import (SIMMCDataset, SIMMCDatasetForActionPrediction,
                   SIMMCFashionConfig, TrainConfig, plotting_loss,
                   Logger)

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used



def plotting(epochs, losses_trend, checkpoint_dir):
    epoch_list = np.arange(1, TrainConfig._N_EPOCHS+1)
    losses = [(losses_trend['train']['global'], 'blue', 'train'), 
                (losses_trend['dev']['global'], 'red', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'global_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Global loss trend', x_label='epochs', y_label='loss')

    losses = [(losses_trend['train']['actions'], 'green', 'train'), 
                (losses_trend['dev']['actions'], 'purple', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'actions_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Actions loss trend', x_label='epochs', y_label='loss')

    losses = [(losses_trend['train']['args'], 'orange', 'train'), 
                (losses_trend['dev']['args'], 'black', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'args_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Arguments loss trend', x_label='epochs', y_label='loss')



def forward_step(model, batch, actions_targets, args_targets, device, actions_criterion, args_criterion, seq_lengths=None):

    batch = batch.to(device)
    actions_targets = actions_targets.to(device)
    args_targets = args_targets.to(device)
    seq_lengths =  seq_lengths.to(device)

    actions_out, args_out = model(batch, seq_lengths)

    actions_loss = actions_criterion(actions_out, actions_targets)
    args_targets = args_targets.type_as(actions_out)
    args_loss = args_criterion(args_out, args_targets)

    predictions = -1 #todo implement
    return actions_loss, args_loss, predictions







def train(train_dataset, dev_dataset, args, device):

    # prepare checkpoint folder
    curr_date = datetime.datetime.now().isoformat().split('.')[0]
    checkpoint_dir = os.path.join(TrainConfig._CHECKPOINT_FOLDER, curr_date)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # prepare logger to redirect both on file and stdout
    sys.stdout = Logger(os.path.join(checkpoint_dir, 'train.log'))
    print('device used: {}'.format(str(device)))
    print('batch used: {}'.format(args.batch_size))

    print('TRAINING DATASET: {}'.format(train_dataset))
    print('VALIDATION DATASET: {}'.format(dev_dataset))

    # prepare model
    vocabulary_train = train_dataset.get_vocabulary()
    vocabulary_dev = dev_dataset.get_vocabulary()
    vocabulary = vocabulary_train.union(vocabulary_dev)
    print('VOCABULARY SIZE: {}'.format(len(vocabulary)))

    model = BlindStatelessLSTM(args.embeddings, 
                                dataset_vocabulary=vocabulary, 
                                OOV_corrections=False, 
                                num_actions=SIMMCFashionConfig._FASHION_ACTION_NO,
                                num_args=SIMMCFashionConfig._FASHION_ARGS_NO,
                                pad_token=TrainConfig._PAD_TOKEN)
    model.to(device)
    print('MODEL: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0}
    trainloader = DataLoader(train_dataset, **params, collate_fn=model.collate_fn)

    devloader = DataLoader(dev_dataset, **params, collate_fn=model.collate_fn)

    #prepare loss and optimizer
    actions_criterion = torch.nn.CrossEntropyLoss().to(device) #todo set weights based on dataset balancing
    args_criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=TrainConfig._LEARNING_RATE) #todo weight_decay=0.1

    #prepare containers for statistics
    losses_trend = {'train': {'global':[], 'actions': [], 'args': []}, 
                    'dev': {'global':[], 'actions': [], 'args': []}}

    best_loss = math.inf
    start_t = time.time()
    for epoch in range(TrainConfig._N_EPOCHS):
        model.train()
        curr_epoch_losses = {'global': [], 'actions': [], 'args': []}

        for curr_step, (batch, actions, seq_lengths, args) in enumerate(trainloader):
            #pdb.set_trace()
            #todo send also the arguments and predict them
            actions_loss, args_loss, _ = forward_step(model, batch, 
                                        actions_targets=actions, 
                                        args_targets=args,
                                        actions_criterion=actions_criterion,
                                        args_criterion=args_criterion,
                                        seq_lengths=seq_lengths,
                                        device=device)
            #backward
            optimizer.zero_grad()
            loss = (actions_loss + args_loss)/2
            loss.backward()
            optimizer.step()

            curr_epoch_losses['global'].append(loss.item())
            curr_epoch_losses['actions'].append(actions_loss.item())
            curr_epoch_losses['args'].append(args_loss.item())
        losses_trend['train']['global'].append(np.mean(curr_epoch_losses['global']))
        losses_trend['train']['actions'].append(np.mean(curr_epoch_losses['actions']))
        losses_trend['train']['args'].append(np.mean(curr_epoch_losses['args']))

        model.eval()
        curr_epoch_losses = {'global': [], 'actions': [], 'args': []} 
        for curr_step, (batch, actions, seq_lengths, args) in enumerate(devloader):

            actions_loss, args_loss, predictions = forward_step(model, batch, 
                                                                actions_targets=actions, 
                                                                args_targets=args,
                                                                actions_criterion=actions_criterion,
                                                                args_criterion=args_criterion,
                                                                seq_lengths=seq_lengths,
                                                                device=device)
            loss = (actions_loss + args_loss)/2

            curr_epoch_losses['global'].append(loss.item())
            curr_epoch_losses['actions'].append(actions_loss.item())
            curr_epoch_losses['args'].append(args_loss.item())
        losses_trend['dev']['global'].append(np.mean(curr_epoch_losses['global']))
        losses_trend['dev']['actions'].append(np.mean(curr_epoch_losses['actions']))
        losses_trend['dev']['args'].append(np.mean(curr_epoch_losses['args']))

        # save checkpoint if best model
        if losses_trend['dev']['global'][-1] < best_loss:
            #todo save checkpoint
            best_loss = losses_trend['dev']['global'][-1]
            torch.save(model.cpu().state_dict(),\
                       os.path.join(checkpoint_dir, 'state_dict.pt'))
            model.to(device)
            pass
        
        print('EPOCH #{} :: train_loss = {:.4f} ; dev_loss = {:.4f}'
                            .format(epoch+1, losses_trend['train']['global'][-1], losses_trend['dev']['global'][-1]))


    end_t = time.time()
    h_count = (end_t-start_t) /60 /60
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60
    print('training time: {}h:{}m:{}s'.format(round(h_count), round(m_count), round(s_count)))

    plotting(epochs=TrainConfig._N_EPOCHS, losses_trend=losses_trend, checkpoint_dir=checkpoint_dir)




if __name__ == '__main__':
    """Example

        python main.py \
        --data ../simmc/data/simmc_fashion/fashion_train_dials.json \
        --metadata ../simmc/data/simmc_fashion/fashion_metadata.json \
        --eval ../simmc/data/simmc_fashion/fashion_dev_dials.json\
        --embeddings embeddings/glove.6B.50d.txt \
        --actions annotations/fashion_train_dials_api_calls.json \
        --eval_actions annotations/fashion_dev_dials_api_calls.json\
        --batch_size 16\
        --cuda 0
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to training dataset json file")
    parser.add_argument(
        "--metadata",
        default=None,
        type=str,
        required=True,
        help="Path to metadata json file")
    parser.add_argument(
        "--eval",
        default=None,
        type=str,
        required=True,
        help="Path to validation json file")
    parser.add_argument(
        "--embeddings",
        default=None,
        type=str,
        required=True,
        help="Path to embedding file")
    parser.add_argument(
        "--actions",
        default=None,
        type=str,
        required=True,
        help="Path to training action annotations file")
    parser.add_argument(
        "--eval_actions",
        default=None,
        type=str,
        required=True,
        help="Path to validation action annotations file")
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="batch size")
    parser.add_argument(
        "--cuda",
        default=None,
        required=False,
        type=int,
        help="id of device to use")

    args = parser.parse_args()
    train_dataset = SIMMCDatasetForActionPrediction(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions)
    dev_dataset = SIMMCDatasetForActionPrediction(data_path=args.eval, metadata_path=args.metadata, actions_path=args.eval_actions)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    train(train_dataset, dev_dataset, args, device)
