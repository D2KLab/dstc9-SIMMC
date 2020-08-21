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
from models import BlindStatefulLSTM, BlindStatelessLSTM, MMStatefulLSTM
from utilities import Logger, plotting_loss
from dataset import FastDataset

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,5"  # specify which GPU(s) to be used


def instantiate_model(args, num_actions, num_attrs, word2id):
    if args.model == 'blindstateless':
        return BlindStatelessLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                num_actions=num_actions,
                                num_attrs=num_attrs,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False,
                                freeze_embeddings=True)
    elif args.model == 'blindstateful':
        return BlindStatefulLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                num_actions=num_actions,
                                num_attrs=num_attrs,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False)
    elif args.model == 'mmstateful':
        return MMStatefulLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                item_embeddings_path=args.metadata_embeddings,
                                num_actions=num_actions,
                                num_attrs=num_attrs,
                                pad_token=TrainConfig._PAD_TOKEN,
                                unk_token=TrainConfig._UNK_TOKEN,
                                seed=TrainConfig._SEED,
                                OOV_corrections=False)
    else:
        raise Exception('Model not present!')


def plotting(epochs, losses_trend, checkpoint_dir):
    epoch_list = np.arange(1, epochs+1)
    losses = [(losses_trend['train']['global'], 'blue', 'train'), 
                (losses_trend['dev']['global'], 'red', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'global_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Global loss trend', x_label='epochs', y_label='loss')

    losses = [(losses_trend['train']['actions'], 'green', 'train'), 
                (losses_trend['dev']['actions'], 'purple', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'actions_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Actions loss trend', x_label='epochs', y_label='loss')

    losses = [(losses_trend['train']['attributes'], 'orange', 'train'), 
                (losses_trend['dev']['attributes'], 'black', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'attributes_loss_plot')
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Arguments loss trend', x_label='epochs', y_label='loss')


def forward_step(model, batch, actions, attributes, actions_criterion, attributes_criterion, device):

    batch['utterances'] = batch['utterances'].to(device)
    actions_targets = actions.to(device)
    attributes_targets = attributes.to(device)
    """
    seq_lengths =  seq_lengths.to(device)
    """

    actions_logits, attributes_logits, actions_probs, attributes_probs = model(**batch, device=device)

    actions_loss = actions_criterion(actions_logits, actions_targets)
    attributes_targets = attributes_targets.type_as(actions_logits)
    attributes_loss = attributes_criterion(attributes_logits, attributes_targets)

    """ Not used
    actions_predictions = torch.argmax(actions_probs, dim=-1)
    attributes_predictions = []
    for batch_idx, t in enumerate(attributes_probs):
        attributes_predictions.append([])
        for pos, val in enumerate(t):
            if val >= .5:
                attributes_predictions[batch_idx].append(pos)
    """
    actions_predictions = None
    attributes_predictions = None

    return actions_loss, attributes_loss, actions_predictions, attributes_predictions


def train(train_dataset, dev_dataset, args, device):

    # prepare checkpoint folder
    curr_date = datetime.datetime.now().isoformat().split('.')[0]
    checkpoint_dir = os.path.join(TrainConfig._CHECKPOINT_FOLDER, curr_date)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # prepare logger to redirect both on file and stdout
    sys.stdout = Logger(os.path.join(checkpoint_dir, 'train.log')) #todo uncomment before training
    print('device used: {}'.format(str(device)))
    print('batch used: {}'.format(args.batch_size))
    print('lr used: {}'.format(TrainConfig._LEARNING_RATE))
    print('weight decay: {}'.format(TrainConfig._WEIGHT_DECAY))

    print('TRAINING DATASET: {}'.format(train_dataset))
    print('VALIDATION DATASET: {}'.format(dev_dataset))

    # prepare model's vocabulary
    with open(args.vocabulary, 'rb') as fp:
        vocabulary = np.load(fp, allow_pickle=True)
        vocabulary = dict(vocabulary.item())

    torch.save(vocabulary, os.path.join(checkpoint_dir, 'vocabulary.pkl'))
    print('VOCABULARY SIZE: {}'.format(len(vocabulary)))

    assert train_dataset.num_actions == dev_dataset.num_actions, 'Number of actions mismatch between train and dev dataset'
    assert train_dataset.num_attributes == dev_dataset.num_attributes, 'Number of actions mismatch between train and dev dataset'
    # prepare model
    model = instantiate_model(args,
                            num_actions=train_dataset.num_actions,
                            num_attrs=train_dataset.num_attributes,
                            word2id=vocabulary)
    model.to(device)
    print('MODEL NAME: {}'.format(args.model))
    print('NETWORK: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': args.batch_size,
            'shuffle': False, #todo set to True
            'num_workers': 0}
    trainloader = DataLoader(train_dataset, **params, collate_fn=model.collate_fn)
    devloader = DataLoader(dev_dataset, **params, collate_fn=model.collate_fn)

    #prepare loss weights
    act_per_class, act_tot_support = train_dataset.act_support['per_class_frequency'], train_dataset.act_support['tot_samples']
    attr_per_class, attr_tot_support = train_dataset.attr_support['per_class_frequency'], train_dataset.attr_support['tot_samples']
    #weights computed as negative_samples/positive_samples
    ce_weights = torch.tensor([(act_tot_support-class_support)/class_support for class_support in act_per_class])
    bce_weights = torch.tensor([(attr_tot_support-class_support)/class_support for class_support in attr_per_class])
    #prepare losses and optimizer
    actions_criterion = torch.nn.CrossEntropyLoss().to(device)
    attributes_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=bce_weights).to(device) #pos_weight=torch.tensor(10.)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=TrainConfig._LEARNING_RATE, weight_decay=TrainConfig._WEIGHT_DECAY)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(range(10, args.epochs, 10)), gamma = 0.8)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=5, threshold=1e-3, cooldown=4, verbose=True)

    #prepare containers for statistics
    losses_trend = {'train': {'global':[], 'actions': [], 'attributes': []}, 
                    'dev': {'global':[], 'actions': [], 'attributes': []}}

    best_loss = math.inf
    start_t = time.time()
    for epoch in range(args.epochs):
        model.train()
        curr_epoch_losses = {'global': [], 'actions': [], 'attributes': []}

        for curr_step, (dial_ids, turns, batch, actions, attributes) in enumerate(trainloader):
            actions_loss, attributes_loss, _, _ = forward_step(model, 
                                                                batch=batch,
                                                                actions=actions,
                                                                attributes=attributes,
                                                                actions_criterion=actions_criterion,
                                                                attributes_criterion=attributes_criterion,
                                                                device=device)
            #backward
            optimizer.zero_grad()
            loss = (actions_loss + attributes_loss)/2
            loss.backward()
            optimizer.step()

            curr_epoch_losses['global'].append(loss.item())
            curr_epoch_losses['actions'].append(actions_loss.item())
            curr_epoch_losses['attributes'].append(attributes_loss.item())
        losses_trend['train']['global'].append(np.mean(curr_epoch_losses['global']))
        losses_trend['train']['actions'].append(np.mean(curr_epoch_losses['actions']))
        losses_trend['train']['attributes'].append(np.mean(curr_epoch_losses['attributes']))

        model.eval()
        curr_epoch_losses = {'global': [], 'actions': [], 'attributes': []}
        with torch.no_grad(): 
            for curr_step, (dial_ids, turns, batch, actions, attributes) in enumerate(devloader):

                actions_loss, attributes_loss, _, _ = forward_step(model, 
                                                                    batch=batch,
                                                                    actions=actions,
                                                                    attributes=attributes,
                                                                    actions_criterion=actions_criterion,
                                                                    attributes_criterion=attributes_criterion,
                                                                    device=device)
                loss = (actions_loss + attributes_loss)/2

                curr_epoch_losses['global'].append(loss.item())
                curr_epoch_losses['actions'].append(actions_loss.item())
                curr_epoch_losses['attributes'].append(attributes_loss.item())
        losses_trend['dev']['global'].append(np.mean(curr_epoch_losses['global']))
        losses_trend['dev']['actions'].append(np.mean(curr_epoch_losses['actions']))
        losses_trend['dev']['attributes'].append(np.mean(curr_epoch_losses['attributes']))

        # save checkpoint if best model
        if losses_trend['dev']['global'][-1] < best_loss:
            best_loss = losses_trend['dev']['global'][-1]
            torch.save(model.cpu().state_dict(),\
                       os.path.join(checkpoint_dir, 'state_dict.pt'))
            model.to(device)
        
        print('EPOCH #{} :: train_loss = {:.4f} ; dev_loss = {:.4f} [act_loss={:.4f}, attr_loss={:.4f}]; (lr={})'
                            .format(epoch+1, losses_trend['train']['global'][-1], 
                                    losses_trend['dev']['global'][-1],
                                    losses_trend['dev']['actions'][-1],
                                    losses_trend['dev']['attributes'][-1],
                                    optimizer.param_groups[0]['lr']))
        scheduler1.step()
        scheduler2.step(losses_trend['dev']['global'][-1])

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
        help="Path to preprocessed training data file .dat")
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        help="Path to preprocessed eval data file .dat")
    parser.add_argument(
        "--vocabulary",
        type=str,
        required=True,
        help="Path to vocabulary file")
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
    train_dataset = FastDataset(dat_path=args.data)
    dev_dataset = FastDataset(dat_path=args.eval)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    train(train_dataset, dev_dataset, args, device)
