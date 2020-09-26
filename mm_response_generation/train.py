import argparse
import datetime
import json
import math
import os
import pdb
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import model_conf, special_toks, train_conf
from dataset import FastDataset
from models import BlindStatelessLSTM, MMStatefulLSTM
from utilities import DataParallelV2, Logger, plotting_loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4,5"  # specify which GPU(s) to be used
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def instantiate_model(args, out_vocab, device):
    """
    if args.model == 'blindstateless':
        return BlindStatelessLSTM(word_embeddings_path=args.embeddings, 
                                pad_token=special_toks['pad_token'],
                                unk_token=special_toks['unk_token'],
                                seed=train_conf['seed'],
                                OOV_corrections=False,
                                freeze_embeddings=True)
    """

    if args.model == 'mmstateful':
        if args.from_checkpoint is not None:
            with open(os.path.join(args.from_checkpoint, 'state_dict.pt'), 'rb') as fp:
                state_dict = torch.load(fp)
            with open(os.path.join(args.from_checkpoint, 'model_conf.json'), 'rb') as fp:
                loaded_conf = json.load(fp)
            loaded_conf.pop('dropout_prob')
            model_conf.update(loaded_conf)

        model = MMStatefulLSTM(**model_conf,
                                seed=train_conf['seed'],
                                device=device,
                                out_vocab=out_vocab,
                                **special_toks)
        if args.from_checkpoint is not None:
            model.load_state_dict(state_dict)
            print('Model loaded from {}'.format(args.from_checkpoint))
        return model
    else:
        raise Exception('Model not present!')


def plotting(epochs, losses_trend, checkpoint_dir=None):
    epoch_list = np.arange(1, epochs+1)
    losses = [(losses_trend['train'], 'blue', 'train'), 
                (losses_trend['dev'], 'red', 'validation')]

    loss_path = os.path.join(checkpoint_dir, 'global_loss_plot') if checkpoint_dir is not None else None
    plotting_loss(x_values=epoch_list, save_path=loss_path, functions=losses, plot_title='Global loss trend', x_label='epochs', y_label='loss')


def move_batch_to_device(batch, device):
    for key in batch.keys():
        if key == 'history':
            raise Exception('Not implemented')
        batch[key] = batch[key].to(device)


def visualize_output(request, responses, item, id2word, genid2word, vocab_logits, device):
    shifted_targets = torch.cat((responses[:, 1:], torch.zeros((responses.shape[0], 1), dtype=torch.long).to(device)), dim=-1)
    rand_idx = random.randint(0, shifted_targets.shape[0]-1)
    eff_len = shifted_targets[rand_idx][shifted_targets[rand_idx] != 0].shape[0]
    """
    inp = ' '.join([id2word[inp_id.item()] for inp_id in responses[rand_idx] if inp_id != vocab['[PAD]']])
    print('input: {}'.format(inp))
    """
    req = ' '.join([id2word[req_id.item()] for req_id in request[rand_idx] if req_id != 0])
    print('user: {}'.format(req))

    out = ' '.join([id2word[out_id.item()] for out_id in shifted_targets[rand_idx] if out_id !=0])
    print('wizard: {}'.format(out))

    item = ' '.join([id2word[item_id.item()] for item_id in item[rand_idx] if item_id !=0])
    print('item: {}'.format(item))

    gens = torch.argmax(torch.nn.functional.softmax(vocab_logits, dim=-1), dim=-1)
    gen = ' '.join([genid2word[gen_id.item()] for gen_id in gens[:, :eff_len][rand_idx]])
    print('generated: {}'.format(gen))


def forward_step(model, batch, generative_targets, response_criterion, device):
    move_batch_to_device(batch, device)
    generative_targets = generative_targets.to(device)
    vocab_logits = model(**batch,
                        history=None,
                        actions=None,
                        attributes=None,
                        candidates=None,
                        candidates_mask=None,
                        candidates_token_type=None)
    #keep the loss outside the forward: complex to compute the mean with a weighted loss
    response_loss = response_criterion(vocab_logits.view(vocab_logits.shape[0]*vocab_logits.shape[1], -1), 
                                        generative_targets.view(vocab_logits.shape[0]*vocab_logits.shape[1]))
    
    p = random.randint(0, 9)
    if p > 8:
        try:
            vocab = model.vocab
            id2word = model.id2word
            genid2word = model.genid2word
        except:
            vocab = model.module.vocab
            id2word = model.module.id2word
            genid2word = model.module.genid2word
        visualize_output(request=batch['utterances'], responses=batch['responses'], item=batch['focus'], id2word=id2word, genid2word=genid2word, vocab_logits=vocab_logits, device=device)

    return response_loss


def train(train_dataset, dev_dataset, args, device):

    # prepare checkpoint folder
    if args.checkpoints:
        curr_date = datetime.datetime.now().isoformat().split('.')[0]
        checkpoint_dir = os.path.join(train_conf['ckpt_folder'], curr_date)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # prepare logger to redirect both on file and stdout
        sys.stdout = Logger(os.path.join(checkpoint_dir, 'train.log'))
        sys.stderr = Logger(os.path.join(checkpoint_dir, 'err.log'))
    print('device used: {}'.format(str(device)))
    print('batch used: {}'.format(args.batch_size))
    print('lr used: {}'.format(train_conf['lr']))
    print('weight decay: {}'.format(train_conf['weight_decay']))

    print('TRAINING DATASET: {}'.format(train_dataset))
    print('VALIDATION DATASET: {}'.format(dev_dataset))

    with open(args.generative_vocab, 'rb') as fp:
        gen_vocab = dict(torch.load(fp))
    bert2genid, inv_freqs = gen_vocab['vocab'], gen_vocab['inv_freqs']
    if args.checkpoints:
        torch.save(bert2genid, os.path.join(checkpoint_dir, 'bert2genid.pkl'))
    print('GENERATIVE VOCABULARY SIZE: {}'.format(len(bert2genid)))

    # prepare model
    #response_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, weight=inv_freqs/inv_freqs.sum()).to(device)
    response_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    model = instantiate_model(args, out_vocab=bert2genid, device=device)
    vocab = model.vocab
    if args.checkpoints:
        with open(os.path.join(checkpoint_dir, 'model_conf.json'), 'w+') as fp:
            json.dump(model_conf, fp)
    # work on multiple GPUs when available
    if torch.cuda.device_count() > 1:
        model = DataParallelV2(model)
    model.to(device)
    print('using {} GPU(s): {}'.format(torch.cuda.device_count(), os.environ["CUDA_VISIBLE_DEVICES"]))
    print('MODEL NAME: {}'.format(args.model))
    print('NETWORK: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': True}
    collate_fn = model.collate_fn if torch.cuda.device_count() <= 1 else model.module.collate_fn
    trainloader = DataLoader(train_dataset, **params, collate_fn=collate_fn)
    devloader = DataLoader(dev_dataset, **params, collate_fn=collate_fn)

    #prepare optimizer
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=train_conf['lr'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_conf['lr'], weight_decay=train_conf['weight_decay'])

    #scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(range(500, 500*5, 100)), gamma = 0.1)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(range(25, 100, 50)), gamma = 0.1)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=12, threshold=1e-3, cooldown=2, verbose=True)

    #prepare containers for statistics
    losses_trend = {'train': [], 
                    'dev': []}

    #candidates_pools_size = 100 if train_conf['distractors_sampling'] < 0 else train_conf['distractors_sampling'] + 1
    #print('candidates\' pool size: {}'.format(candidates_pools_size))
    #accumulation_steps = 8
    best_loss = math.inf
    global_step = 0
    start_t = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        model.train()
        curr_epoch_losses = []
        for batch_idx, (dial_ids, turns, batch, generative_targets) in enumerate(trainloader):
            global_step += 1
            step_start = time.time()
            response_loss = forward_step(model, 
                                        batch=batch,
                                        response_criterion=response_criterion,
                                        generative_targets=generative_targets,
                                        device=device)
            optimizer.zero_grad()
            #averaging losses from various GPUs by dividing by the batch size
            response_loss.mean().backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            step_end = time.time()
            h_count = (step_end-step_start) /60 /60
            m_count = ((step_end-step_start)/60) % 60
            s_count = (step_end-step_start) % 60
            print('step {}, loss: {}, time: {}h:{}m:{}s'.format(global_step, round(response_loss.mean().item(), 4), round(h_count), round(m_count), round(s_count)))

            """
            if (batch_idx+1) % accumulation_steps == 0: 
                optimizer.step()
                optimizer.zero_grad()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                #step_end = time.time()
                print('step {}, loss: {}'.format(global_step, round(response_loss.item()*accumulation_steps, 4)))
                p = random.randint(0, 9)
                if p > 8:
                    h_count = (step_end-step_start) /60 /60
                    m_count = ((step_end-step_start)/60) % 60
                    s_count = (step_end-step_start) % 60
                    print('step {}, loss: {}, time: {}h:{}m:{}s'.format(global_step, round(response_loss.mean().item(), 4), round(h_count), round(m_count), round(s_count)))
            """
            #scheduler1.step()
            #scheduler2.step(response_loss.item())
            curr_epoch_losses.append(response_loss.mean().item())
        losses_trend['train'].append(np.mean(curr_epoch_losses))

        model.eval()
        curr_epoch_losses = []
        with torch.no_grad(): 
            for curr_step, (dial_ids, turns, batch, generative_targets) in enumerate(devloader):
                response_loss = forward_step(model, 
                                            batch=batch,
                                            response_criterion=response_criterion,
                                            generative_targets=generative_targets,
                                            device=device)
                curr_epoch_losses.append(response_loss.mean().item())
        losses_trend['dev'].append(np.mean(curr_epoch_losses))
        # save checkpoint if best model
        if losses_trend['dev'][-1] < best_loss:
            best_loss = losses_trend['dev'][-1]
            if args.checkpoints:
                try:
                    state_dict = model.cpu().module.state_dict()
                except AttributeError:
                    state_dict = model.cpu().state_dict()
                torch.save(state_dict, os.path.join(checkpoint_dir, 'state_dict.pt'))
                #torch.save(model.cpu().state_dict(), os.path.join(checkpoint_dir, 'state_dict.pt'))
            model.to(device)
        ep_end = time.time()
        ep_h_count = (ep_end-ep_start) /60 /60
        ep_m_count = ((ep_end-ep_start)/60) % 60
        ep_s_count = (ep_end-ep_start) % 60
        time_str = '{}h:{}m:{}s'.format(round(ep_h_count), round(ep_m_count), round(ep_s_count))
        print('EPOCH #{} :: train_loss = {:.4f} ; dev_loss = {:.4f} ; (lr={}); --time: {}'.format(epoch+1, 
                                                                                    losses_trend['train'][-1], 
                                                                                    losses_trend['dev'][-1],
                                                                                    optimizer.param_groups[0]['lr'],
                                                                                    time_str))
        #TODO uncomment
        #scheduler1.step()
        scheduler2.step(losses_trend['dev'][-1])

    end_t = time.time()
    h_count = (end_t-start_t) /60 /60
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('training time: {}h:{}m:{}s'.format(round(h_count), round(m_count), round(s_count)))

    if not args.checkpoints:
        checkpoint_dir = None
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
        "--metadata_ids",
        type=str,
        required=True,
        help="Path to metadata ids file")
    parser.add_argument(
        "--generative_vocab",
        type=str,
        required=True,
        help="Path to generative vocabulary file")
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
        "--from_checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint to load")
    parser.add_argument(
        "--checkpoints",
        action='store_true',
        default=False,
        required=False,
        help="Flag to enable checkpoint saving for best model, logs and plots")
    parser.add_argument(
        "--cuda",
        action='store_true',
        default=False,
        required=False,
        help="flag to use cuda")

    args = parser.parse_args()
    if not args.checkpoints:
        print('************ NO CHECKPOINT SAVE !!! ************')

    train_dataset = FastDataset(dat_path=args.data, metadata_ids_path= args.metadata_ids, distractors_sampling=train_conf['distractors_sampling'])
    dev_dataset = FastDataset(dat_path=args.eval, metadata_ids_path= args.metadata_ids, distractors_sampling=train_conf['distractors_sampling']) #? sampling on eval
    print('TRAIN DATA LEN: {}'.format(len(train_dataset)))
    device = torch.device('cuda:0'.format(args.cuda) if torch.cuda.is_available() and args.cuda else 'cpu')

    train(train_dataset, dev_dataset, args, device)
