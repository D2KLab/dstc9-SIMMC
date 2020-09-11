import argparse
import json
import os
import pdb
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append('.')

from config import special_toks, train_conf
from dataset import FastDataset
from models import BlindStatelessLSTM, MMStatefulLSTM
from tools.simmc_dataset import SIMMCDatasetForResponseGeneration


"""expected form for model output
    [
	{
		"dialog_id": <dialog_id>,
		"candidate_scores": [
			<list of 100 scores for 100 candidates for round 1>
			<list of 100 scores for 100 candidates for round 2>
			...
		]
	}
	...
    ]
"""


def instantiate_model(args, word2id, model_configurations, device):
    if args.model == 'blindstateless':
        return BlindStatelessLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                pad_token=special_toks['pad_token'],
                                unk_token=special_toks['unk_token'],
                                seed=train_conf['seed'],
                                OOV_corrections=False,
                                freeze_embeddings=True)
    elif args.model == 'mmstateful':
        return MMStatefulLSTM(word_embeddings_path=args.embeddings, 
                                word2id=word2id,
                                seed=train_conf['seed'],
                                device=device,
                                retrieval_eval=args.retrieval_eval,
                                mode='inference',
                                **special_toks,
                                **model_configurations)
    else:
        raise Exception('Model not present!')


def create_eval_dict(dataset):
    dataset.create_id2turns()
    eval_dict = {}
    for dial_id, num_turns in dataset.id2turns.items():
        eval_dict[dial_id] = {'dialog_id': dial_id, 'candidate_scores': []}
    return eval_dict


def remove_dataparallel(load_checkpoint_path):
    # original saved file with DataParallel
    state_dict = torch.load(load_checkpoint_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


def move_batch_to_device(batch, device):
    batch['utterances'] = batch['utterances'].to(device)
    batch['utterances_mask'] = batch['utterances_mask'].to(device)
    for h_idx in range(len(batch['history'])):
        if len(batch['history'][h_idx]):
            batch['history'][h_idx] = batch['history'][h_idx].to(device)
    for i_idx in range(len(batch['focus_items'])):
        batch['focus_items'][i_idx][0] =  batch['focus_items'][i_idx][0].to(device)
        batch['focus_items'][i_idx][1] =  batch['focus_items'][i_idx][1].to(device)
    #batch['seq_lengths'] = batch['seq_lengths'].to(device)


def eval(model, test_dataset, args, save_folder, device):

    model.eval()
    model.to(device)
    print('MODEL: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 0}
    testloader = DataLoader(test_dataset, **params, collate_fn=model.collate_fn)

    eval_dict = create_eval_dict(test_dataset)
    with torch.no_grad():
        for curr_step, (dial_ids, turns, batch, candidates_pool, candidates_padding_mask) in enumerate(testloader):
            assert len(dial_ids) == 1, 'Only unitary batch size is allowed during testing'
            dial_id = dial_ids[0]
            turn = turns[0]

            move_batch_to_device(batch, device)
            candidates_pool = candidates_pool.to(device)
            candidates_padding_mask = candidates_padding_mask.to(device)

            res = model(**batch, 
                            candidates_pool=candidates_pool,
                            pools_padding_mask=candidates_padding_mask)
            if args.retrieval_eval:
                responses = res[0]
                scores = res[1]
            else:
                scores = res

            #get retrieved response index in the pool
            eval_dict[dial_id]['candidate_scores'].append(scores.squeeze(0).tolist())

    eval_list = []
    for key in eval_dict:
        eval_list.append(eval_dict[key])
    save_file = os.path.join(save_folder, 'eval_out.json')
    try:
        with open(save_file, 'w+') as fp:
            json.dump(eval_list, fp)
        print('results saved in {}'.format(save_file))
    except:
        print('Error in writing the resulting JSON')



if __name__ == '__main__':
    #TODO make "infer": dataset with unknown labels (modify the dataset class)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=['blindstateless', 'blindstateful', 'mmstateful'],
        required=True,
        help="Type of the model (options: 'blindstateless', 'blindstateful', 'mmstateful')")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to the weights of the model")
    parser.add_argument(
        "--vocabulary",
        default=None,
        type=str,
        required=True,
        help="Path to the vocabulary pickle file")
    parser.add_argument(
        "--model_conf",
        default=None,
        type=str,
        required=True,
        help="Path to the model configuration JSON file")     
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to training dataset json file")
    parser.add_argument(
        "--embeddings",
        default=None,
        type=str,
        required=True,
        help="Path to embedding file")
    parser.add_argument(
        "--metadata_ids",
        type=str,
        required=True,
        help="Path to metadata ids file")
    parser.add_argument(
        "--retrieval_eval",
        action='store_true',
        default=False,
        required=False,
        help="Flag to enable retrieval evaluation")
    parser.add_argument(
        "--cuda",
        default=None,
        required=False,
        type=int,
        help="id of device to use")

    start_t = time.time()

    args = parser.parse_args()
    test_dataset = FastDataset(dat_path=args.data, metadata_ids_path= args.metadata_ids)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    eval_dict = create_eval_dict(test_dataset)
    print('EVAL DATASET: {}'.format(test_dataset))

    # prepare model
    word2id = torch.load(args.vocabulary)
    with open(args.model_conf) as fp:
        model_configurations = json.load(fp)
    #pdb.set_trace()
    model = instantiate_model(args, 
                            word2id=word2id,
                            model_configurations=model_configurations,
                            device=device)
    #model.load_state_dict(remove_dataparallel(args.model_path))
    model.load_state_dict(torch.load(args.model_path))

    model_folder = '/'.join(args.model_path.split('/')[:-1])
    print('model loaded from {}'.format(model_folder))

    eval(model, test_dataset, args, save_folder=model_folder, device=device)

    end_t = time.time()
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('training time: {}m:{}s'.format(round(m_count), round(s_count)))
