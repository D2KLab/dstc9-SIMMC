import argparse
import json
import os
import pdb
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('.')

from config import TrainConfig
from dataset import FastDataset
from models import BlindStatelessLSTM
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


def create_eval_dict(dataset):
    dataset.create_id2turns()
    eval_dict = {}
    for dial_id, num_turns in dataset.id2turns.items():
        eval_dict[dial_id] = {'dialog_id': dial_id, 'candidate_scores': []}
    return eval_dict


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
        for curr_step, (dial_ids, turns, batch, candidates_pool) in enumerate(testloader):
            assert len(dial_ids) == 1, 'Only unitary batch size is allowed during testing'
            dial_id = dial_ids[0]
            turn = turns[0]

            batch['utterances'] = batch['utterances'].to(device)

            _, matching_scores = model(**batch, candidates_pool=candidates_pool, device=device)

            #get retrieved response index in the pool
            #retrieved_response_idx = torch.argmax(matching_scores, dim=-1)
            eval_dict[dial_id]['candidate_scores'].append(matching_scores.squeeze(0).tolist())

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
        "--metadata_embeddings",
        type=str,
        required=True,
        help="Path to metadata embeddings file")
    parser.add_argument(
        "--cuda",
        default=None,
        required=False,
        type=int,
        help="id of device to use")

    args = parser.parse_args()
    test_dataset = FastDataset(dat_path=args.data)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    eval_dict = create_eval_dict(test_dataset)
    print('EVAL DATASET: {}'.format(test_dataset))

    # prepare model
    word2id = torch.load(args.vocabulary)

    model = instantiate_model(args, 
                            word2id=word2id)
    model.load_state_dict(torch.load(args.model_path))

    model_folder = '/'.join(args.model_path.split('/')[:-1])
    print('model loaded from {}'.format(model_folder))

    eval(model, test_dataset, args, save_folder=model_folder, device=device)
