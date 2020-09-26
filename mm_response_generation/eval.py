import argparse
import json
import os
import pdb
import sys
import time
import string

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


def instantiate_model(args, model_configurations, out_vocab, device):
    if args.model == 'blindstateless':
        return BlindStatelessLSTM(word_embeddings_path=args.embeddings, 
                                pad_token=special_toks['pad_token'],
                                unk_token=special_toks['unk_token'],
                                seed=train_conf['seed'],
                                OOV_corrections=False,
                                freeze_embeddings=True)
    elif args.model == 'mmstateful':
        return MMStatefulLSTM(**model_configurations,
                                seed=train_conf['seed'],
                                device=device,
                                out_vocab=out_vocab,
                                retrieval_eval=args.retrieval_eval,
                                beam_size=args.beam_size,
                                mode='inference',
                                **special_toks,
                                )
    else:
        raise Exception('Model not present!')


def create_eval_dicts(dataset):
    dataset.create_id2turns()
    gen_eval_dict = {}
    retr_eval_dict = {}
    for dial_id, num_turns in dataset.id2turns.items():
        gen_eval_dict[dial_id] = {'dialog_id': dial_id, 'predictions': []}
        retr_eval_dict[dial_id] = {'dialog_id': dial_id, 'candidate_scores': []}
    return gen_eval_dict, retr_eval_dict


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
    for key in batch.keys():
        if key == 'history':
            raise Exception('Not implemented')
        batch[key] = batch[key].to(device)
    """
    for h_idx in range(len(batch['history'])):
        if len(batch['history'][h_idx]):
            batch['history'][h_idx] = batch['history'][h_idx].to(device)
    for i_idx in range(len(batch['focus_items'])):
        batch['focus_items'][i_idx][0] =  batch['focus_items'][i_idx][0].to(device)
        batch['focus_items'][i_idx][1] =  batch['focus_items'][i_idx][1].to(device)
    batch['seq_lengths'] = batch['seq_lengths'].to(device)
    """


def visualize_result(utt_ids, item_ids, id2word, gen_ids=None):
    """
    keys = []
    vals = []
    for key, val in zip(item_ids[0], item_ids[1]):
        keys.append(' '.join([id2word[id.item()] for id in key if id != 0]))
        vals.append(' '.join([id2word[id.item()] for id in val if id != 0]))
    item = ['{}: {}'.format(key, val) for key, val in zip(keys, vals)]
    """

    item = [id2word[id.item()] for id in item_ids if id != 0]
    words_request = [id2word[id.item()] for id in utt_ids if id != 0]
    if gen_ids is not None:
        words_resp = [id2word[id] for id in gen_ids]
    #cleaned_req = clean_response(words_request)
    #cleaned_resp = clean_response(words_resp)
    print('USER: {}'.format(words_request))
    if gen_ids is not None:
        print('GEN: {}'.format(words_resp))
    print('Item: {}'.format(item))


def eval(model, test_dataset, args, save_folder, device):

    model.eval()
    model.to(device)
    print('MODEL: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 0}
    testloader = DataLoader(test_dataset, **params, collate_fn=model.collate_fn)

    gen_eval_dict, retr_eval_dict = create_eval_dicts(test_dataset)
    with torch.no_grad():
        for curr_step, (dial_ids, turns, batch) in enumerate(testloader):
            assert len(dial_ids) == 1, 'Only unitary batch size is allowed during testing'
            dial_id = dial_ids[0]
            turn = turns[0]

            move_batch_to_device(batch, device)
            #visualize_result(batch['utterances'][0], batch['focus'][0], model.id2word)
            res = model(**batch,
                        history=None,
                        actions=None,
                        attributes=None)
            response = res[0]['string']
            if args.retrieval_eval:
                scores = res[-1]

            #visualize_result(batch['utterances'][0], batch['focus_items'][0], id2word, responses)
            gen_eval_dict[dial_id]['predictions'].append({'response': response})
            retr_eval_dict[dial_id]['candidate_scores'].append(scores.squeeze(0).tolist())
            #todo here adjust candidates scores based on semantic attribute informations

    retr_eval_list = []
    gen_eval_list = []
    for key in retr_eval_dict:
        retr_eval_list.append(retr_eval_dict[key])
        gen_eval_list.append(gen_eval_dict[key])
    save_file = os.path.join(save_folder, 'eval_retr.json')
    try:
        with open(save_file, 'w+') as fp:
            json.dump(retr_eval_list, fp)
        print('retrieval results saved in {}'.format(save_file))
    except:
        print('Error in writing the resulting JSON')
    save_file = os.path.join(save_folder, 'eval_gen.json')
    try:
        with open(save_file, 'w+') as fp:
            json.dump(gen_eval_list, fp)
        print('generation results saved in {}'.format(save_file))
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
        "--model_conf",
        default=None,
        type=str,
        required=True,
        help="Path to the model configuration JSON file")
    parser.add_argument(
        "--vocabulary",
        default=None,
        type=str,
        required=True,
        help="Path to output vocabulary for the model")     
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
        "--beam_size",
        type=int,
        required=True,
        help="Size of the beam for the beam search at inference time")
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
    test_dataset = FastDataset(dat_path=args.data,
                                metadata_ids_path= args.metadata_ids,
                                retrieval=args.retrieval_eval)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    print('EVAL DATASET: {}'.format(test_dataset))

    # prepare model
    #word2id = torch.load(args.vocabulary)
    with open(args.model_conf) as fp:
        model_configurations = json.load(fp)
    with open(args.vocabulary, 'rb') as fp:
        bert2genid = torch.load(fp)

    model = instantiate_model(args,
                            model_configurations=model_configurations,
                            out_vocab=bert2genid,
                            device=device)
    model.load_state_dict(torch.load(args.model_path))
    """
    try:
        #if the model was not trained with DataParallel then an exception will be raised
        model.load_state_dict(remove_dataparallel(args.model_path))
    except:
        model.load_state_dict(torch.load(args.model_path))
    """

    model_folder = '/'.join(args.model_path.split('/')[:-1])
    print('model loaded from {}'.format(model_folder))

    eval(model, test_dataset, args, save_folder=model_folder, device=device)

    end_t = time.time()
    m_count = ((end_t-start_t)/60) % 60
    s_count = (end_t-start_t) % 60

    print('evaluation time: {}m:{}s'.format(round(m_count), round(s_count)))
