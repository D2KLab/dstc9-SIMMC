import argparse
import json
import os
import pdb

import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from models import BlindStatefulLSTM, BlindStatelessLSTM
from utilities import SIMMCDatasetForActionPrediction, plotting_loss

"""expected form for model output
    [
        {
            "dialog_id": ...,  
            "predictions": [
                {
                    "action": <predicted_action>,
                    "action_log_prob": {
                        <action_token>: <action_log_prob>,
                        ...
                    },
                    "attributes": {
                        <attribute_label>: <attribute_val>,
                        ...
                    }
                }
            ]
        }
    ]

    Where <attribute_label> is "focus" or "attributes" (only "attributes" for fashion dataset).
"""

id2act = SIMMCDatasetForActionPrediction._LABEL2ACT
id2attrs = SIMMCDatasetForActionPrediction._ATTRS


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


def create_eval_dict(dataset):
    eval_dict = {}
    for dial_id in dataset.id2dialog:
        num_turns = len(dataset.id2dialog[dial_id]['dialogue'])
        eval_dict[dial_id] = {'dialog_id': dial_id, 'predictions': [None] * num_turns}
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
        for curr_step, (dial_ids, turns, batch, actions, attributes) in enumerate(testloader):
            assert len(dial_ids) == 1, 'Only unitary batch size is allowed during testing'
            dial_id = dial_ids[0]
            turn = turns[0]

            batch['utterances'] = batch['utterances'].to(device)
            actions = actions.to(device)
            attributes = attributes.to(device)

            actions_out, attributes_out, actions_probs, attributes_probs = model(**batch, device=device)

            #get predicted action and arguments
            actions_predictions = torch.argmax(actions_probs, dim=-1)
            attributes_predictions = []
            for batch_idx, t in enumerate(attributes_probs):
                attributes_predictions.append([])
                for pos, val in enumerate(t):
                    if val >= .5:
                        attributes_predictions[batch_idx].append(pos)

            actions_predictions = actions_predictions[0].item()
            attributes_predictions = attributes_predictions[0]

            predicted_action = SIMMCDatasetForActionPrediction._LABEL2ACT[actions_predictions]
            action_log_prob = {}
            for idx, prob in enumerate(actions_probs[0]):
                action_log_prob[SIMMCDatasetForActionPrediction._LABEL2ACT[idx]] = torch.log(prob).item()
            attributes = {}
            #for arg in args_predictions:
            attributes['attributes'] = [SIMMCDatasetForActionPrediction._ATTRS[attr] for attr in attributes_predictions]
            
            eval_dict[dial_id]['predictions'][turn] = {'action': predicted_action, 
                                                        'action_log_prob': action_log_prob, 
                                                        'attributes': attributes}

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
        "--metadata",
        default=None,
        type=str,
        required=True,
        help="Path to metadata json file")
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
        "--actions",
        default=None,
        type=str,
        required=True,
        help="Path to training action annotations file")
    parser.add_argument(
        "--cuda",
        default=None,
        required=False,
        type=int,
        help="id of device to use")

    args = parser.parse_args()
    test_dataset = SIMMCDatasetForActionPrediction(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    eval_dict = create_eval_dict(test_dataset)
    print('EVAL DATASET: {}'.format(test_dataset))

    # prepare model
    vocabulary_test = test_dataset.get_vocabulary()
    print('VOCABULARY SIZE: {}'.format(len(vocabulary_test)))

    word2id = torch.load(args.vocabulary)

    model = instantiate_model(args, 
                                num_actions=len(SIMMCDatasetForActionPrediction._LABEL2ACT), 
                                num_attrs=len(SIMMCDatasetForActionPrediction._ATTRS), 
                                word2id=word2id)
    model.load_state_dict(torch.load(args.model_path))

    model_folder = '/'.join(args.model_path.split('/')[:-1])
    print('model loaded from {}'.format(model_folder))

    eval(model, test_dataset, args, save_folder=model_folder, device=device)
