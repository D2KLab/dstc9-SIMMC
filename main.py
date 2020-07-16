import argparse
import pdb
import time

import torch
from torch.utils.data import DataLoader

from models import BlindStatelessLSTM
from tools import (SIMMCDataset, SIMMCDatasetForActionPrediction, 
                    SIMMCFashionConfig, TrainConfig)

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used

HIDDEN_SIZE = 300

def train(dataset, args, device):

    print('DATASET: {}'.format(dataset))
    
    # prepare model
    vocabulary = dataset.get_vocabulary()
    model = BlindStatelessLSTM(args.embeddings, dataset_vocabulary=vocabulary, OOV_corrections=False, 
                                num_labels=SIMMCFashionConfig._FASHION_ACTION_NO, hidden_size=HIDDEN_SIZE,
                                pad_token = TrainConfig._PAD_TOKEN, device=device)
    model.to(device)
    print('MODEL: {}'.format(model))

    # prepare DataLoader
    params = {'batch_size': TrainConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}
    loader = DataLoader(dataset, **params, collate_fn=model.collate_fn)
    #loader = DataLoader(dataset, **params)

    start_t = time.time()
    for epoch in range(10):
        for curr_step, (batch, targets, seq_lengths) in enumerate(loader):
            pdb.set_trace()
            targets = targets.to(device)





    end_t = time.time()
    h_count = (end_t-start_t)/60/60
    print('training time: '+str(h_count)+'h')




if __name__ == '__main__':
    """Example

        python main.py \
        --data ../simmc/data/simmc_fashion/fashion_train_dials.json \
        --metadata ../simmc/data/simmc_fashion/fashion_metadata.json \
        --embeddings embeddings/glove.6B.50d.txt \
        --actions action_annotations/fashion_train_dials_api_calls.json \
        --cuda
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to dataset json file")
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
        "--actions",
        default=None,
        type=str,
        required=True,
        help="Path to action annotations file")
    parser.add_argument(
        "--cuda",
        default=False,
        required=False,
        action='store_true',
        help="Path to action annotations file")

    args = parser.parse_args()
    dataset = SIMMCDatasetForActionPrediction(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    train(dataset, args, device)

