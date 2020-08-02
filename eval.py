import argparse

from torch.utils.data import DataLoader

from models import BlindStatelessLSTM
from tools import (SIMMCDatasetForActionPrediction, SIMMCFashionConfig,
                   TrainConfig, plotting_loss)


def eval(test_dataset, args, device):

    print('EVAL DATASET: {}'.format(test_dataset))

    # prepare model
    vocabulary_test = train_dataset.get_vocabulary()
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
            'shuffle': False,
            'num_workers': 0}
    testloader = DataLoader(test_dataset, **params, collate_fn=model.collate_fn)


    for curr_step, (batch, actions, seq_lengths, args) in enumerate(trainloader):
        batch = batch.to(device)
        actions = actions.to(device)
        args = args.to(device)

        actions_out, args_out, actions_probs, args_probs = model(batch, seq_lengths)
        
        pass




if __name__ == '__main__':
    """Example

        python main.py \
        --model checkpoints/2020-08-02T21:47:10\
        --data ../simmc/data/simmc_fashion/fashion_devtrest_dials.json \
        --metadata ../simmc/data/simmc_fashion/fashion_metadata.json \
        --actions annotations/fashion_devtest_dials_api_calls.json \
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
    test_dataset = SIMMCDatasetForActionPrediction(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else "cpu")

    eval(test_dataset, args, device)
