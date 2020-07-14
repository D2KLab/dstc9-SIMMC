import argparse
import pdb

from models import BlindStatelessLSTM

from tools import SIMMCDataset, SIMMCDatasetForActionPrediction, print_sample_dialogue
from torch.utils.data import DataLoader


#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used

HIDDEN_SIZE = 300

def train(dataset, args):

    dataset[0]




    loader = DataLoader(dataset)
    vocabulary = dataset.get_vocabulary()
    model = BlindStatelessLSTM(args.embeddings, dataset_vocabulary=vocabulary, OOV_corrections=False, hidden_size=HIDDEN_SIZE)
    print('MODEL: {}'.format(model))
    pdb.set_trace()









if __name__ == '__main__':
    """Example

        python main.py \
        --data ../simmc/data/simmc_fashion/fashion_train_dials.json \
        --metadata ../simmc/data/simmc_fashion/fashion_metadata.json \
        --embeddings embeddings/glove.6B.50d.txt \
        --actions action_annotations/fashion_train_dials_api_calls.json
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

    args = parser.parse_args()
    dataset = SIMMCDatasetForActionPrediction(data_path=args.data, metadata_path=args.metadata, actions_path=args.actions)
    print(str(dataset))

    train(dataset, args)


    """
    slots = set()
    actions = set()
    for dialogue, coref in loader:
        print_sample_dialogue(dialogue, annotations=True)
        pdb.set_trace()
        for dial in dialogue:
            #belief state with more than one element
            #if len(dial['belief_state']) > 1:
            #    pdb.set_trace()

            #find all the slot types
            #for state in dial['belief_state']:
            #    for slot in state['slots']:
            #        slots.add(slot[0])

            #pdb.set_trace()

            #TODO find the API hierarchy
            #for state in dial['belief_state']:
            #    if len(state['act']) > 1:
            #        pdb.set_trace()
            #    if state['act'][0][:3] == 'ERR':
            #        pdb.set_trace()
            #    actions.add(state['act'][0])
            if len(dial['transcript']) > 1 or len(dial['system_transcript']) > 1:
                pdb.set_trace()
    print(slots)
    """
    pdb.set_trace()



