import argparse
import pdb

from tools import SIMMCDataset, SIMMCDatasetForActionPrediction
from torch.utils.data import DataLoader









if __name__ == '__main__':
    """Example

        python main.py \
        --data ../simmc/data/simmc_fashion/fashion_train_dials.json
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to dataset")

    args = parser.parse_args()
    dataset = SIMMCDatasetForActionPrediction(args.data)
    print(str(dataset))

    loader = DataLoader(dataset)

    #TODO meaning of belief state with length > 1 ??
    #TODO we always have only one act in the belief state (in the paper they said that multiple actions are possible sometimes)

    slots = set()
    actions = set()
    for dialogue, coref in loader:
        for dial in dialogue:
            #belief state with more than one element
            #if len(dial['belief_state']) > 1:
            #    pdb.set_trace()

            #find all the slot types
            #for state in dial['belief_state']:
            #    for slot in state['slots']:
            #        slots.add(slot[0])

            #TODO find the API hierarchy
            for state in dial['belief_state']:
                if len(state['act']) > 1:
                    pdb.set_trace()
                if state['act'][0][:3] == 'ERR':
                    pdb.set_trace()
                actions.add(state['act'][0])

    print(slots)
    pdb.set_trace()



