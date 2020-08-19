import numpy as np
import torch
from torch.utils.data import Dataset
import pdb


class FastDataset(Dataset):

    def __init__(self, npy_path):

        super(FastDataset, self).__init__()
        raw_data = np.load(npy_path, allow_pickle=True)
        self.data = dict(raw_data.item())
        pdb.set_trace()
