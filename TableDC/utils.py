import numpy as np

import torch
from torch.utils.data import Dataset



class load_data(Dataset):
    def __init__(self, dataset):

        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/label.txt'.format(dataset), dtype=int)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


