import torch
import utils
import numpy as np
from torch.utils import data


class ImgTrain(data.Dataset):
    def __init__(self, x1_data,patch_size):
        self.x1 = x1_data
        self.patch_size = patch_size
        assert patch_size <= x1_data.shape[1]

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        x1 = self.x1[item]
        H,W = x1.shape
        ox = np.random.randint(low = 0, high = H - self.patch_size)
        oy = np.random.randint(low = 0, high = W - self.patch_size)
        x1 = x1[ox:ox + self.patch_size, oy:oy + self.patch_size]
        return x1


class ImgVal(data.Dataset):
    def __init__(self, x1_data):
        self.x1 = x1_data

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        x1 = self.x1[item]
        return x1


def loader_train(x1,patch_size,batch_size):
    return data.DataLoader(
        dataset=ImgTrain(x1,patch_size),
        batch_size = batch_size,
        shuffle = True
    )

def loader_val(x1,batch_size):
    return data.DataLoader(
        dataset=ImgVal(x1),
        batch_size = batch_size,
        shuffle = False
    )