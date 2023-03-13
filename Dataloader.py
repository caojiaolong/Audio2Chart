"""
this file is used to:

1. define the dataset structure used in training or testing.
2. read the preprocessed data named "data.json", this file is a dictionary in shape {index: {"input": [beat_num, 1, 128, 87],"label": [beat_num, 256]}}

for example: to get a sample from data.json, we can use `data[index]["input"]` and `data[index]["label"]`
"""
import torch.utils.data
import pickle

from torch.utils.data.dataset import T_co


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.data = data

    def __getitem__(self, index):
        return self.data[list(self.data.keys())[index]]

    def __len__(self):
        return len(self.data.keys())
