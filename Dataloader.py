"""
this file is used to:

1. define the dataset structure used in training or testing.
2. read the preprocessed data named "data.json", this file is a dictionary in shape {index: {"input": [beat_num, 1, 128, 87],"label": [beat_num, 256]}}

for example: to get a sample from data.json, we can use `data[index]["input"]` and `data[index]["label"]`
"""
import torch.utils.data
import pickle


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super(Dataset, self).__init__()
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data.keys())