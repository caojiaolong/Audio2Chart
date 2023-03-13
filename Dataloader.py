"""
this file is used to:

1. define the dataset structure used in training or testing.
2. read the preprocessed data named "data.json", this file is a dictionary in shape {index: {"input": [beat_num, 1, 128, 87],"label": [beat_num, 256]}}

for example: to get a sample from data.json, we can use `data[index]["input"]` and `data[index]["label"]`
"""
import torch.utils.data


class dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(dataset, self).__init__()
        self.train = train
