import torch
from torch.utils.data import Dataset


class AudioDataSet(Dataset):
    def __init__(self, mfcc_list, labels):
        self.mfcc_list = mfcc_list
        self.labels = labels
        self.labels = list(map(self.subType_to_vector, self.labels))

    def __len__(self):
        return len(self.mfcc_list)

    def subType_to_vector(self, value):
        data = {25: 0, 26: 1}
        return data.get(value, None)

    def __getitem__(self, idx):
        mfcc = self.mfcc_list[idx]
        label = torch.LongTensor(self.labels)[idx]

        return mfcc, label
