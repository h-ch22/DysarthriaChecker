from torch.utils.data import Dataset


class AudioDataSet(Dataset):
    def __init__(self, figs, labels):
        self.figs = figs
        self.labels = labels
        self.labels = list(map(self.subType_to_vector, self.labels))

    def __len__(self):
        return len(self.figs)

    def subType_to_vector(self, value):
        data = {21: 0, 24: 1, 27: 2}
        return data.get(value, None)

    def __getitem__(self, idx):
        mfcc = self.figs[idx]
        label = self.labels[idx]

        return mfcc, label
