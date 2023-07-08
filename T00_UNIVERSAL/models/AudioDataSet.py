from torch.utils.data import Dataset


class AudioDataSet(Dataset):
    def __init__(self, figs, labels):
        self.figs = figs
        self.labels = labels

    def __len__(self):
        return len(self.figs)

    def __getitem__(self, idx):
        mfcc = self.figs[idx]
        label = self.labels[idx]

        return mfcc, label
