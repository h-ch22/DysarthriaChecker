import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(512, 128)
        self.dense_2 = nn.Linear(128, 512)
        self.dense_3 = nn.Linear(512, 128)
        self.dense_4 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = torch.sigmoid(x)

        return x
