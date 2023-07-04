import torch.nn as nn
import torch


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()

        self.dropout = nn.Dropout2d()

        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.conv_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.pool_1 = nn.AvgPool2d(kernel_size=2)

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.pool_2 = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.relu(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = torch.relu(x)
        x = self.pool_2(x)

        x = self.flatten(x)

        return x
