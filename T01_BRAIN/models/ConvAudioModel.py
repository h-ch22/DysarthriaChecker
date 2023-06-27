import torch.nn as nn


class ConvAudioModel(nn.Module):
    def __init__(self):
        super(ConvAudioModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(20, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=0)
        )

        self.dense = nn.Linear(30720, 512)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(24),
        #     nn.ReLU(),
        #
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(48),
        #     nn.ReLU(),
        #
        #     nn.ConvTranspose2d(128, 20, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        encoded = self.encoder(x)
        dense = self.dense(encoded)
        decoded = self.decoder(dense)

        return decoded
