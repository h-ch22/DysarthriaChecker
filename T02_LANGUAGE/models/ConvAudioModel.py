import torch.nn as nn
import torch


class ConvAudioModel(nn.Module):
    def __init__(self):
        super(ConvAudioModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
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
        self.r = nn.ReLU()
        # self.bm = nn.BatchNorm1d(512)
        self.output = nn.Linear(512, 2)
        self.sig = nn.Sigmoid()

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(24),
        #     nn.ReLU(),
        #
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(48),
        #     nn.ReLU(),
        #
        #     nn.ConvTranspose2d(128, 30, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        a = torch.tensor(x.size(0))
        print(a.device)

        self.encoder[1] = nn.Conv2d(a, 128, kernel_size=3, stride=1, padding=1)  # encoder의 첫 번째 Conv2d 레이어를 초기화

        encoded = self.encoder(x)
        dense = self.dense(encoded)
        # decoded = self.decoder(dense)
        # bm = self.bm()
        output = self.output(self.r(dense))
        sig = self.sig(output)
        return sig
