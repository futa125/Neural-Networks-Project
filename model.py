import torch

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, channels_img: int, features_d: int, classes_count: int):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(classes_count, features_d * features_d)

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_d * 1, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.embedding(labels)
        c = c.view(*labels.shape, 1, 64, 64)

        x = torch.cat((x, c), dim=1)

        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim: int, channels_img: int, features_g: int, classes_count: int):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(classes_count, classes_count)

        self.gen = nn.Sequential(
            self._block(z_dim + 10, features_g * 16, 4, 2, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.embedding(y)
        c = c.view(*c.shape, 1, 1)

        x = torch.cat((z, c), dim=1)

        return self.gen(x)
