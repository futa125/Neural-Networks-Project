import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_channels_image: int, img_size: int, num_features: int, num_classes: int) -> None:
        super(Discriminator, self).__init__()

        self.img_size: int = img_size
        self.embed: nn.Embedding = nn.Embedding(num_classes, img_size*img_size)
        self.disc: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels_image + 1,
                out_channels=num_features,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            self._block(
                in_channels=num_features,
                out_channels=num_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 2,
                out_channels=num_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 4,
                out_channels=num_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 8,
                out_channels=num_features * 16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=num_features * 16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.embed(labels).view(*labels.shape, 1, self.img_size, self.img_size)
        x: torch.Tensor = torch.cat(tensors=(images, embedding), dim=1)

        return self.disc(x)


class Generator(nn.Module):
    def __init__(
            self,
            num_channels_noise: int,
            num_channels_image: int,
            num_features: int,
            num_classes: int,
            embedding_size: int,
    ) -> None:
        super(Generator, self).__init__()

        self.embed: nn.Embedding = nn.Embedding(num_classes, embedding_size)
        self.gen: nn.Sequential = nn.Sequential(
            self._block(
                in_channels=num_channels_noise + embedding_size,
                out_channels=num_features * 32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            self._block(
                in_channels=num_features * 32,
                out_channels=num_features * 16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 16,
                out_channels=num_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 8,
                out_channels=num_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self._block(
                in_channels=num_features * 4,
                out_channels=num_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=num_features * 2,
                out_channels=num_channels_image,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x: torch.Tensor = torch.cat(tensors=(noise, embedding), dim=1)

        return self.gen(x)
