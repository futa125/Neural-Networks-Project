import torch
import torch.nn as nn

from model import Discriminator


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(tensor=module.weight.data, mean=0.0, std=0.02)


def gradient_penalty(
        discriminator: Discriminator,
        labels: torch.Tensor,
        real: torch.Tensor,
        fake: torch.Tensor,
        device: str,
) -> torch.Tensor:
    batch_size: int
    channels: int
    height: int
    width: int

    batch_size, channels, height, width = real.shape

    alpha: torch.Tensor = torch.rand(batch_size, 1, 1, 1).repeat(1, channels, height, width).to(device)
    interpolated_images: torch.Tensor = real * alpha + fake * (1 - alpha)

    scores: torch.Tensor = discriminator(interpolated_images, labels)

    gradient: torch.Tensor
    gradient, = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )
    gradient: torch.Tensor = gradient.view(batch_size, -1)
    gradient_norm: torch.Tensor = gradient.norm(p=2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)
