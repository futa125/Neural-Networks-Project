from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import Tufts
from model import Discriminator, Generator
from utils import gradient_penalty, initialize_weights

LEARNING_RATE_GENERATOR: float = 1e-4
LEARNING_RATE_DISCRIMINATOR: float = 1e-4
BETAS: Tuple[float, float] = (0.0, 0.9)

BATCH_SIZE: int = 64

# Editing image size requires changing the architecture of the network.
IMAGE_SIZE: int = 64

NUM_EPOCHS: int = 1000
NUM_CHANNELS_IMAGE: int = 1
NUM_CHANNELS_NOISE: int = 100
NUM_FEATURES_DISCRIMINATOR: int = 64
NUM_FEATURES_GENERATOR: int = 64
EMBEDDING_SIZE: int = 64

CRITIC_ITERATIONS: int = 5
LAMBDA: int = 10

GRID_ROWS = 10


def train(
        generator: Generator,
        discriminator: Discriminator,
        optimizer_generator: optim.Optimizer,
        optimizer_discriminator: optim.Optimizer,
        loader: DataLoader,
        device: str,
        num_classes: int,
):
    generator.train()
    discriminator.train()

    for epoch in range(NUM_EPOCHS):
        real_images: torch.Tensor
        real_labels: torch.Tensor

        for batch, (real_images, real_labels) in enumerate(loader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            discriminator_loss: Optional[torch.Tensor] = None
            for _ in range(CRITIC_ITERATIONS):
                noise: torch.Tensor = torch.randn(*real_labels.shape, NUM_CHANNELS_NOISE, 1, 1).to(device)
                fake_images: torch.Tensor = generator(noise, real_labels)

                discriminator_output_real: torch.Tensor = discriminator(real_images, real_labels)
                discriminator_output_fake: torch.Tensor = discriminator(fake_images, real_labels)

                gp: torch.Tensor = gradient_penalty(discriminator, real_labels, real_images, fake_images, device=device)
                discriminator_loss: torch.Tensor = (
                        -(torch.mean(discriminator_output_real) - torch.mean(discriminator_output_fake)) + LAMBDA * gp
                )

                discriminator.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                optimizer_discriminator.step()

            discriminator_output_fake: torch.Tensor = discriminator(fake_images, real_labels)
            generator_loss: torch.Tensor = -torch.mean(discriminator_output_fake)

            generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()

            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch {batch + 1}/{len(loader)} "
                  f"Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}")

        with torch.no_grad():
            noise: torch.Tensor = torch.randn(num_classes * GRID_ROWS, NUM_CHANNELS_NOISE, 1, 1).to(device)
            labels: torch.Tensor = torch.LongTensor(np.tile(range(num_classes), GRID_ROWS)).to(device)

            fake_images: torch.Tensor = generator(noise, labels)

            grid: torch.Tensor = torchvision.utils.make_grid(tensor=fake_images, normalize=True, nrow=num_classes)
            save_image(tensor=grid, fp=f"./generated/grid-epoch-{epoch + 1}.png")


def main() -> None:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    transform: transforms.Compose = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(NUM_CHANNELS_IMAGE)], [0.5 for _ in range(NUM_CHANNELS_IMAGE)])]
    )

    dataset: Tufts = Tufts(folder="./datasets/tufts", transform=transform)
    loader: DataLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(dataset.classes)

    generator: Generator = Generator(
        num_channels_noise=NUM_CHANNELS_NOISE,
        num_channels_image=NUM_CHANNELS_IMAGE,
        num_features=NUM_FEATURES_GENERATOR,
        num_classes=num_classes,
        embedding_size=EMBEDDING_SIZE,
    ).to(device)

    discriminator: Discriminator = Discriminator(
        num_channels_image=NUM_CHANNELS_IMAGE,
        img_size=IMAGE_SIZE,
        num_features=NUM_FEATURES_DISCRIMINATOR,
        num_classes=num_classes,
    ).to(device)

    try:
        generator.load_state_dict(torch.load("./weights/generator.pth"))
        discriminator.load_state_dict(torch.load("./weights/discriminator.pth"))

        print("Weights loaded from previous run.")
    except FileNotFoundError:
        initialize_weights(generator)
        initialize_weights(discriminator)

        print("New weights initialized.")

    optimizer_generator: optim.Adam = optim.Adam(
        params=generator.parameters(),
        lr=LEARNING_RATE_GENERATOR,
        betas=BETAS,
    )
    optimizer_discriminator: optim.Adam = optim.Adam(
        params=discriminator.parameters(),
        lr=LEARNING_RATE_DISCRIMINATOR,
        betas=BETAS,
    )

    try:
        train(
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            loader=loader,
            device=device,
            num_classes=num_classes,
        )
    except KeyboardInterrupt:
        torch.save(obj=generator.state_dict(), f="./weights/generator.pth")
        torch.save(obj=discriminator.state_dict(), f="./weights/discriminator.pth")


if __name__ == "__main__":
    main()
