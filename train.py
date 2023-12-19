import numpy as np
import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from model import Generator, Discriminator


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    learning_rate = 2e-4
    batch_size = 128
    image_size = 64
    channels_img = 1
    z_dim = 100
    num_epochs = 10
    num_workers = 2

    features_disc = 64
    features_gen = 64

    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
    ])

    dataset = datasets.MNIST(root="./datasets/", download=True, transform=trans)
    classes_count = len(dataset.classes)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    generator = Generator(z_dim, channels_img, features_gen, classes_count).to(device)
    discriminator = Discriminator(channels_img, features_disc, classes_count).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        real: torch.Tensor
        for i, (real, real_labels) in enumerate(loader):
            real = real.to(device)
            real_labels = real_labels.to(device)

            random_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_labels = torch.IntTensor(np.random.randint(0, classes_count, batch_size)).to(device)

            fake = generator(random_noise, fake_labels)

            # Train Discriminator
            discriminator.zero_grad()

            disc_real = discriminator(real, real_labels)
            loss_disc_real = loss_function(disc_real, torch.ones_like(disc_real))

            loss_disc_real.backward(retain_graph=True)
            discriminator_optimizer.step()

            disc_fake = discriminator(fake, fake_labels)
            loss_disc_fake = loss_function(disc_fake, torch.zeros_like(disc_fake))

            loss_disc_fake.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train Generator
            generator.zero_grad()

            disc_output = discriminator(fake, fake_labels)
            loss_gen = loss_function(disc_output, torch.ones_like(disc_output))

            loss_gen.backward()
            generator_optimizer.step()

            with torch.no_grad():
                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(loader)}")
                print(f"Loss D: {(loss_disc_real + loss_disc_fake) / 2:.4f}, Loss G: {loss_gen:.4f}")
                print()

                grid_size = 50
                row_length = classes_count
                row_count = grid_size // row_length

                noise = torch.randn(grid_size, z_dim, 1, 1).to(device)
                labels = torch.IntTensor(np.tile(np.arange(row_length), row_count)).to(device)

                fake = generator(noise, labels).to(device)

                img_grid = torchvision.utils.make_grid(fake, normalize=True, nrow=row_length)
                save_image(img_grid, f"./generated/grid.png")


if __name__ == "__main__":
    main()
