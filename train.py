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

    features_disc = 64
    features_gen = 64

    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
    ])

    dataset = datasets.MNIST(root="", download=True, transform=trans)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(z_dim, channels_img, features_gen).to(device)
    discriminator = Discriminator(channels_img, features_disc).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        real: torch.Tensor
        for i, (real, _) in enumerate(loader):
            real = real.to(device)

            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)

            # Train Discriminator
            disc_real = discriminator(real).reshape(-1)
            loss_disk_real = loss_function(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake).reshape(-1)
            loss_disc_fake = loss_function(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disk_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train Generator
            disc_output = discriminator(fake).reshape(-1)
            loss_gen = loss_function(disc_output, torch.ones_like(disc_output))

            generator.zero_grad()
            loss_gen.backward()
            generator_optimizer.step()

            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(loader)}")
            print(f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
            print()

            with torch.no_grad():
                fake = generator(fixed_noise)

                img_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
                save_image(img_grid, f"./generated/fake_grid.png")


if __name__ == "__main__":
    main()
