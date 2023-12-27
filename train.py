import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LEARNING_RATE_DISC = 1e-4
LEARNING_RATE_GEN = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
GEN_EMBEDDING = 128
Z_DIM = 128
NUM_EPOCHS = 500
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

def main():
    trans = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    dataset = datasets.ImageFolder(root="datasets/ck+", transform=trans)

    NUM_CLASSES = len(dataset.classes)

    # comment mnist above and uncomment below for training on CelebA
    # dataset = datasets.ImageFolder(root="datasets/celeb", transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_DISC, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(NUM_CLASSES * 10, Z_DIM, 1, 1).to(device)
    fixed_labels = torch.IntTensor(np.tile(np.array(range(NUM_CLASSES)), 10)).to(device)

    writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
    writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, labels) in enumerate(tqdm(loader)):
            real = real.to(device)
            labels = labels.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 1 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise, fixed_labels)

                    # take out (up to) 32 examples
                    # img_grid_real = torchvision.utils.make_grid(real[:50], normalize=True, nrow=10)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True, nrow=NUM_CLASSES)

                    # save_image(img_grid_real, f"./generated/real.png")
                    save_image(img_grid_fake, f"./generated/fake.png")

                step += 1


    torch.save(gen.state_dict(), "weights")

if __name__ == "__main__":
    main()
