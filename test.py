import time

import numpy as np
import torch
import torchvision
from torchvision.utils import save_image

from model import Generator
from train import CHANNELS_IMG, FEATURES_GEN, GEN_EMBEDDING, IMAGE_SIZE, Z_DIM, device

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, 10, IMAGE_SIZE, GEN_EMBEDDING).to(device)

gen.load_state_dict(torch.load("weights"))

gen.eval()

start = time.time()
noise = torch.randn(100, Z_DIM, 1, 1).to(device)
labels = torch.IntTensor(np.tile(np.array(range(10)), 10)).to(device)
out = gen(noise, labels)

img_grid = torchvision.utils.make_grid(out, nrow=10)

save_image(img_grid, f"./generated/test.png")
