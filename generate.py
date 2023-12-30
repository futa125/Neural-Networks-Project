import torch
import torchvision
from matplotlib import pyplot as plt

from dataset import TFEIDCombined
from model import Generator
from train import EMBEDDING_SIZE, NUM_CHANNELS_IMAGE, NUM_CHANNELS_NOISE, NUM_FEATURES_GENERATOR


def main() -> None:
    label: int = 0

    while True:
        try:
            label = int(input("0 -> Anger\n"
                              "1 -> Contempt\n"
                              "2 -> Disgust\n"
                              "3 -> Fear\n"
                              "4 -> Happiness\n"
                              "5 -> Neutral\n"
                              "6 -> Sadness\n"
                              "7 -> Surprise\n"
                              "Enter the emotion you want to generate: ").strip())

            if label < 0 or label > 7:
                raise ValueError

            break

        except ValueError:
            print("Input must be a valid integer between 0 and 7.")
            print()

            continue

    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    num_classes: int = 8

    generator: Generator = Generator(
        num_channels_noise=NUM_CHANNELS_NOISE,
        num_channels_image=NUM_CHANNELS_IMAGE,
        num_features=NUM_FEATURES_GENERATOR,
        num_classes=num_classes,
        embedding_size=EMBEDDING_SIZE,
    ).to(device)
    generator.load_state_dict(state_dict=torch.load(
        f="./weights/2023-30-12-tfeid-combined-64x64-1-channel-1k-epoch-affine-true/generator.pth",
    ))
    generator.eval()

    batch_size: int = 3

    noise: torch.Tensor = torch.randn(batch_size, NUM_CHANNELS_NOISE, 1, 1).to(device)
    labels: torch.Tensor = torch.LongTensor([label] * batch_size).to(device)

    images: torch.Tensor = generator(noise, labels)

    # Grid needs to be on CPU for plotting to work
    grid: torch.Tensor = torchvision.utils.make_grid(tensor=images, normalize=True, nrow=batch_size).to("cpu")

    # Permute so that the channels are last
    plt.imshow(grid.permute(1, 2, 0))

    # White title is not visible on white background
    title = plt.title(TFEIDCombined.class_index_to_name(label))
    plt.setp(title, color="black")

    plt.show()


if __name__ == "__main__":
    main()
