from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import dataset
from torchvision import transforms


class Tufts(dataset.Dataset):
    def __init__(self, folder: str, transform: Optional[transforms.Compose] = None):
        self.transform = transform
        self.files: List[Path] = [file for file in Path(folder).rglob("*.jpg")]
        self.classes: List[str] = [
            "0 - neutral",
            "1 - smiling",
            "2 - eyes closed",
            "3 - shocked",
            "4 - sunglasses",
        ]

    def __getitem__(self, i: int):
        file: Path = self.files[i]
        image: Image = Image.open(file)

        if self.transform is not None:
            image = self.transform(image)

        # Original is indexed 1-5, this makes it 0-4
        _, _, _, index = file.stem.split("_")
        img_class = int(index) - 1

        return image, torch.tensor(img_class)

    def __len__(self):
        return len(self.files)
