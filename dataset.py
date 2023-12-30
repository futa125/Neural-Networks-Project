from pathlib import Path
from typing import Dict, List, Optional

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

    def __getitem__(self, i: int) -> [Image, torch.Tensor]:
        file: Path = self.files[i]
        image: Image = Image.open(file)

        if self.transform is not None:
            image: Image = self.transform(image)

        # Original is indexed 1-5, this makes it 0-4
        _, _, _, index = file.stem.split("_")
        img_class: int = int(index) - 1

        return image, torch.tensor(img_class)

    def __len__(self) -> int:
        return len(self.files)


class TFEIDCombined(dataset.Dataset):
    def __init__(self, folder_high: str, folder_slight: str, transform: transforms.Compose):
        self.transform: transforms.Compose = transform

        self.files: List[Path] = [file for file in Path(folder_high).rglob("*.*")]
        self.files.extend([file for file in Path(folder_slight).rglob("*.*")])

        self.classes: List[str] = [
            "0 - anger",
            "1 - contempt",
            "2 - disgust",
            "3 - fear",
            "4 - happiness",
            "5 - neutral",
            "6 - sadness",
            "7 - surprise",
        ]
        self.class_name_to_index: Dict[str, int] = {
            "anger":     0,
            "contempt":  1,
            "disgust":   2,
            "fear":      3,
            "happiness": 4,
            "neutral":   5,
            "sadness":   6,
            "surprise":  7,
        }

    def __getitem__(self, i: int) -> [Image, torch.Tensor]:
        file: Path = self.files[i]
        image: Image = self.transform(Image.open(file))

        emotion_name: str
        _, emotion_name, _ = file.parent.name.split("_")

        return image, torch.tensor(self.class_name_to_index[emotion_name])

    def __len__(self) -> int:
        return len(self.files)
