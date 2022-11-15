from typing import List

import torchvision.transforms as tr


def wrap_transforms(transforms: List) -> tr.Compose:
    return tr.Compose(transforms=transforms + to_tensor())


def train_basic_augs() -> List:
    return [tr.RandomCrop(32, padding=4), tr.RandomHorizontalFlip()]


def to_tensor() -> List:
    return [tr.ToTensor()]


def to_tensor_normalized() -> List:
    return [
        tr.ToTensor(),
        tr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
