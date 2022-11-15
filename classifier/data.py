from typing import Tuple, Any

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper

from .config import EClrConfig, DatasetName
from .transforms import wrap_transforms

CIFAR_LABELS_MAP = {
    DatasetName.CIFAR10: {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
}


class CifarDatasetWrapper(Dataset):
    def __init__(self, cifar: Dataset, dataset_name: DatasetName) -> None:
        super().__init__()
        self.cifar = cifar
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, index: int) -> Tuple[Any, Any, str]:
        img, target = self.cifar[index]
        return (
            img,
            target,
            f"{index}_label_{CIFAR_LABELS_MAP[self.dataset_name][target]}",
        )


def get_train_dataset(config: EClrConfig) -> Dataset:
    if config.dataset_name == DatasetName.CIFAR10:
        return CifarDatasetWrapper(
            datasets.CIFAR10(
                root=config.data_root,
                train=True,
                download=True,
                transform=wrap_transforms(config.train_transforms),
            ),
            dataset_name=config.dataset_name,
        )
    elif config.dataset_name == DatasetName.CIFAR100:
        return CifarDatasetWrapper(
            datasets.CIFAR100(
                root=config.data_root,
                train=True,
                download=True,
                transform=wrap_transforms(config.train_transforms),
            ),
            dataset_name=config.dataset_name,
        )
    else:
        raise ValueError(f"Given dataset {config.dataset_name} is not supported!")


def get_validation_dataset(config: EClrConfig) -> Dataset:
    if config.dataset_name == DatasetName.CIFAR10:
        return CifarDatasetWrapper(
            datasets.CIFAR10(
                root=config.data_root,
                train=False,
                download=True,
                transform=wrap_transforms(config.valid_transforms),
            ),
            dataset_name=config.dataset_name,
        )
    elif config.dataset_name == DatasetName.CIFAR100:
        return CifarDatasetWrapper(
            datasets.CIFAR100(
                root=config.data_root,
                train=False,
                download=True,
                transform=wrap_transforms(config.valid_transforms),
            ),
            dataset_name=config.dataset_name,
        )
    else:
        raise ValueError(f"Given dataset {config.dataset_name} is not supported!")


def get_train_loader(config: EClrConfig) -> DataLoader:
    loader = DataloaderSchedulerWrapper(
        auto_dataloader(
            get_train_dataset(config),
            num_workers=config.data_loader_workers,
            batch_size=config.batch_size * idist.get_world_size(),
            shuffle=config.shuffle_train,
            drop_last=False,
            persistent_workers=config.data_loader_workers > 0,
        ),
        single_pass_length=config.single_pass_length,
    )
    return loader


def get_validation_loader(config: EClrConfig) -> DataLoader:
    return auto_dataloader(
        get_validation_dataset(config),
        num_workers=config.data_loader_workers,
        batch_size=config.batch_size * idist.get_world_size(),
        shuffle=False,
        drop_last=False,
        persistent_workers=config.data_loader_workers > 0,
    )
