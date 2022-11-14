import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper

from .config import EClrConfig, DatasetName
from .transforms import wrap_transforms


def get_train_dataset(config: EClrConfig) -> Dataset:
    if config.dataset_name == DatasetName.CIFAR10:
        return datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=True,
            transform=wrap_transforms(config.train_transforms),
        )
    elif config.dataset_name == DatasetName.CIFAR100:
        return datasets.CIFAR100(
            root=config.data_root,
            train=True,
            download=True,
            transform=wrap_transforms(config.train_transforms),
        )
    else:
        raise ValueError(f"Given dataset {config.dataset_name} is not supported!")


def get_validation_dataset(config: EClrConfig) -> Dataset:
    if config.dataset_name == DatasetName.CIFAR10:
        return datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=True,
            transform=wrap_transforms(config.valid_transforms),
        )
    elif config.dataset_name == DatasetName.CIFAR100:
        return datasets.CIFAR100(
            root=config.data_root,
            train=False,
            download=True,
            transform=wrap_transforms(config.valid_transforms),
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
