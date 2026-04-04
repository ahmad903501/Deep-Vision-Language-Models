from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from src.utils.config import DataConfig
from src.utils.io import ensure_dir


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    shape: Tuple[int, int, int]
    value_range: Tuple[float, float]
    real_grid_path: Path


def get_transform(scale_to_minus_one_to_one: bool) -> transforms.Compose:
    tx = [transforms.ToTensor()]
    if scale_to_minus_one_to_one:
        tx.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    return transforms.Compose(tx)


def _dataset_factory(name: str):
    registry = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
    }
    if name not in registry:
        raise ValueError(f"Unsupported dataset: {name}. Use MNIST or FashionMNIST.")
    return registry[name]


def _check_tensor_range(batch: torch.Tensor, scale_to_minus_one_to_one: bool) -> Tuple[float, float]:
    observed_min = float(batch.min().item())
    observed_max = float(batch.max().item())

    if scale_to_minus_one_to_one:
        lower, upper = -1.0, 1.0
    else:
        lower, upper = 0.0, 1.0

    eps = 1e-5
    if observed_min < lower - eps or observed_max > upper + eps:
        raise ValueError(
            f"Input range check failed. Observed [{observed_min:.4f}, {observed_max:.4f}] "
            f"expected within [{lower:.1f}, {upper:.1f}]"
        )
    return observed_min, observed_max


def _save_real_grid(images: torch.Tensor, path: Path, scale_to_minus_one_to_one: bool) -> None:
    grid = make_grid(images[:64], nrow=8)
    if scale_to_minus_one_to_one:
        # Store reference grid in visible image space while preserving model-space range.
        save_image(grid, path, normalize=True, value_range=(-1.0, 1.0))
    else:
        save_image(grid, path, normalize=False)


def build_dataloaders(config: DataConfig, output_root: str, seed: int) -> DataBundle:
    transform = get_transform(config.scale_to_minus_one_to_one)
    dataset_cls = _dataset_factory(config.dataset)

    full_train: Dataset = dataset_cls(
        root=config.data_root,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset: Dataset = dataset_cls(
        root=config.data_root,
        train=False,
        transform=transform,
        download=True,
    )

    val_size = 5000
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    first_batch, _ = next(iter(train_loader))
    value_range = _check_tensor_range(first_batch, config.scale_to_minus_one_to_one)

    image_dir = ensure_dir(Path(output_root) / "images")
    real_grid_path = image_dir / "real_grid_8x8.png"
    _save_real_grid(first_batch, real_grid_path, config.scale_to_minus_one_to_one)

    shape = (first_batch.shape[1], first_batch.shape[2], first_batch.shape[3])
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        shape=shape,
        value_range=value_range,
        real_grid_path=real_grid_path,
    )


def make_overfit_subset(dataset: Dataset, subset_size: int) -> Subset:
    if subset_size > len(dataset):
        raise ValueError(f"subset_size={subset_size} exceeds dataset size={len(dataset)}")
    indices = list(range(subset_size))
    return Subset(dataset, indices)
