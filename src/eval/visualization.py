from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from src.utils.io import ensure_dir


def save_tensor_grid(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    grid = make_grid(images[: nrow * nrow], nrow=nrow)
    save_image(grid, path, normalize=True, value_range=value_range)


def save_trajectory_grids(
    trajectory: Dict[int, torch.Tensor],
    output_dir: str | Path,
    nrow: int = 8,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    target_dir = ensure_dir(output_dir)
    for step, images in sorted(trajectory.items(), reverse=True):
        save_tensor_grid(
            images=images,
            output_path=target_dir / f"trajectory_step_{step:04d}.png",
            nrow=nrow,
            value_range=value_range,
        )


def plot_schedule(alpha_bars: torch.Tensor, snr: torch.Tensor, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)

    x = list(range(len(alpha_bars)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, alpha_bars.cpu().numpy())
    axes[0].set_title("alpha_bar(t)")
    axes[0].set_xlabel("timestep")

    axes[1].plot(x, snr.cpu().numpy())
    axes[1].set_yscale("log")
    axes[1].set_title("SNR(t)")
    axes[1].set_xlabel("timestep")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_loss(losses: list[float], output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_title("Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
