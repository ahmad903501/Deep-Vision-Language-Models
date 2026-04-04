from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        # Some Kaggle images can report CUDA available even when the wheel
        # does not contain kernels for the attached GPU architecture.
        probe = torch.tensor([1.0], device="cuda")
        _ = probe + 1.0
        return torch.device("cuda")
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
        print(
            "CUDA reported available but failed runtime probe. "
            f"Falling back to CPU. Details: {exc}"
        )
        return torch.device("cpu")
