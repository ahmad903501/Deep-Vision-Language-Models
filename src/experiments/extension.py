from __future__ import annotations

import torch

from src.diffusion.ddpm import ddim_step
from src.diffusion.schedule import ScheduleState


def deterministic_ddim_sample(
    model: torch.nn.Module,
    state: ScheduleState,
    shape: tuple[int, int, int, int],
    device: torch.device,
    eta: float = 0.0,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(shape, device=device)

    if state.one_based_indexing:
        timesteps = range(state.num_timesteps, 0, -1)
    else:
        timesteps = range(state.num_timesteps - 1, -1, -1)

    with torch.no_grad():
        for i in timesteps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps_hat = model(x, t)
            x = ddim_step(x, eps_hat, t, state, eta=eta)

    return x.detach()
