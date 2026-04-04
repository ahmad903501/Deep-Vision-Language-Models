from __future__ import annotations

from typing import Dict, Iterable

import torch

from src.diffusion.posterior import p_sample_step
from src.diffusion.schedule import ScheduleState


def ancestral_sample(
    model: torch.nn.Module,
    state: ScheduleState,
    shape: tuple[int, int, int, int],
    device: torch.device,
    save_steps: Iterable[int],
) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    model.eval()
    x = torch.randn(shape, device=device)
    saved: Dict[int, torch.Tensor] = {}
    save_steps_set = set(save_steps)

    if state.one_based_indexing:
        timesteps = range(state.num_timesteps, 0, -1)
    else:
        timesteps = range(state.num_timesteps - 1, -1, -1)

    with torch.no_grad():
        for i in timesteps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps_hat = model(x, t)
            x = p_sample_step(x, t, eps_hat, state)
            if i in save_steps_set:
                saved[i] = x.detach().cpu()

    return x.detach(), saved


def ddim_step(
    xt: torch.Tensor,
    eps_hat: torch.Tensor,
    timestep: torch.Tensor,
    state: ScheduleState,
    eta: float = 0.0,
) -> torch.Tensor:
    from src.diffusion.posterior import predict_x0_from_eps
    from src.diffusion.schedule import extract

    x0_hat = predict_x0_from_eps(xt, timestep, eps_hat, state)

    alpha_bar_t = extract(state.alpha_bars, timestep, xt.shape)
    t_prev = torch.clamp(timestep - 1, min=0 if not state.one_based_indexing else 0)
    alpha_bar_prev = extract(state.alpha_bars, t_prev, xt.shape)

    sigma = (
        eta
        * torch.sqrt(torch.clamp((1 - alpha_bar_prev) / torch.clamp(1 - alpha_bar_t, min=1e-12), min=0.0))
        * torch.sqrt(torch.clamp(1 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-12), min=0.0))
    )

    pred_dir = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * eps_hat
    noise = sigma * torch.randn_like(xt)
    return torch.sqrt(alpha_bar_prev) * x0_hat + pred_dir + noise
