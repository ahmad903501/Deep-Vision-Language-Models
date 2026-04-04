from __future__ import annotations

import torch

from src.diffusion.schedule import ScheduleState, extract


def sample_timesteps(batch_size: int, state: ScheduleState, device: torch.device) -> torch.Tensor:
    if state.one_based_indexing:
        return torch.randint(1, state.num_timesteps + 1, (batch_size,), device=device)
    return torch.randint(0, state.num_timesteps, (batch_size,), device=device)


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    state: ScheduleState,
    eps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps is None:
        eps = torch.randn_like(x0)
    sqrt_alpha_bar = extract(state.sqrt_alpha_bars, timesteps, x0.shape)
    sqrt_one_minus_alpha_bar = extract(state.sqrt_one_minus_alpha_bars, timesteps, x0.shape)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps
    return xt, eps


def empirical_forward_stats(
    x0: torch.Tensor,
    timestep: int,
    state: ScheduleState,
    num_samples: int = 128,
) -> tuple[float, float, float, float]:
    repeated = x0.repeat(num_samples, 1, 1, 1)
    t = torch.full((num_samples,), timestep, device=x0.device, dtype=torch.long)
    xt, _ = q_sample(repeated, t, state)

    observed_mean = float(xt.mean().item())
    observed_var = float(xt.var(unbiased=False).item())

    target_mean = float((state.sqrt_alpha_bars[timestep] * x0).mean().item())
    target_var = float((1.0 - state.alpha_bars[timestep]).item())
    return observed_mean, observed_var, target_mean, target_var
