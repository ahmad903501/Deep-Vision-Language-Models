from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScheduleState:
    num_timesteps: int
    one_based_indexing: bool
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    inv_sqrt_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def make_beta_schedule(
    num_timesteps: int,
    schedule_type: str,
    beta_min: float,
    beta_max: float,
) -> torch.Tensor:
    if schedule_type == "linear":
        return torch.linspace(beta_min, beta_max, num_timesteps, dtype=torch.float32)
    if schedule_type == "cosine":
        # Cosine schedule from Nichol and Dhariwal, clipped for numerical stability.
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        t = steps / num_timesteps
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(min=1e-6, max=0.999)
    raise ValueError(f"Unsupported schedule_type={schedule_type}")


def build_schedule_state(
    num_timesteps: int,
    schedule_type: str,
    beta_min: float,
    beta_max: float,
    indexing: str,
    device: torch.device,
) -> ScheduleState:
    one_based = indexing == "one_based"
    if indexing not in {"one_based", "zero_based"}:
        raise ValueError("indexing must be one_of: {'one_based', 'zero_based'}")

    betas_raw = make_beta_schedule(num_timesteps, schedule_type, beta_min, beta_max).to(device)
    alphas_raw = 1.0 - betas_raw
    alpha_bars_raw = torch.cumprod(alphas_raw, dim=0)

    if one_based:
        betas = torch.zeros(num_timesteps + 1, device=device)
        betas[1:] = betas_raw

        alphas = torch.ones(num_timesteps + 1, device=device)
        alphas[1:] = alphas_raw

        alpha_bars = torch.ones(num_timesteps + 1, device=device)
        alpha_bars[1:] = alpha_bars_raw
    else:
        betas = betas_raw
        alphas = alphas_raw
        alpha_bars = alpha_bars_raw

    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(torch.clamp(1.0 - alpha_bars, min=1e-12))
    inv_sqrt_alphas = 1.0 / torch.sqrt(torch.clamp(alphas, min=1e-12))

    if one_based:
        alpha_bar_prev = torch.ones_like(alpha_bars)
        alpha_bar_prev[1:] = alpha_bars[:-1]
    else:
        alpha_bar_prev = torch.ones_like(alpha_bars)
        alpha_bar_prev[1:] = alpha_bars[:-1]

    posterior_variance = torch.clamp(
        ((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bars, min=1e-12)) * betas,
        min=0.0,
    )

    return ScheduleState(
        num_timesteps=num_timesteps,
        one_based_indexing=one_based,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        inv_sqrt_alphas=inv_sqrt_alphas,
        posterior_variance=posterior_variance,
    )


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    if timesteps.dtype != torch.long:
        timesteps = timesteps.long()
    out = values.gather(dim=0, index=timesteps)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def compute_snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    return alpha_bars / torch.clamp(1.0 - alpha_bars, min=1e-12)
