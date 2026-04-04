from __future__ import annotations

import torch

from src.diffusion.schedule import ScheduleState, extract


def q_posterior_mean_var(
    x0: torch.Tensor,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    state: ScheduleState,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar_t = extract(state.alpha_bars, timesteps, xt.shape)
    beta_t = extract(state.betas, timesteps, xt.shape)
    alpha_t = extract(state.alphas, timesteps, xt.shape)

    if state.one_based_indexing:
        t_prev = torch.clamp(timesteps - 1, min=0)
    else:
        t_prev = torch.clamp(timesteps - 1, min=0)

    alpha_bar_prev = extract(state.alpha_bars, t_prev, xt.shape)

    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)

    posterior_mean = coef_x0 * x0 + coef_xt * xt
    posterior_var = extract(state.posterior_variance, timesteps, xt.shape)
    return posterior_mean, posterior_var


def predict_x0_from_eps(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    eps_hat: torch.Tensor,
    state: ScheduleState,
) -> torch.Tensor:
    sqrt_alpha_bar = extract(state.sqrt_alpha_bars, timesteps, xt.shape)
    sqrt_one_minus_alpha_bar = extract(state.sqrt_one_minus_alpha_bars, timesteps, xt.shape)
    return (xt - sqrt_one_minus_alpha_bar * eps_hat) / torch.clamp(sqrt_alpha_bar, min=1e-12)


def p_mean_from_eps(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    eps_hat: torch.Tensor,
    state: ScheduleState,
) -> torch.Tensor:
    beta_t = extract(state.betas, timesteps, xt.shape)
    inv_sqrt_alpha_t = extract(state.inv_sqrt_alphas, timesteps, xt.shape)
    sqrt_one_minus_alpha_bar = extract(state.sqrt_one_minus_alpha_bars, timesteps, xt.shape)

    return inv_sqrt_alpha_t * (xt - (beta_t / torch.clamp(sqrt_one_minus_alpha_bar, min=1e-12)) * eps_hat)


def p_sample_step(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    eps_hat: torch.Tensor,
    state: ScheduleState,
) -> torch.Tensor:
    mean = p_mean_from_eps(xt, timesteps, eps_hat, state)
    var = extract(state.posterior_variance, timesteps, xt.shape)

    if state.one_based_indexing:
        add_noise = (timesteps > 1).float().view(-1, 1, 1, 1)
    else:
        add_noise = (timesteps > 0).float().view(-1, 1, 1, 1)

    noise = torch.randn_like(xt)
    return mean + add_noise * torch.sqrt(torch.clamp(var, min=0.0)) * noise
