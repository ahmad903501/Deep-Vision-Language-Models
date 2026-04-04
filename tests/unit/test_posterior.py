import torch

from src.diffusion.posterior import p_mean_from_eps, p_sample_step, q_posterior_mean_var
from src.diffusion.schedule import build_schedule_state


def test_posterior_variance_validity() -> None:
    state = build_schedule_state(
        num_timesteps=1000,
        schedule_type="linear",
        beta_min=1e-4,
        beta_max=0.02,
        indexing="one_based",
        device=torch.device("cpu"),
    )

    x0 = torch.randn(16, 1, 28, 28)
    xt = torch.randn_like(x0)
    t = torch.randint(2, 1001, (16,), dtype=torch.long)

    _, var = q_posterior_mean_var(x0, xt, t, state)
    beta_t = state.betas.gather(0, t).view(-1, 1, 1, 1)

    assert torch.all(var >= 0.0)
    assert torch.all(var <= beta_t)


def test_no_noise_at_final_step() -> None:
    state = build_schedule_state(
        num_timesteps=1000,
        schedule_type="linear",
        beta_min=1e-4,
        beta_max=0.02,
        indexing="one_based",
        device=torch.device("cpu"),
    )

    xt = torch.randn(8, 1, 28, 28)
    eps_hat = torch.randn_like(xt)
    t = torch.ones(8, dtype=torch.long)

    mean = p_mean_from_eps(xt, t, eps_hat, state)
    sample = p_sample_step(xt, t, eps_hat, state)

    assert torch.allclose(mean, sample)
