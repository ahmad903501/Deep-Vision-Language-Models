import torch

from src.diffusion.forward import q_sample
from src.diffusion.schedule import build_schedule_state, compute_snr


def test_alpha_bar_and_snr_monotonicity() -> None:
    state = build_schedule_state(
        num_timesteps=1000,
        schedule_type="linear",
        beta_min=1e-4,
        beta_max=0.02,
        indexing="one_based",
        device=torch.device("cpu"),
    )

    alpha_bar = state.alpha_bars[1:]
    snr = compute_snr(alpha_bar)

    assert torch.all(alpha_bar[1:] <= alpha_bar[:-1])
    assert torch.all(snr[1:] <= snr[:-1])


def test_q_sample_statistics_are_reasonable() -> None:
    state = build_schedule_state(
        num_timesteps=1000,
        schedule_type="linear",
        beta_min=1e-4,
        beta_max=0.02,
        indexing="one_based",
        device=torch.device("cpu"),
    )

    x0 = torch.randn(1024, 1, 28, 28)
    t = torch.full((x0.shape[0],), 500, dtype=torch.long)
    xt, _ = q_sample(x0, t, state)

    target_var = float(1.0 - state.alpha_bars[500].item())
    observed_var = float((xt - torch.sqrt(state.alpha_bars[500]) * x0).var(unbiased=False).item())
    assert abs(observed_var - target_var) < 0.05
