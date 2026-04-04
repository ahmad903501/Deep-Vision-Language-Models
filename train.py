from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.data.pipeline import build_dataloaders
from src.diffusion.forward import empirical_forward_stats
from src.diffusion.schedule import build_schedule_state, compute_snr
from src.eval.visualization import plot_schedule
from src.models.unet import EpsilonUNet
from src.train.trainer import DDPMTrainer
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import ensure_dir, save_json
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM baseline for DVLM PA1")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    device = resolve_device()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(config.experiment.output_root) / config.experiment.name / run_stamp)

    data_bundle = build_dataloaders(
        config=config.data,
        output_root=str(run_dir),
        seed=config.experiment.seed,
    )

    schedule_state = build_schedule_state(
        num_timesteps=config.schedule.num_timesteps,
        schedule_type=config.schedule.schedule_type,
        beta_min=config.schedule.beta_min,
        beta_max=config.schedule.beta_max,
        indexing=config.schedule.indexing,
        device=device,
    )

    alpha_bars = schedule_state.alpha_bars[1:] if schedule_state.one_based_indexing else schedule_state.alpha_bars
    snr = compute_snr(alpha_bars)
    plot_schedule(alpha_bars, snr, run_dir / "plots" / "alpha_bar_snr.png")

    first_batch, _ = next(iter(data_bundle.train_loader))
    x0 = first_batch[:1].to(device)
    probe_t = config.schedule.num_timesteps // 2
    obs_mean, obs_var, tgt_mean, tgt_var = empirical_forward_stats(
        x0=x0,
        timestep=probe_t if schedule_state.one_based_indexing else probe_t - 1,
        state=schedule_state,
    )

    model = EpsilonUNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        channel_multipliers=config.model.channel_multipliers,
        num_res_blocks=config.model.num_res_blocks,
        time_embed_dim=config.model.time_embed_dim,
        dropout=config.model.dropout,
    ).to(device)

    trainer = DDPMTrainer(
        model=model,
        state=schedule_state,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        config=config,
        device=device,
        run_dir=Path(run_dir),
        image_shape=data_bundle.shape,
    )
    artifacts = trainer.train()

    save_json(
        {
            "device": str(device),
            "real_grid_path": str(data_bundle.real_grid_path),
            "value_range": list(data_bundle.value_range),
            "forward_empirical_check": {
                "observed_mean": obs_mean,
                "observed_var": obs_var,
                "target_mean": tgt_mean,
                "target_var": tgt_var,
            },
            "final_checkpoint": str(artifacts.final_checkpoint),
            "overfit_final_loss": artifacts.overfit_final_loss,
            "posterior_check": artifacts.posterior_check,
        },
        Path(run_dir) / "reports" / "run_summary.json",
    )

    print(f"Training complete. Artifacts at: {run_dir}")


if __name__ == "__main__":
    main()
