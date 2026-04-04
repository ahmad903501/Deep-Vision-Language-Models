from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.pipeline import build_dataloaders
from src.diffusion.ddpm import ancestral_sample
from src.diffusion.schedule import build_schedule_state
from src.eval.metrics import (
    MetricReport,
    SmallFeatureExtractor,
    compute_fid,
    compute_kid,
    estimate_bpd,
    nearest_neighbor_l2,
    train_feature_extractor,
)
from src.eval.visualization import save_tensor_grid
from src.models.unet import EpsilonUNet
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import ensure_dir, save_json
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDPM run")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    return parser.parse_args()


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _generate_samples(
    model: torch.nn.Module,
    state,
    num_samples: int,
    batch_size: int,
    image_shape,
    device: torch.device,
) -> torch.Tensor:
    samples = []
    produced = 0
    while produced < num_samples:
        current = min(batch_size, num_samples - produced)
        batch, _ = ancestral_sample(
            model=model,
            state=state,
            shape=(current, *image_shape),
            device=device,
            save_steps=[],
        )
        samples.append(batch.cpu())
        produced += current
    return torch.cat(samples, dim=0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    device = resolve_device()

    out_dir = ensure_dir(Path(config.experiment.output_root) / config.experiment.name / "evaluation")

    data_bundle = build_dataloaders(
        config=config.data,
        output_root=str(out_dir),
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

    model = EpsilonUNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        channel_multipliers=config.model.channel_multipliers,
        num_res_blocks=config.model.num_res_blocks,
        time_embed_dim=config.model.time_embed_dim,
        dropout=config.model.dropout,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(_normalize_state_dict(checkpoint["model"]))
    model.eval()

    generated = _generate_samples(
        model=model,
        state=schedule_state,
        num_samples=args.num_samples,
        batch_size=config.data.batch_size,
        image_shape=data_bundle.shape,
        device=device,
    )

    save_tensor_grid(generated[:64], out_dir / "generated_samples_8x8.png")

    feature_extractor = SmallFeatureExtractor(in_channels=config.model.in_channels)
    feature_extractor = train_feature_extractor(feature_extractor, data_bundle.train_loader, device=device)

    real_feats = []
    generated_loader = DataLoader(TensorDataset(generated, torch.zeros(len(generated), dtype=torch.long)), batch_size=config.data.batch_size)

    from src.eval.metrics import _collect_features  # local import to keep public API small

    real_feats_np = _collect_features(feature_extractor, data_bundle.test_loader, device, max_samples=args.num_samples)
    gen_feats_np = _collect_features(feature_extractor, generated_loader, device, max_samples=args.num_samples)

    fid = compute_fid(real_feats_np, gen_feats_np)
    kid = compute_kid(real_feats_np, gen_feats_np)
    bpd = estimate_bpd(model, data_bundle.test_loader, schedule_state, device=device)

    train_batch, _ = next(iter(data_bundle.train_loader))
    nn_l2 = nearest_neighbor_l2(generated[: train_batch.size(0)], train_batch)

    report = MetricReport(dataset_fid=fid, dataset_kid=kid, bpd=bpd, nearest_neighbor_l2=nn_l2)
    save_json(report.__dict__, out_dir / "metrics_report.json")

    print(f"Evaluation complete. Metrics saved to: {out_dir / 'metrics_report.json'}")


if __name__ == "__main__":
    main()
