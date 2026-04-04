from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_root: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    data_root: str
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    scale_to_minus_one_to_one: bool


@dataclass(frozen=True)
class ScheduleConfig:
    num_timesteps: int
    schedule_type: str
    beta_min: float
    beta_max: float
    indexing: str


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    base_channels: int
    channel_multipliers: List[int]
    num_res_blocks: int
    time_embed_dim: int
    use_attention: bool
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    num_epochs: int
    max_steps: int
    grad_clip_norm: float
    log_every: int
    sample_every: int
    save_every: int
    amp: bool
    overfit_subset_size: int


@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int
    save_trajectory_steps: List[int]


@dataclass(frozen=True)
class AblationConfig:
    enabled: bool
    compare_schedule: str


@dataclass(frozen=True)
class ExtensionConfig:
    enabled: bool
    type: str


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    data: DataConfig
    schedule: ScheduleConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    ablation: AblationConfig
    extension: ExtensionConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_config_path(config_path: str | Path) -> Path:
    raw_path = Path(config_path)
    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]

    candidates = [
        raw_path,
        cwd / raw_path,
        cwd / "configs" / raw_path.name,
        repo_root / raw_path,
        repo_root / "configs" / raw_path.name,
    ]

    checked = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        checked.append(str(resolved))
        if candidate.exists() and candidate.is_file():
            return candidate

    checked_str = "\n".join(f"- {item}" for item in checked)
    raise FileNotFoundError(
        "Config file not found. Checked:\n"
        f"{checked_str}\n"
        "Tip: pass --config configs/kaggle.yaml (or default.yaml)."
    )


def load_config(config_path: str | Path) -> AppConfig:
    path = _resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or invalid YAML: {path}")

    return AppConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        data=DataConfig(**raw["data"]),
        schedule=ScheduleConfig(**raw["schedule"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        sampling=SamplingConfig(**raw["sampling"]),
        ablation=AblationConfig(**raw["ablation"]),
        extension=ExtensionConfig(**raw["extension"]),
    )
