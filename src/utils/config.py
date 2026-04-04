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


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

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
