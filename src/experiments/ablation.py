from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Dict

from src.utils.config import AppConfig


def build_schedule_ablation_configs(config: AppConfig) -> Dict[str, AppConfig]:
    variants: Dict[str, AppConfig] = {}

    linear_schedule = replace(config.schedule, schedule_type="linear")
    variants["linear"] = replace(config, schedule=linear_schedule)

    cosine_schedule = replace(config.schedule, schedule_type="cosine")
    variants["cosine"] = replace(config, schedule=cosine_schedule)

    return variants


def freeze_non_schedule_hparams(config: AppConfig) -> AppConfig:
    # Return a deep copy to make accidental in-place edits obvious during experiments.
    return deepcopy(config)
