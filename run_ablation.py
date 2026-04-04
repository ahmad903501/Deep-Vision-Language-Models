from __future__ import annotations

import argparse
from dataclasses import replace

from src.experiments.ablation import build_schedule_ablation_configs
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare schedule ablation configurations")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    variants = build_schedule_ablation_configs(base)

    print("Prepared ablation variants:")
    for name, cfg in variants.items():
        print(f"- {name}: schedule_type={cfg.schedule.schedule_type}")


if __name__ == "__main__":
    main()
