from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
