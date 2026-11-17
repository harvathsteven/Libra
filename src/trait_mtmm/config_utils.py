"""Helpers to freeze L* into configs/default.yaml after Phase 4 extraction."""

from __future__ import annotations

from pathlib import Path

import yaml


def set_L_star(config_path: Path, L_star: int) -> None:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("layer", {})
    cfg["layer"]["L_star"] = int(L_star)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
