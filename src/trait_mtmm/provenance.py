"""Config loading, paths, and results provenance metadata."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def stimuli_dir(cfg: dict[str, Any]) -> Path:
    return repo_root() / cfg["paths"]["stimuli_dir"]


def results_dir(cfg: dict[str, Any]) -> Path:
    d = repo_root() / cfg["paths"]["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(cfg: dict[str, Any]) -> Path:
    d = repo_root() / cfg["paths"]["figures_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def provenance_dict(cfg: dict[str, Any], *, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "seed": cfg["seed"],
        "config_path": cfg.get("_config_path"),
        "L_star": cfg.get("layer", {}).get("L_star"),
        "libraries": library_versions(),
        "is_smoke": "smoke" in cfg,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
