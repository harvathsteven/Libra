"""Load stimuli with optional smoke-subsetting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_adjectives(stimuli: Path) -> dict[str, Any]:
    with (stimuli / "adjectives.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ipip(stimuli: Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
    with (stimuli / "ipip_neo_120_EA.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_items is not None:
        # Keep balance across traits when truncating
        e = [r for r in rows if r["trait"] == "E"][: max_items]
        a = [r for r in rows if r["trait"] == "A"][: max_items]
        rows = e + a
    return rows


def load_bc(stimuli: Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _jsonl(stimuli / "bc_items.jsonl")
    if max_items is not None:
        e = [r for r in rows if r["trait"] == "E"][:max_items]
        a = [r for r in rows if r["trait"] == "A"][:max_items]
        rows = e + a
    return rows


def load_questions(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def load_personas(
    stimuli: Path,
    *,
    which: str = "test",
    max_personas: int | None = None,
    include_baseline: bool = False,
) -> list[dict[str, Any]]:
    name = "personas_dev.jsonl" if which == "dev" else "personas_test.jsonl"
    rows = _jsonl(stimuli / name)
    if which == "test" and not include_baseline:
        rows = [r for r in rows if r.get("set") != "baseline"]
    if max_personas is not None:
        rows = rows[:max_personas]
    return rows
