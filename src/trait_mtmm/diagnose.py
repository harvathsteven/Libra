"""Phase 4 diagnostics: option mass, floor/ceiling, score variance on dev pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def diagnose_pilot(
    pilot_dir: Path,
    *,
    min_option_mass: float = 0.80,
    endpoint_frac_max: float = 0.5,
    sr_lo: float = 1.05,
    sr_hi: float = 4.95,
    bc_lo: float = 0.05,
    bc_hi: float = 0.95,
) -> dict[str, Any]:
    """Check Phase 4 acceptance criteria on saved pilot parquet files."""
    sr_item = pd.read_parquet(pilot_dir / "sr_item.parquet")
    bc_item = pd.read_parquet(pilot_dir / "bc_item.parquet")
    wide = pd.read_parquet(pilot_dir / "condition_scores.parquet")

    sr_mass = float(sr_item["option_mass"].mean())
    bc_mass = float(bc_item["option_mass"].mean())
    mass_ok = sr_mass >= min_option_mass and bc_mass >= min_option_mass

    n_personas = int(wide["persona_id"].nunique())
    pinned: dict[str, Any] = {}
    pin_ok = True
    for col, lo, hi in (
        ("E_SR", sr_lo, sr_hi),
        ("A_SR", sr_lo, sr_hi),
        ("E_BC", bc_lo, bc_hi),
        ("A_BC", bc_lo, bc_hi),
    ):
        if col not in wide.columns or n_personas == 0:
            continue
        at_end = ((wide[col] <= lo) | (wide[col] >= hi)).sum()
        frac = float(at_end) / n_personas
        pinned[col] = {"n_pinned": int(at_end), "frac": frac, "passed": frac <= endpoint_frac_max}
        pin_ok = pin_ok and frac <= endpoint_frac_max

    variance: dict[str, Any] = {}
    vary_ok = True
    for col in ("E_SR", "A_SR", "E_BC", "A_BC", "E_IP", "A_IP"):
        if col not in wide.columns:
            continue
        std = float(wide[col].std(ddof=0))
        variance[col] = {"std": std, "passed": std > 1e-6}
        vary_ok = vary_ok and std > 1e-6

    report = {
        "n_personas": n_personas,
        "option_mass": {
            "sr_mean": sr_mass,
            "bc_mean": bc_mass,
            "threshold": min_option_mass,
            "passed": mass_ok,
        },
        "floor_ceiling": pinned,
        "floor_ceiling_passed": pin_ok,
        "variance": variance,
        "variance_passed": vary_ok,
        "phase4_acceptance_passed": bool(mass_ok and pin_ok and vary_ok),
    }
    out = pilot_dir / "phase4_diagnostics.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
