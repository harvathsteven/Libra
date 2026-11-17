"""Unit tests for Phase 4 diagnostics (no model)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trait_mtmm.diagnose import diagnose_pilot


def test_diagnose_passes_on_varied_scores(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    sr_item = pd.DataFrame(
        {
            "persona_id": ["p1", "p1", "p2", "p2"],
            "option_mass": [0.9, 0.85, 0.88, 0.92],
            "score": [2.0, 4.0, 3.0, 3.5],
        }
    )
    bc_item = pd.DataFrame(
        {
            "persona_id": ["p1", "p1", "p2", "p2"],
            "option_mass": [0.9, 0.91, 0.87, 0.86],
            "score": [0.2, 0.8, 0.4, 0.6],
        }
    )
    wide = pd.DataFrame(
        {
            "persona_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
            "E_SR": [2.0, 2.5, 3.0, 3.5, 4.0, 2.2, 3.8, 3.1],
            "A_SR": [4.0, 3.5, 3.0, 2.5, 2.0, 3.8, 2.2, 3.1],
            "E_BC": [0.2, 0.3, 0.4, 0.5, 0.6, 0.25, 0.55, 0.45],
            "A_BC": [0.7, 0.6, 0.5, 0.4, 0.3, 0.65, 0.35, 0.5],
            "E_IP": [0.1, -0.1, 0.2, -0.2, 0.0, 0.15, -0.15, 0.05],
            "A_IP": [-0.1, 0.1, -0.2, 0.2, 0.0, -0.15, 0.15, -0.05],
        }
    )
    sr_item.to_parquet(pilot / "sr_item.parquet", index=False)
    bc_item.to_parquet(pilot / "bc_item.parquet", index=False)
    wide.to_parquet(pilot / "condition_scores.parquet", index=False)
    report = diagnose_pilot(pilot)
    assert report["phase4_acceptance_passed"] is True
