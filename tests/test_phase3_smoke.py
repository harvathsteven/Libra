"""Phase 3: smoke pipeline produces results table and figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trait_mtmm.run import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_smoke_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run against real smoke config; write under results/ (gitignored parquet ok)
    main(["smoke", "--config", "configs/smoke.yaml"])
    scores = ROOT / "results" / "pilot_smoke" / "condition_scores.parquet"
    assert scores.exists()
    df = pd.read_parquet(scores)
    assert len(df) >= 1
    for col in ("E_SR", "A_SR", "E_BC", "A_BC", "E_IP", "A_IP"):
        assert col in df.columns

    figs = list((ROOT / "figures").glob("smoke_*.png"))
    assert len(figs) >= 4

    extract = ROOT / "results" / "extract_smoke" / "extraction_summary.json"
    assert extract.exists()
