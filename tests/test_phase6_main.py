"""Phase 6 main-run wiring (smoke model; numbers meaningless)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from trait_mtmm.run import main

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "main_smoke"


@pytest.mark.slow
def test_main_smoke_runs_resumes_and_reports_no_outcomes() -> None:
    if not (ROOT / "results" / "extract_smoke" / "trait_vectors.pt").exists():
        main(["extract", "--config", "configs/smoke.yaml"])
    if OUT.exists():
        shutil.rmtree(OUT)
    main(["main", "--config", "configs/smoke.yaml"])

    wide = pd.read_parquet(OUT / "condition_scores.parquet")
    assert "baseline_none" in set(wide["persona_id"])
    assert len(wide) == 3  # 2 smoke personas + baseline
    assert len(list((OUT / "ip_residuals").glob("*.pt"))) == 3

    quality = json.loads((OUT / "main_quality.json").read_text())
    assert quality["rows_complete"] == {"sr_item": True, "bc_item": True, "ip_probe": True}
    # No outcome statistics before the Phase 7 analysis code exists.
    text = json.dumps(quality) + (OUT / "provenance.json").read_text()
    for banned in ("mtmm", "pearson", "corr", "std"):
        assert banned not in text.lower()

    # Resume: everything already done, outputs unchanged.
    before = (OUT / "sr_item.parquet").read_bytes()
    main(["main", "--config", "configs/smoke.yaml"])
    assert pd.read_parquet(OUT / "sr_item.parquet").equals(pd.read_parquet(pd.io.common.BytesIO(before)))

    # Resume refuses parts written at another commit.
    done = OUT / "parts" / "baseline_none.done.json"
    done.write_text(json.dumps({"persona_id": "baseline_none", "git_commit": "deadbeef"}))
    with pytest.raises(SystemExit):
        main(["main", "--config", "configs/smoke.yaml"])
