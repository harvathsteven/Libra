"""prereg-v2 additions: IP residual cache, extraction activations, vector pin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from trait_mtmm.measures.ip import cosine_scores, project
from trait_mtmm.provenance import load_config
from trait_mtmm.run import _load_L_star_and_vectors, main, sha256_file
from trait_mtmm.vectors import load_vectors, vectors_from_activations

ROOT = Path(__file__).resolve().parents[1]


def test_cosine_scores_matches_project() -> None:
    g = torch.Generator().manual_seed(0)
    res = torch.randn(5, 16, generator=g)
    vec = torch.randn(16, generator=g)
    batched = cosine_scores(res, vec)
    single = torch.tensor([project(r, vec) for r in res])
    assert torch.allclose(batched, single, atol=1e-6)


@pytest.mark.slow
def test_smoke_writes_caches_consistent_with_scores() -> None:
    main(["smoke", "--config", "configs/smoke.yaml"])
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    extract = ROOT / "results" / "extract_smoke"
    pilot = ROOT / "results" / "pilot_smoke"

    # Extraction activations reproduce the saved per-layer vectors.
    acts = torch.load(extract / "extraction_activations.pt", weights_only=False)
    blob = load_vectors(extract / "trait_vectors.pt")
    recomputed = vectors_from_activations(acts)
    for t in ("E", "A"):
        for i, v in enumerate(recomputed[t]):
            # float16 on disk -> allow small rounding
            assert float(torch.dot(v, blob["vectors_by_layer"][t][i].float())) > 0.999

    # IP residual cache at L* reproduces the per-probe IP scores.
    L_star, vectors, _ = _load_L_star_and_vectors(cfg)
    ip_probe = pd.read_parquet(pilot / "ip_probe.parquet")
    files = sorted((pilot / "ip_residuals").glob("*.pt"))
    assert files
    for f in files:
        rec = torch.load(f, weights_only=False)
        res = rec["residuals"]  # [n_probes, n_layers, d]
        assert res.dtype == torch.float16
        for t, vec in vectors.items():
            want = (
                ip_probe[(ip_probe.persona_id == rec["persona_id"]) & (ip_probe.trait == t)]
                .sort_values("probe_index")["projection"]
                .to_numpy()
            )
            got = cosine_scores(res[:, L_star], vec).numpy()
            assert got.shape == want.shape
            assert abs(got - want).max() < 5e-3


@pytest.mark.slow
def test_vector_pin_mismatch_refuses() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    pt = ROOT / "results" / "extract_smoke" / "trait_vectors.pt"
    if not pt.exists():
        pytest.skip("run smoke first")
    cfg["vectors"] = {"sha256": sha256_file(pt)}
    _load_L_star_and_vectors(cfg, require_pin=True)  # matches: ok
    cfg["vectors"] = {"sha256": "0" * 64}
    with pytest.raises(SystemExit):
        _load_L_star_and_vectors(cfg)
    cfg["vectors"] = {"sha256": None}
    with pytest.raises(SystemExit):
        _load_L_star_and_vectors(cfg, require_pin=True)
