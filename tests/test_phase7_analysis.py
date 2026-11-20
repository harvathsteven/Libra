"""Phase 7 analysis: unit checks and synthetic end-to-end runs (no test-set data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import rankdata, spearmanr

from trait_mtmm.measures.ip import cosine_scores
from trait_mtmm.mtmm import (
    SAMPLES,
    TRAITS,
    VARS,
    boot_p,
    cluster_labels,
    cluster_weights,
    condition_table,
    condition_weights,
    cronbach_alpha,
    holm,
    mean_unit_residuals,
    run_analysis,
    shuffled_null_vectors,
    wcorr,
    wranks,
)
from trait_mtmm.stimuli_io import load_personas

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = load_personas(ROOT / "stimuli", which="test", include_baseline=True)


# --------------------------------------------------------------------------- unit


def _expand(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.repeat(x, w.astype(int))


def test_wcorr_equals_pearson_on_expanded_resample() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=20), rng.normal(size=20)
    W = condition_weights(20, 50, rng)
    r = wcorr(W, x, y)
    for b in range(50):
        want = np.corrcoef(_expand(W[b], x), _expand(W[b], y))[0, 1]
        assert abs(r[b] - want) < 1e-12


def test_wranks_spearman_equals_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 6, size=25).astype(float)  # heavy ties
    y = rng.normal(size=25)
    W = condition_weights(25, 40, rng)
    rx, ry = wranks(W, x), wranks(W, y)
    r = wcorr(W, rx, ry)
    for b in range(40):
        want = spearmanr(_expand(W[b], x), _expand(W[b], y)).statistic
        assert abs(r[b] - want) < 1e-12
    # Unweighted midranks equal scipy's average ranks.
    assert np.allclose(wranks(np.ones((1, 25)), x)[0], rankdata(x))


def test_constant_resample_is_undefined() -> None:
    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    r = wcorr(W, x, x)
    assert np.isnan(r[0]) and r[1] == pytest.approx(1.0)


def test_singleton_clusters_equal_condition_bootstrap() -> None:
    labels = np.array([f"p{i:02d}" for i in range(32)])
    a = condition_weights(32, 100, np.random.default_rng(42))
    b = cluster_weights(labels, 100, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_cluster_weights_shared_within_cluster() -> None:
    labels = np.array(["c1", "c1", "c2", "c3", "c3", "c3"])
    W = cluster_weights(labels, 200, np.random.default_rng(3))
    assert np.array_equal(W[:, 0], W[:, 1])
    assert np.array_equal(W[:, 3], W[:, 5])
    # Total clusters drawn per resample is fixed at 3.
    uniq_counts = W[:, [0, 2, 3]]
    assert np.all(uniq_counts.sum(axis=1) == 3)


def test_boot_p_and_holm() -> None:
    d = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, np.nan])
    # defined 5; #<=0 = 2, #>=0 = 4 -> 2*2/5
    assert boot_p(d) == pytest.approx(0.8)
    h = holm({"E": 0.01, "A": 0.04}, 0.05)
    assert h["E"] == {"p_holm": pytest.approx(0.02), "reject": True}
    assert h["A"]["p_holm"] == pytest.approx(0.04) and h["A"]["reject"]
    h = holm({"E": 0.03, "A": 0.04}, 0.05)
    assert not h["E"]["reject"] and not h["A"]["reject"]  # 0.03 > 0.025 stops the step-down


def test_cronbach_alpha_matches_formula() -> None:
    rng = np.random.default_rng(4)
    t = rng.normal(size=(50, 1))
    M = t + rng.normal(scale=0.7, size=(50, 6))
    k = 6
    want = k / (k - 1) * (1 - M.var(axis=0, ddof=1).sum() / M.sum(axis=1).var(ddof=1))
    a, used = cronbach_alpha(M)
    assert used == 6 and a == pytest.approx(want)
    M[3, 2] = np.nan
    a2, used2 = cronbach_alpha(M)
    assert used2 == 5


def _items(pid: str, method: str, trait: str, masses: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "persona_id": pid,
            "persona_set": "explicit",
            "trait": trait,
            "item_id": [f"{method}{trait}{i}" for i in range(len(masses))],
            "score": np.arange(len(masses), dtype=float),
            "score_before_keying": np.arange(len(masses), dtype=float),
            "option_mass": masses,
        }
    )


def test_exclusion_thresholds() -> None:
    sr = pd.concat(
        [
            _items("p_ok", "SR", "E", [0.4] * 4 + [0.5] * 20),  # 4/24 dropped -> kept; 0.50 kept
            _items("p_bad", "SR", "E", [0.4] * 5 + [0.9] * 19),  # 5/24 -> excluded
        ]
    )
    bc = pd.concat(
        [
            _items("p_ok", "BC", "E", [0.1] * 20 + [0.9] * 80),  # 20% -> kept
            _items("p_bad", "BC", "E", [0.1] * 21 + [0.9] * 79),  # 21% -> excluded
        ]
    )
    ip = pd.DataFrame(
        {
            "persona_id": ["p_ok", "p_bad"],
            "persona_set": "explicit",
            "trait": "E",
            "probe_index": 0,
            "projection": [0.1, np.inf],
        }
    )
    wide, log = condition_table(
        sr, bc, ip, n_items={"SR": {"E": 24, "A": 24}, "BC": {"E": 100, "A": 100}}, drop_mass=0.5, drop_frac=0.2
    )
    w = wide.set_index("persona_id")
    assert w.loc["p_ok", "E_SR"] == pytest.approx(np.arange(4, 24).mean())
    assert np.isnan(w.loc["p_bad", "E_SR"]) and np.isnan(w.loc["p_bad", "ACQ_E"])
    assert w.loc["p_ok", "E_BC"] == pytest.approx(np.arange(20, 100).mean())
    assert np.isnan(w.loc["p_bad", "E_BC"])
    assert np.isnan(w.loc["p_bad", "E_IP"])  # non-finite IP is missing
    # A missing SR row counts as dropped.
    wide2, _ = condition_table(
        _items("p", "SR", "E", [0.9] * 19), bc.iloc[:0], ip.iloc[:0],
        n_items={"SR": {"E": 24, "A": 24}, "BC": {"E": 100, "A": 100}}, drop_mass=0.5, drop_frac=0.2,
    )
    assert np.isnan(wide2.loc[0, "E_SR"])


def test_implicit_cue_clusters() -> None:
    labels = cluster_labels(PERSONAS)
    imp = {p["persona_id"]: p for p in PERSONAS if p["set"] == "implicit"}
    cues = pd.Series({pid: labels[pid] for pid in imp})
    assert cues.nunique() == 12
    assert sorted(cues.value_counts().unique()) == [5, 6]
    for _, pids in cues.groupby(cues):
        assert len({imp[p]["quadrant"] for p in pids.index}) == 1
        assert len({int(p.rsplit("_", 1)[1]) % 3 for p in pids.index}) == 1
    exp = [p["persona_id"] for p in PERSONAS if p["set"] == "explicit"]
    assert len({labels[p] for p in exp}) == 32


def test_shuffled_null_vectors_keep_counts_and_are_seeded() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(128, 10))
    labels = np.r_[np.ones(64, bool), np.zeros(64, bool)]
    a = shuffled_null_vectors(X, labels, 30, seed=43)
    b = shuffled_null_vectors(X, labels, 30, seed=43)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    # first draw equals the explicit procedure
    lab = np.random.default_rng(43).permutation(labels)
    v = X[lab].mean(0) - X[~lab].mean(0)
    assert np.allclose(a[0], v / np.linalg.norm(v))


def test_mean_unit_residuals_matches_cosine_scores() -> None:
    g = torch.Generator().manual_seed(0)
    res = torch.randn(7, 3, 16, generator=g)
    vec = torch.randn(16, generator=g)
    mbar = mean_unit_residuals(res)
    want = float(cosine_scores(res[:, 1].double(), vec.double()).mean())
    got = float(mbar[1] @ (vec.double() / vec.double().norm()).numpy())
    assert got == pytest.approx(want, abs=1e-6)  # cosine_scores works in float32


# ---------------------------------------------------------------------- synthetic


N_LAYERS, D, N_PROBES, L_STAR = 4, 24, 6, 2
N_ITEMS = {"SR": {"E": 6, "A": 6}, "BC": {"E": 10, "A": 10}}


def _make_synthetic(tmp: Path, *, signal: bool, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(D, 3)))
    v = {"E": q[:, 0], "A": q[:, 1]}
    base = q[:, 2] * 6.0

    latent = {}
    for p in PERSONAS:
        z = {}
        for t in TRAITS:
            num = p.get(f"intended_{t}_numeric")
            if p["set"] == "explicit":
                mu = (num - 2.5) / 1.1
            elif p["set"] == "implicit":
                mu = 1.0 if num >= 3 else -1.0
            else:
                mu = 0.0
            z[t] = mu + rng.normal(scale=0.4) if signal else rng.normal()
        latent[p["persona_id"]] = z

    def noisy(pid: str, t: str) -> float:
        # without signal each measure gets its own independent latent
        return latent[pid][t] if signal else rng.normal()

    sr, bc, ip, stored = [], [], [], []
    res_dir = tmp / "main" / "ip_residuals"
    res_dir.mkdir(parents=True)
    for p in PERSONAS:
        pid, pset = p["persona_id"], p["set"]
        zs = {t: noisy(pid, t) for t in TRAITS}
        zb = {t: noisy(pid, t) for t in TRAITS}
        zi = {t: noisy(pid, t) for t in TRAITS}
        for t in TRAITS:
            for i in range(N_ITEMS["SR"][t]):
                s = 3 + 0.6 * zs[t] + rng.normal(scale=0.5)
                sr.append({"persona_id": pid, "persona_set": pset, "trait": t, "item_id": f"S{t}{i}",
                           "score": s, "score_before_keying": s, "option_mass": 0.99})
            for i in range(N_ITEMS["BC"][t]):
                b = 1 / (1 + np.exp(-(zb[t] + rng.normal(scale=0.7))))
                bc.append({"persona_id": pid, "persona_set": pset, "trait": t, "item_id": f"B{t}{i}",
                           "score": b, "option_mass": 0.99})
        h = np.stack(
            [
                np.stack([base + 1.5 * zi["E"] * v["E"] + 1.5 * zi["A"] * v["A"] + rng.normal(scale=1.0, size=D)
                          for _ in range(N_LAYERS)])
                for _ in range(N_PROBES)
            ]
        )  # (probes, layers, d)
        h16 = torch.tensor(h, dtype=torch.float16)
        torch.save({"persona_id": pid, "residuals": h16}, res_dir / f"{pid}.pt")
        for t in TRAITS:
            cos = cosine_scores(h16[:, L_STAR].double(), torch.tensor(v[t]))
            for k in range(N_PROBES):
                ip.append({"persona_id": pid, "persona_set": pset, "trait": t, "probe_index": k,
                           "probe": f"q{k}", "projection": float(cos[k])})
    main = tmp / "main"
    sr_df, bc_df, ip_df = pd.DataFrame(sr), pd.DataFrame(bc), pd.DataFrame(ip)
    sr_df.to_parquet(main / "sr_item.parquet")
    bc_df.to_parquet(main / "bc_item.parquet")
    ip_df.to_parquet(main / "ip_probe.parquet")
    wide = {}
    for m, df, col in (("SR", sr_df, "score"), ("BC", bc_df, "score"), ("IP", ip_df, "projection")):
        piv = df.groupby(["persona_id", "trait"])[col].mean().unstack()
        for t in TRAITS:
            wide[f"{t}_{m}"] = piv[t]
    stored = pd.DataFrame(wide).reset_index()
    stored.to_parquet(main / "condition_scores.parquet")

    blob = {
        "L_star": L_STAR,
        "vectors": {t: torch.tensor(v[t], dtype=torch.float32) for t in TRAITS},
        "vectors_by_layer": {t: [torch.tensor(v[t], dtype=torch.float32)] * N_LAYERS for t in TRAITS},
    }
    acts = {}
    for t in TRAITS:
        acts[t] = {}
        for pole, sign in (("high", 1.0), ("low", -1.0)):
            a = sign * 2.0 * v[t][None, None, :] + rng.normal(size=(120, N_LAYERS, D))
            is_train = [True] * 64 + [False] * 56
            acts[t][pole] = {"acts": torch.tensor(a, dtype=torch.float16), "is_train": is_train}
    return {"main_dir": main, "vectors_blob": blob, "extraction_acts": acts}


def _run(tmp: Path, *, signal: bool) -> dict:
    data = _make_synthetic(tmp, signal=signal)
    return run_analysis(
        **data,
        personas=PERSONAS,
        n_items=N_ITEMS,
        L_star=L_STAR,
        B=300,
        seed=42,
        n_null=200,
        alpha=0.05,
        drop_mass=0.5,
        drop_frac=0.2,
    )


def _verdict(res: dict, sample: str, hyp: str, t: str, method: str = "pearson") -> dict:
    return next(v for v in res["verdicts"] if v["sample"] == sample and v["hypothesis"] == hyp
                and v["trait"] == t and v["corr_method"] == method)


def test_synthetic_with_signal_supports_h1_h2_h4(tmp_path: Path) -> None:
    res = _run(tmp_path, signal=True)
    info = res["sample_info"]
    assert info["pooled"]["n_listwise"] == 96 and info["implicit"]["n_clusters"] == 12
    assert info["explicit"]["n_clusters"] == 32
    assert res["checks"]["ip_cache_max_abs_diff"] < 1e-6
    for t in TRAITS:
        for hyp in ("H1", "H2", "H4"):
            assert _verdict(res, "pooled", hyp, t)["holds"], (hyp, t)
        assert "All three converge and discriminate" in res["outcome_patterns"][t]
    # Every hypothesis × trait × sample × method has a verdict.
    keys = {(v["sample"], v["corr_method"], v["hypothesis"], v["trait"]) for v in res["verdicts"]}
    for s in SAMPLES:
        for m in ("pearson", "spearman"):
            for h in ("H1", "H2", "H3", "H4"):
                for t in TRAITS:
                    assert (s, m, h, t) in keys
    assert ("explicit_vs_implicit", "pearson", "H5", "E") in keys
    # Cluster intervals are not narrower than condition intervals on the implicit H1 rows (typical).
    rows = pd.DataFrame(res["rows"])
    assert set(rows["bootstrap"]) == {"condition", "cluster", "none"}
    # Explicit: the two bootstraps are identical.
    ex = rows[(rows["sample"] == "explicit") & rows["bootstrap"].isin(["condition", "cluster"])]
    piv = ex.pivot_table(index=["corr_method", "component"], columns="bootstrap", values="ci_low")
    assert np.allclose(piv["condition"], piv["cluster"])
    # Reliabilities are high with this signal.
    assert res["reliability"]["pooled"]["E_SR"]["value"] > 0.7
    assert len(res["layer_sweep"]) == 3 * 2 * N_LAYERS * 2 * 2


def test_synthetic_noise_supports_nothing(tmp_path: Path) -> None:
    res = _run(tmp_path, signal=False)
    for t in TRAITS:
        assert not _verdict(res, "pooled", "H1", t)["holds"]
        assert not _verdict(res, "pooled", "H2", t)["holds"]
        assert "IP tracks neither, or fails H4" in res["outcome_patterns"][t]


def test_figures_render(tmp_path: Path) -> None:
    from trait_mtmm.figures_main import write_main_figures

    res = _run(tmp_path, signal=True)
    steer = pd.DataFrame(
        [{"trait": t, "measure": m, "norm_frac": c, "score": c} for t in TRAITS
         for m in ("bc", "sr_balanced") for c in (-0.2, 0.0, 0.2)]
    )
    dprime = pd.DataFrame({"layer": range(4), "dprime_E": [1, 2, 3, 2], "dprime_A": [1, 3, 4, 2],
                           "dprime_mean": [1, 2.5, 3.5, 2]})
    paths = write_main_figures(res, fig_dir=tmp_path / "figs", L_star=L_STAR, steer=steer, dprime=dprime)
    assert len(paths) == 6 and all(p.stat().st_size > 0 for p in paths)
