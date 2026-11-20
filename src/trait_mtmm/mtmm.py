"""Phase 7 preregistered analysis: PREREGISTRATION.md §2–§3 at tag `prereg-v2`.

This module was written and committed before it was run on the main-run results
(prereg §5). Where the prereg text needed a concrete reading, the choice is logged
in DEVIATIONS.md as a clarification (D-numbers in comments below).

Conventions
- Variables per condition: E_SR, E_BC, E_IP, A_SR, A_BC, A_IP (T = trait, T' = other).
- Bootstraps are represented as frequency-weight matrices W of shape (B, n): W[b, i]
  is how often condition i appears in resample b. A weighted Pearson r with
  frequency weights equals the Pearson r on the expanded resample, so the
  condition-level and the cluster bootstrap share one code path.
- Rows of every sample are sorted by persona_id (D1), so the resample index
  matrix is defined by the seed alone.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

TRAITS = ("E", "A")
METHODS = ("SR", "BC", "IP")
VARS = tuple(f"{t}_{m}" for t in TRAITS for m in METHODS)
SAMPLES = ("pooled", "explicit", "implicit")
METHOD_PAIRS = (("SR", "BC"), ("SR", "IP"), ("BC", "IP"))
OTHER = {"E": "A", "A": "E"}
CORR_METHODS = ("pearson", "spearman")
BOOT_KINDS = ("condition", "cluster")

# Seeds fixed by the prereg (§3.0, §3.2 H4).
SHUFFLE_SEEDS = {"E": 43, "A": 44}

# Implicit personas: the cue sentence follows one of three template markers
# (scripts/build_stimuli.py, build_implicit_personas).
CUE_MARKERS = ("After work, ", "Among the people you see regularly, ", "At work and nearby, ")


# ---------------------------------------------------------------------------
# §2 Exclusions and condition scores
# ---------------------------------------------------------------------------


def condition_table(
    sr_item: pd.DataFrame,
    bc_item: pd.DataFrame,
    ip_probe: pd.DataFrame,
    *,
    n_items: dict[str, dict[str, int]],
    drop_mass: float,
    drop_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply prereg §2 rules 1–4 and return (wide condition table, exclusion log).

    `n_items[method][trait]` is the number of items on the scale (SR 24, BC 100).
    Rows missing from the item table count as dropped (D2). The SR acquiescence
    index (§1.5) is the mean `score_before_keying` over the kept items, and is
    missing when the SR condition score is missing (D3).
    """
    rows: dict[str, dict[str, Any]] = {}
    log: list[dict[str, Any]] = []

    for method, df in (("SR", sr_item), ("BC", bc_item)):
        for (pid, pset, trait), g in df.groupby(["persona_id", "persona_set", "trait"], sort=True):
            if g["item_id"].duplicated().any():
                raise ValueError(f"duplicate {method} items for {pid} {trait}")
            expected = n_items[method][trait]
            kept = g[g["option_mass"] >= drop_mass]  # "dropped if below 0.50"
            n_dropped = expected - len(kept)
            excluded = (n_dropped / expected) > drop_frac  # "more than 20%"
            row = rows.setdefault(pid, {"persona_id": pid, "persona_set": pset})
            row[f"{trait}_{method}"] = np.nan if excluded or kept.empty else float(kept["score"].mean())
            if method == "SR":
                row[f"ACQ_{trait}"] = (
                    np.nan if excluded or kept.empty else float(kept["score_before_keying"].mean())
                )
            log.append(
                {
                    "persona_id": pid,
                    "method": method,
                    "trait": trait,
                    "n_items_expected": expected,
                    "n_items_present": int(len(g)),
                    "n_items_dropped": int(n_dropped),
                    "dropped_item_ids": sorted(set(g["item_id"]) - set(kept["item_id"])),
                    "condition_score_excluded": bool(excluded),
                }
            )

    for (pid, pset, trait), g in ip_probe.groupby(["persona_id", "persona_set", "trait"], sort=True):
        s = float(g["projection"].mean())
        row = rows.setdefault(pid, {"persona_id": pid, "persona_set": pset})
        row[f"{trait}_IP"] = s if np.isfinite(s) else np.nan
        log.append(
            {
                "persona_id": pid,
                "method": "IP",
                "trait": trait,
                "n_items_expected": None,
                "n_items_present": int(len(g)),
                "n_items_dropped": 0,
                "dropped_item_ids": [],
                "condition_score_excluded": not np.isfinite(s),
            }
        )

    wide = pd.DataFrame(list(rows.values()))
    for c in (*VARS, "ACQ_E", "ACQ_A"):
        if c not in wide:
            wide[c] = np.nan
    wide = wide.sort_values("persona_id").reset_index(drop=True)
    return wide, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Samples and clusters
# ---------------------------------------------------------------------------


def implicit_cue(system_prompt: str) -> str:
    for marker in CUE_MARKERS:
        i = system_prompt.find(marker)
        if i >= 0:
            return system_prompt[i + len(marker) :].split(". ")[0].rstrip(".")
    raise ValueError(f"no cue marker in implicit persona: {system_prompt[:60]!r}")


def cluster_labels(personas: Iterable[dict[str, Any]]) -> dict[str, str]:
    """§3.0: implicit personas cluster by cue sentence; explicit personas are singletons."""
    out: dict[str, str] = {}
    for p in personas:
        if p["set"] == "implicit":
            out[p["persona_id"]] = "cue: " + implicit_cue(p["system_prompt"])
        elif p["set"] == "explicit":
            out[p["persona_id"]] = "persona: " + p["persona_id"]
    return out


def build_samples(
    wide: pd.DataFrame, personas: list[dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """§2.5 listwise samples (pooled, explicit, implicit), with cluster and intended levels."""
    meta = pd.DataFrame(
        [
            {
                "persona_id": p["persona_id"],
                "intended_E_numeric": p["intended_E_numeric"],
                "intended_A_numeric": p["intended_A_numeric"],
            }
            for p in personas
            if p["set"] in ("explicit", "implicit")
        ]
    )
    clusters = cluster_labels(personas)
    test = wide[wide["persona_set"].isin(["explicit", "implicit"])].merge(meta, on="persona_id", how="left")
    test["cluster"] = test["persona_id"].map(clusters)
    if test["cluster"].isna().any() or test["intended_E_numeric"].isna().any():
        raise ValueError("test conditions without persona metadata")
    samples: dict[str, pd.DataFrame] = {}
    info: dict[str, Any] = {}
    for name in SAMPLES:
        sub = test if name == "pooled" else test[test["persona_set"] == name]
        complete = sub.dropna(subset=list(VARS)).sort_values("persona_id").reset_index(drop=True)
        samples[name] = complete
        info[name] = {
            "n_conditions": int(len(sub)),
            "n_listwise": int(len(complete)),
            "excluded_ids": sorted(set(sub["persona_id"]) - set(complete["persona_id"])),
            "n_clusters": int(complete["cluster"].nunique()),
        }
    return samples, info


# ---------------------------------------------------------------------------
# Weighted correlations (frequency weights)
# ---------------------------------------------------------------------------


def _const_rows(pos: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.broadcast_to(x, pos.shape)
    hi = np.where(pos, xb, -np.inf).max(axis=1)
    lo = np.where(pos, xb, np.inf).min(axis=1)
    return hi == lo


def wcorr(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r per row of W (B, n). x, y are (n,) or (B, n). NaN where a variable is constant."""
    W = np.asarray(W, dtype=float)
    N = W.sum(axis=1)
    mx = (W * x).sum(axis=1) / N
    my = (W * y).sum(axis=1) / N
    dx = x - mx[:, None]
    dy = y - my[:, None]
    sxy = (W * dx * dy).sum(axis=1)
    sxx = (W * dx * dx).sum(axis=1)
    syy = (W * dy * dy).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sxy / np.sqrt(sxx * syy)
    pos = W > 0
    r[_const_rows(pos, x) | _const_rows(pos, y)] = np.nan
    return np.clip(r, -1.0, 1.0)


def wranks(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Average (mid) ranks of x within each weighted resample, shape (B, n).

    Equal to scipy.stats.rankdata on the expanded resample; conditions with
    weight 0 get a rank that never enters a weighted statistic.
    """
    uniq, inv = np.unique(x, return_inverse=True)
    onehot = np.zeros((len(x), len(uniq)))
    onehot[np.arange(len(x)), inv] = 1.0
    G = np.asarray(W, dtype=float) @ onehot
    before = np.cumsum(G, axis=1) - G
    return (before + (G + 1.0) / 2.0)[:, inv]


def corr_set(W: np.ndarray, cols: dict[str, np.ndarray], method: str) -> dict[tuple[str, str], np.ndarray]:
    """All pairwise correlations among `cols` for each row of W."""
    if method == "spearman":
        z = {k: wranks(W, v) for k, v in cols.items()}
    elif method == "pearson":
        z = cols
    else:
        raise ValueError(method)
    out: dict[tuple[str, str], np.ndarray] = {}
    keys = list(cols)
    for a, b in itertools.combinations(keys, 2):
        r = wcorr(W, z[a], z[b])
        out[(a, b)] = r
        out[(b, a)] = r
    return out


def point_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    W = np.ones((1, len(x)))
    if method == "spearman":
        x, y = wranks(W, x)[0], wranks(W, y)[0]
    return float(wcorr(W, x, y)[0])


# ---------------------------------------------------------------------------
# Bootstrap weight matrices (§3.0)
# ---------------------------------------------------------------------------


def _counts(idx: np.ndarray, m: int) -> np.ndarray:
    B = idx.shape[0]
    flat = (np.arange(B)[:, None] * m + idx).ravel()
    return np.bincount(flat, minlength=B * m).reshape(B, m).astype(float)


def condition_weights(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Resample conditions with replacement: rng.integers(0, n, size=(B, n)) (D1)."""
    return _counts(rng.integers(0, n, size=(B, n)), n)


def cluster_weights(labels: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Resample clusters (sorted labels) with replacement; each condition inherits its cluster's count."""
    uniq, inv = np.unique(np.asarray(labels), return_inverse=True)
    C = len(uniq)
    return _counts(rng.integers(0, C, size=(B, C)), C)[:, inv]


def sample_weights(sample: pd.DataFrame, kind: str, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "condition":
        return condition_weights(len(sample), B, rng)
    if kind == "cluster":
        return cluster_weights(sample["cluster"].to_numpy(), B, rng)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Interval / p helpers
# ---------------------------------------------------------------------------


def interval(boot: np.ndarray) -> dict[str, Any]:
    B = len(boot)
    valid = boot[np.isfinite(boot)]
    n_undef = B - len(valid)
    lo, hi = (np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan))
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot_valid": int(len(valid)),
        "n_boot_undefined": int(n_undef),
        "undefined_over_1pct": bool(n_undef / B > 0.01),
    }


def boot_p(boot: np.ndarray) -> float:
    """§3.0: p = min(1, 2·min(#{d* ≤ 0}, #{d* ≥ 0}) / B), over defined resamples (D4)."""
    valid = boot[np.isfinite(boot)]
    if not len(valid):
        return float("nan")
    return float(min(1.0, 2.0 * min((valid <= 0).sum(), (valid >= 0).sum()) / len(valid)))


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    """Holm step-down; returns adjusted p and reject flag per key."""
    keys = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(keys)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    still = True
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        still = still and pvals[k] <= alpha / (m - i)
        out[k] = {"p_holm": float(running), "reject": bool(still)}
    return out


# ---------------------------------------------------------------------------
# H1–H3 for one sample
# ---------------------------------------------------------------------------


def _v(t: str, m: str) -> str:
    return f"{t}_{m}"


def _stat_defs() -> list[dict[str, Any]]:
    """Every preregistered H1–H3 statistic as a function of a correlation lookup."""
    defs: list[dict[str, Any]] = []
    for t in TRAITS:
        o = OTHER[t]
        for mi, mj in METHOD_PAIRS:
            val = (_v(t, mi), _v(t, mj))
            defs.append(
                {
                    "hypothesis": "H1",
                    "trait": t,
                    "component": f"r({_v(t, mi)},{_v(t, mj)})",
                    "role": "primary",
                    "f": lambda c, val=val: c[val],
                }
            )
            hetero = [
                (_v(t, mi), _v(o, mj)),
                (_v(o, mi), _v(t, mj)),
            ]
            for h in hetero:
                defs.append(
                    {
                        "hypothesis": "H2",
                        "trait": t,
                        "component": f"r({val[0]},{val[1]}) - |r({h[0]},{h[1]})|",
                        "role": "primary",
                        "f": lambda c, val=val, h=h: c[val] - np.abs(c[h]),
                    }
                )
                defs.append(
                    {
                        "hypothesis": "H2",
                        "trait": t,
                        "component": f"r({val[0]},{val[1]}) - r({h[0]},{h[1]})",
                        "role": "secondary_signed",
                        "f": lambda c, val=val, h=h: c[val] - c[h],
                    }
                )
            same_method = [(_v(t, mi), _v(o, mi)), (_v(t, mj), _v(o, mj))]
            for h in same_method:
                defs.append(
                    {
                        "hypothesis": "H2",
                        "trait": t,
                        "component": f"r({val[0]},{val[1]}) - |r({h[0]},{h[1]})|",
                        "role": "secondary_same_method",
                        "f": lambda c, val=val, h=h: c[val] - np.abs(c[h]),
                    }
                )
                defs.append(
                    {
                        "hypothesis": "H2",
                        "trait": t,
                        "component": f"r({val[0]},{val[1]}) - r({h[0]},{h[1]})",
                        "role": "secondary_same_method_signed",
                        "f": lambda c, val=val, h=h: c[val] - c[h],
                    }
                )
        a = (_v(t, "IP"), _v(t, "SR"))
        b = (_v(t, "IP"), _v(t, "BC"))
        defs.append(
            {
                "hypothesis": "H3",
                "trait": t,
                "component": f"r({a[0]},{a[1]}) - r({b[0]},{b[1]})",
                "role": "primary",
                "f": lambda c, a=a, b=b: c[a] - c[b],
            }
        )
    return defs


def h123_sample(
    sample: pd.DataFrame,
    *,
    sample_name: str,
    weights: dict[str, np.ndarray],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """H1–H3 rows (one per statistic × corr method × bootstrap) plus point MTMM matrices."""
    cols = {v: sample[v].to_numpy(dtype=float) for v in VARS}
    rows: list[dict[str, Any]] = []
    matrices: dict[str, pd.DataFrame] = {}
    defs = _stat_defs()
    for method in CORR_METHODS:
        point = {k: float(v[0]) for k, v in corr_set(np.ones((1, len(sample))), cols, method).items()}
        mat = pd.DataFrame(np.eye(len(VARS)), index=list(VARS), columns=list(VARS))
        for (a, b), r in point.items():
            mat.loc[a, b] = r
        matrices[method] = mat
        for kind in BOOT_KINDS:
            boot = corr_set(weights[kind], cols, method)
            for d in defs:
                est = float(d["f"](point))
                bs = np.asarray(d["f"](boot), dtype=float)
                row = {
                    "sample": sample_name,
                    "corr_method": method,
                    "bootstrap": kind,
                    "hypothesis": d["hypothesis"],
                    "trait": d["trait"],
                    "component": d["component"],
                    "role": d["role"],
                    "estimate": est,
                    **interval(bs),
                }
                if d["hypothesis"] == "H3":
                    row["p_boot"] = boot_p(bs)
                    row["passes"] = None  # decided after Holm
                else:
                    row["passes"] = bool(row["ci_low"] > 0)
                rows.append(row)
    return rows, matrices


# ---------------------------------------------------------------------------
# H4 null distributions
# ---------------------------------------------------------------------------


def random_null_vectors(d: int, n: int, seed: int) -> np.ndarray:
    """§3.2 H4(a): n unit vectors, u ~ N(0, I) normalized; one draw rng.standard_normal((n, d)) (D5)."""
    U = np.random.default_rng(seed).standard_normal((n, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def train_activations(acts_trait: dict[str, Any], L: int) -> tuple[np.ndarray, np.ndarray]:
    """Train activations at layer L for one trait: X (n_hi + n_lo, d), labels (True = high)."""
    xs, labels = [], []
    for pole, flag in (("high", True), ("low", False)):
        rec = acts_trait[pole]
        mask = torch.as_tensor(rec["is_train"], dtype=torch.bool)
        x = rec["acts"][mask, L].double().numpy()
        xs.append(x)
        labels.append(np.full(len(x), flag))
    return np.concatenate(xs), np.concatenate(labels)


def shuffled_null_vectors(X: np.ndarray, labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """§3.2 H4(b): permute high/low labels (counts kept) n times via rng.permutation (D5)."""
    rng = np.random.default_rng(seed)
    M = np.stack([rng.permutation(labels) for _ in range(n)]).astype(float)
    n_hi = labels.sum()
    n_lo = len(labels) - n_hi
    V = (M @ X) / n_hi - ((1.0 - M) @ X) / n_lo
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def mean_unit_residuals(residuals: torch.Tensor) -> np.ndarray:
    """(n_probes, n_layers, d) -> (n_layers, d): mean over probes of unit residuals.

    mean_p cos(h_p, u) = (mean_p h_p/|h_p|) · u for any unit u.
    """
    r = residuals.double()
    r = r / r.norm(dim=-1, keepdim=True)
    return r.mean(dim=0).numpy()


def column_corr(S: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Correlation of each column of S (n, k) with y (n,)."""
    if method == "spearman":
        from scipy.stats import rankdata

        S = rankdata(S, axis=0)
        y = rankdata(y)
    Sc = S - S.mean(axis=0)
    yc = y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Sc.T @ yc) / (np.linalg.norm(Sc, axis=0) * np.linalg.norm(yc))


def h4_sample(
    sample: pd.DataFrame,
    *,
    sample_name: str,
    mbar_Lstar: np.ndarray,
    nulls: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """H4 rows and the null correlation arrays (for figures).

    `nulls[kind][trait]` is (n_null, d); the random set is the same for both traits.
    """
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for method in CORR_METHODS:
        for t in TRAITS:
            ip = sample[_v(t, "IP")].to_numpy(dtype=float)
            for target_m in ("SR", "BC"):
                y = sample[_v(t, target_m)].to_numpy(dtype=float)
                target = point_corr(ip, y, method)
                for kind in ("random", "shuffled"):
                    null_r = column_corr(mbar_Lstar @ nulls[kind][t].T, y, method)
                    thr = float(np.percentile(null_r, 95))
                    arrays[f"{sample_name}|{method}|{t}|{target_m}|{kind}"] = null_r
                    rows.append(
                        {
                            "sample": sample_name,
                            "corr_method": method,
                            "bootstrap": "none",
                            "hypothesis": "H4",
                            "trait": t,
                            "component": f"r({_v(t, 'IP')},{_v(t, target_m)}) > p95 {kind} null",
                            "role": "primary",
                            "estimate": target,
                            "null_p95": thr,
                            "frac_null_below": float(np.mean(null_r < target)),
                            "n_null_undefined": int(np.sum(~np.isfinite(null_r))),
                            "passes": bool(target > thr),
                        }
                    )
    return rows, arrays


# ---------------------------------------------------------------------------
# H5 (stratified bootstrap)
# ---------------------------------------------------------------------------


def h5_rows(exp: pd.DataFrame, imp: pd.DataFrame, *, B: int, seed: int) -> list[dict[str, Any]]:
    """Δ_T = r_explicit(T_SR, T_BC) − r_implicit(T_SR, T_BC).

    One default_rng(seed) per bootstrap kind: explicit draws (B, n_exp) first,
    then implicit draws (D6). Cluster kind resamples implicit by cue cluster.
    """
    rows: list[dict[str, Any]] = []
    for kind in BOOT_KINDS:
        rng = np.random.default_rng(seed)
        W_e = condition_weights(len(exp), B, rng)
        W_i = (
            condition_weights(len(imp), B, rng)
            if kind == "condition"
            else cluster_weights(imp["cluster"].to_numpy(), B, rng)
        )
        for method in CORR_METHODS:
            for t in TRAITS:
                xs = [d[[_v(t, "SR"), _v(t, "BC")]].to_numpy(dtype=float) for d in (exp, imp)]
                pe = point_corr(xs[0][:, 0], xs[0][:, 1], method)
                pi = point_corr(xs[1][:, 0], xs[1][:, 1], method)
                be = corr_set(W_e, {"x": xs[0][:, 0], "y": xs[0][:, 1]}, method)[("x", "y")]
                bi = corr_set(W_i, {"x": xs[1][:, 0], "y": xs[1][:, 1]}, method)[("x", "y")]
                row = {
                    "sample": "explicit_vs_implicit",
                    "corr_method": method,
                    "bootstrap": kind,
                    "hypothesis": "H5",
                    "trait": t,
                    "component": f"r_explicit({_v(t, 'SR')},{_v(t, 'BC')}) - r_implicit(...)",
                    "role": "secondary",
                    "estimate": pe - pi,
                    "r_explicit": pe,
                    "r_implicit": pi,
                    **interval(be - bi),
                }
                row["passes"] = bool(row["ci_low"] > 0)
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def apply_h3_holm(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    """§3.0/§3.2: per sample and corr method, p_used = max over both bootstraps; Holm over {E, A}."""
    h3 = [r for r in rows if r["hypothesis"] == "H3"]
    for sample, method in sorted({(r["sample"], r["corr_method"]) for r in h3}):
        group = [r for r in h3 if r["sample"] == sample and r["corr_method"] == method]
        p_used = {t: max(r["p_boot"] for r in group if r["trait"] == t) for t in TRAITS}
        adj = holm(p_used, alpha)
        for r in group:
            r["p_used_max_over_bootstraps"] = p_used[r["trait"]]
            r["p_holm"] = adj[r["trait"]]["p_holm"]
            r["passes"] = adj[r["trait"]]["reject"]
    return rows


def verdicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A component passes only if it passes under both bootstraps; a hypothesis
    holds for a trait only if all its primary components pass (§3.0)."""
    df = pd.DataFrame(rows)
    df = df[df["role"].isin(["primary", "secondary"])]
    out: list[dict[str, Any]] = []
    for (sample, method, hyp, trait), g in df.groupby(["sample", "corr_method", "hypothesis", "trait"], sort=False):
        comps = []
        for comp, gc in g.groupby("component", sort=False):
            by_boot = {r["bootstrap"]: bool(r["passes"]) for _, r in gc.iterrows()}
            comps.append({"component": comp, "estimate": float(gc["estimate"].iloc[0]),
                          "by_bootstrap": by_boot, "passes": all(by_boot.values())})
        entry = {
            "sample": sample,
            "corr_method": method,
            "hypothesis": hyp,
            "trait": trait,
            "holds": all(c["passes"] for c in comps),
            "components": comps,
        }
        if hyp == "H3":
            gr = g.iloc[0]
            entry["d"] = float(gr["estimate"])
            entry["direction"] = "IP-SR > IP-BC" if gr["estimate"] > 0 else "IP-SR < IP-BC"
            entry["p_holm"] = float(gr["p_holm"])
        out.append(entry)
    return out


def outcome_patterns(v: list[dict[str, Any]]) -> dict[str, list[str]]:
    """§3.4 table, applied to Pearson verdicts (D7 gives the operational definitions)."""

    def get(sample: str, hyp: str, t: str) -> dict[str, Any]:
        return next(x for x in v if x["sample"] == sample and x["corr_method"] == "pearson"
                    and x["hypothesis"] == hyp and x["trait"] == t)

    out: dict[str, list[str]] = {}
    for t in TRAITS:
        pats: list[str] = []
        h3 = get("pooled", "H3", t)
        if h3["holds"]:
            pats.append("IP tracks BC more than SR" if h3["d"] < 0 else "IP tracks SR more than BC")
        if get("pooled", "H1", t)["holds"] and get("pooled", "H2", t)["holds"]:
            pats.append("All three converge and discriminate")
        h1 = {c["component"]: c["passes"] for c in get("pooled", "H1", t)["components"]}
        ip_any = h1[f"r({t}_SR,{t}_IP)"] or h1[f"r({t}_BC,{t}_IP)"]
        if not ip_any or not get("pooled", "H4", t)["holds"]:
            pats.append("IP tracks neither, or fails H4")
        if get("explicit", "H1", t)["holds"] and not get("implicit", "H1", t)["holds"]:
            pats.append("Convergence in the explicit set only")
        out[t] = pats
    return out


# ---------------------------------------------------------------------------
# §3.1 preliminary checks, §1.5 acquiescence, §3.3 disattenuation and layer sweep
# ---------------------------------------------------------------------------


def manipulation_check(samples: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for name, s in samples.items():
        for t_int in TRAITS:
            num = s[f"intended_{t_int}_numeric"].to_numpy(dtype=float)
            coding = "level_1_4" if name == "explicit" else "pole_0_1"
            x = num if coding == "level_1_4" else (num >= 3).astype(float)
            for var in VARS:
                rows.append(
                    {
                        "sample": name,
                        "score": var,
                        "intended_trait": t_int,
                        "same_trait": var.startswith(t_int + "_"),
                        "coding": coding,
                        "n": int(len(s)),
                        "pearson_r": point_corr(s[var].to_numpy(dtype=float), x, "pearson"),
                    }
                )
    return rows


def cronbach_alpha(M: np.ndarray) -> tuple[float, int]:
    """Alpha on the item columns that are complete for every row (D8). Returns (alpha, k used)."""
    M = M[:, ~np.isnan(M).any(axis=0)]
    n, k = M.shape
    if k < 2 or n < 2:
        return float("nan"), int(k)
    item_var = M.var(axis=0, ddof=1).sum()
    total_var = M.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan"), int(k)
    return float(k / (k - 1) * (1 - item_var / total_var)), int(k)


def _item_matrix(item: pd.DataFrame, pids: list[str], trait: str, drop_mass: float) -> np.ndarray:
    g = item[(item["trait"] == trait) & item["persona_id"].isin(pids)].copy()
    g.loc[g["option_mass"] < drop_mass, "score"] = np.nan
    piv = g.pivot(index="persona_id", columns="item_id", values="score").reindex(pids)
    return piv.to_numpy(dtype=float)


def reliabilities(
    samples: dict[str, pd.DataFrame],
    sr_item: pd.DataFrame,
    bc_item: pd.DataFrame,
    ip_probe: pd.DataFrame,
    *,
    drop_mass: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, s in samples.items():
        pids = list(s["persona_id"])
        rel: dict[str, dict[str, Any]] = {}
        for t in TRAITS:
            for m, item in (("SR", sr_item), ("BC", bc_item)):
                a, k = cronbach_alpha(_item_matrix(item, pids, t, drop_mass))
                rel[_v(t, m)] = {"kind": "cronbach_alpha", "value": a, "k_items": k}
            g = ip_probe[(ip_probe["trait"] == t) & ip_probe["persona_id"].isin(pids)]
            piv = g.pivot(index="persona_id", columns="probe_index", values="projection").reindex(pids)
            odd = piv.loc[:, [c for c in piv.columns if c % 2 == 1]].mean(axis=1).to_numpy()
            even = piv.loc[:, [c for c in piv.columns if c % 2 == 0]].mean(axis=1).to_numpy()
            r = point_corr(odd, even, "pearson")
            rel[_v(t, "IP")] = {
                "kind": "split_half_spearman_brown",
                "value": float(2 * r / (1 + r)),
                "split_half_r": r,
                "k_probes": int(piv.shape[1]),
            }
        out[name] = rel
    return out


def disattenuated(
    matrices: dict[str, dict[str, pd.DataFrame]], rel: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for name, mats in matrices.items():
        mat = mats["pearson"]
        for a, b in itertools.combinations(VARS, 2):
            ra, rb = rel[name][a]["value"], rel[name][b]["value"]
            ok = np.isfinite(ra) and np.isfinite(rb) and ra > 0 and rb > 0
            val = float(mat.loc[a, b] / np.sqrt(ra * rb)) if ok else None
            rows.append(
                {
                    "sample": name,
                    "pair": f"{a}~{b}",
                    "r": float(mat.loc[a, b]),
                    "rel_a": ra,
                    "rel_b": rb,
                    "r_disattenuated": val,
                    "above_1": bool(val is not None and abs(val) > 1),
                }
            )
    return rows


def acquiescence(samples: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for name, s in samples.items():
        for t in TRAITS:
            acq = s[f"ACQ_{t}"].to_numpy(dtype=float)
            ok = np.isfinite(acq)
            rows.append(
                {
                    "sample": name,
                    "index": f"ACQ_{t}",
                    "n": int(ok.sum()),
                    "mean": float(np.nanmean(acq)),
                    "r_with_E_SR": point_corr(acq[ok], s["E_SR"].to_numpy(dtype=float)[ok], "pearson"),
                    "r_with_A_SR": point_corr(acq[ok], s["A_SR"].to_numpy(dtype=float)[ok], "pearson"),
                }
            )
    return rows


def layer_sweep(
    samples: dict[str, pd.DataFrame],
    mbar: dict[str, np.ndarray],
    vectors_by_layer: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """§3.3: IP at each layer with that layer's vector, correlated with T_SR and T_BC."""
    rows = []
    for name, s in samples.items():
        M = np.stack([mbar[p] for p in s["persona_id"]])  # (n, L, d)
        for t in TRAITS:
            V = vectors_by_layer[t]  # (L, d) unit
            ip = np.einsum("nld,ld->nl", M, V)
            for layer in range(V.shape[0]):
                for target in ("SR", "BC"):
                    y = s[_v(t, target)].to_numpy(dtype=float)
                    for method in CORR_METHODS:
                        rows.append(
                            {
                                "sample": name,
                                "trait": t,
                                "layer": layer,
                                "target": target,
                                "corr_method": method,
                                "r": point_corr(ip[:, layer], y, method),
                            }
                        )
    return rows


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mbar(residual_dir: Path, pids: Iterable[str]) -> dict[str, np.ndarray]:
    out = {}
    for pid in pids:
        rec = torch.load(residual_dir / f"{pid}.pt", weights_only=False)
        if rec["persona_id"] != pid:
            raise ValueError(f"residual file {pid}.pt holds {rec['persona_id']}")
        out[pid] = mean_unit_residuals(rec["residuals"])
    return out


def unit_vectors_by_layer(blob: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for t in TRAITS:
        V = np.stack([torch.as_tensor(v).double().numpy() for v in blob["vectors_by_layer"][t]])
        out[t] = V / np.linalg.norm(V, axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    main_dir: Path,
    vectors_blob: dict[str, Any],
    extraction_acts: dict[str, Any],
    personas: list[dict[str, Any]],
    n_items: dict[str, dict[str, int]],
    L_star: int,
    B: int,
    seed: int,
    n_null: int,
    alpha: float,
    drop_mass: float,
    drop_frac: float,
) -> dict[str, Any]:
    sr_item = pd.read_parquet(main_dir / "sr_item.parquet")
    bc_item = pd.read_parquet(main_dir / "bc_item.parquet")
    ip_probe = pd.read_parquet(main_dir / "ip_probe.parquet")

    wide, excl_log = condition_table(
        sr_item, bc_item, ip_probe, n_items=n_items, drop_mass=drop_mass, drop_frac=drop_frac
    )
    # Cross-check against the scores written by the run (identical when nothing was dropped).
    stored = pd.read_parquet(main_dir / "condition_scores.parquet").set_index("persona_id")
    w = wide.set_index("persona_id")
    no_drops = int(excl_log["n_items_dropped"].sum()) == 0
    max_diff_vs_stored = float(np.nanmax(np.abs(w.loc[stored.index, list(VARS)].to_numpy()
                                                 - stored[list(VARS)].to_numpy())))
    if no_drops and max_diff_vs_stored > 1e-9:
        raise ValueError(f"condition scores disagree with stored table ({max_diff_vs_stored})")

    samples, sample_info = build_samples(wide, personas)

    vbl = unit_vectors_by_layer(vectors_blob)
    v_star = {t: torch.as_tensor(vectors_blob["vectors"][t]).double().numpy() for t in TRAITS}
    for t in TRAITS:
        cos = float(v_star[t] @ vbl[t][L_star] / np.linalg.norm(v_star[t]))
        if cos < 0.999:
            raise ValueError(f"vectors[{t}] is not vectors_by_layer[{t}][{L_star}] (cos {cos})")

    pooled_ids = list(samples["pooled"]["persona_id"])
    mbar = load_mbar(main_dir / "ip_residuals", pooled_ids)
    # Cache check: IP from cached residuals vs primary IP scores.
    ip_cache_diff = max(
        float(np.max(np.abs(np.array([mbar[p][L_star] @ vbl[t][L_star] for p in pooled_ids])
                            - samples["pooled"][_v(t, "IP")].to_numpy())))
        for t in TRAITS
    )
    if ip_cache_diff > 1e-3:
        raise ValueError(f"cached residuals do not reproduce IP scores (max |Δ| {ip_cache_diff})")

    d = vbl["E"].shape[1]
    U = random_null_vectors(d, n_null, seed)
    nulls: dict[str, dict[str, np.ndarray]] = {"random": {t: U for t in TRAITS}, "shuffled": {}}
    shuffle_check = {}
    for t in TRAITS:
        X, labels = train_activations(extraction_acts[t], L_star)
        v_re = X[labels].mean(0) - X[~labels].mean(0)
        shuffle_check[t] = {
            "n_high_train": int(labels.sum()),
            "n_low_train": int((~labels).sum()),
            "cos_unshuffled_vs_vector": float(v_re @ vbl[t][L_star] / np.linalg.norm(v_re)),
        }
        nulls["shuffled"][t] = shuffled_null_vectors(X, labels, n_null, SHUFFLE_SEEDS[t])

    rows: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, pd.DataFrame]] = {}
    null_arrays: dict[str, np.ndarray] = {}
    for name in SAMPLES:
        s = samples[name]
        weights = {kind: sample_weights(s, kind, B, seed) for kind in BOOT_KINDS}
        r123, mats = h123_sample(s, sample_name=name, weights=weights, alpha=alpha)
        rows += r123
        matrices[name] = mats
        M_star = np.stack([mbar[p][L_star] for p in s["persona_id"]])
        r4, arr = h4_sample(s, sample_name=name, mbar_Lstar=M_star, nulls=nulls)
        rows += r4
        null_arrays.update(arr)
    rows += h5_rows(samples["explicit"], samples["implicit"], B=B, seed=seed)
    rows = apply_h3_holm(rows, alpha)
    v = verdicts(rows)

    rel = reliabilities(samples, sr_item, bc_item, ip_probe, drop_mass=drop_mass)
    baseline = wide[wide["persona_set"] == "baseline"]

    return {
        "sample_info": sample_info,
        "exclusions": {
            "n_items_dropped_total": int(excl_log["n_items_dropped"].sum()),
            "condition_scores_excluded": excl_log[excl_log["condition_score_excluded"]][
                ["persona_id", "method", "trait"]
            ].to_dict("records"),
            "max_abs_diff_vs_stored_condition_scores": max_diff_vs_stored,
        },
        "checks": {
            "ip_cache_max_abs_diff": ip_cache_diff,
            "shuffle_null_inputs": shuffle_check,
            "cluster_sizes_implicit": samples["implicit"]["cluster"].value_counts().sort_index().tolist(),
        },
        "rows": rows,
        "verdicts": v,
        "outcome_patterns": outcome_patterns(v),
        "matrices": matrices,
        "manipulation_check": manipulation_check(samples),
        "reliability": rel,
        "disattenuated": disattenuated(matrices, rel),
        "acquiescence": acquiescence(samples),
        "layer_sweep": layer_sweep(samples, mbar, vbl),
        "baseline": baseline[["persona_id", *VARS, "ACQ_E", "ACQ_A"]].to_dict("records"),
        "null_arrays": null_arrays,
        "samples": samples,
        "exclusion_log": excl_log,
    }
