"""Phase 7 figures (PREREGISTRATION.md §3.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from trait_mtmm.mtmm import SAMPLES, TRAITS, VARS  # noqa: E402

SET_COLORS = {"explicit": "#2a6fdb", "implicit": "#e07b39"}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_mtmm(matrices: dict[str, dict[str, pd.DataFrame]], info: dict[str, Any], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, name in zip(axes, SAMPLES, strict=True):
        mat = matrices[name]["pearson"].loc[list(VARS), list(VARS)].to_numpy(dtype=float)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(len(VARS)):
            for j in range(len(VARS)):
                if i != j:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
                ti, mi = VARS[i].split("_")
                tj, mj = VARS[j].split("_")
                if ti == tj and mi != mj:  # validity diagonal: same trait, different method
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2, ec="black"))
        ax.set_xticks(range(len(VARS)), VARS, rotation=45, ha="right")
        ax.set_yticks(range(len(VARS)), VARS)
        ax.set_title(f"{name} (N = {info[name]['n_listwise']})")
    fig.colorbar(im, ax=axes, fraction=0.02, label="Pearson r")
    fig.suptitle("MTMM correlations; boxed cells = validity diagonal (same trait, different method)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ip_scatter(pooled: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, t in enumerate(TRAITS):
        for col, target in enumerate(("SR", "BC")):
            ax = axes[row, col]
            for pset, g in pooled.groupby("persona_set"):
                ax.scatter(g[f"{t}_IP"], g[f"{t}_{target}"], s=18, alpha=0.8,
                           color=SET_COLORS.get(pset, "grey"), label=pset)
            ax.set_xlabel(f"{t}_IP (mean cosine)")
            ax.set_ylabel(f"{t}_{target}")
            ax.set_title(f"{t}: IP vs {target}")
    axes[0, 0].legend(frameon=False)
    return _save(fig, path)


def fig_layer_sweep(sweep: list[dict[str, Any]], L_star: int, path: Path) -> Path:
    df = pd.DataFrame(sweep)
    df = df[(df["sample"] == "pooled") & (df["corr_method"] == "pearson")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, t in zip(axes, TRAITS, strict=True):
        for target, style in (("SR", "-"), ("BC", "--")):
            g = df[(df["trait"] == t) & (df["target"] == target)].sort_values("layer")
            ax.plot(g["layer"], g["r"], style, marker="o", ms=3, label=f"r({t}_IP, {t}_{target})")
        ax.axvline(L_star, color="black", ls=":", label=f"L* = {L_star} (preregistered)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("layer (0-based block)")
        ax.set_title(f"{t}: layer sweep (pooled, robustness only)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Pearson r")
    return _save(fig, path)


def fig_nulls(null_arrays: dict[str, np.ndarray], rows: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    h4 = [r for r in rows if r["hypothesis"] == "H4" and r["sample"] == "pooled" and r["corr_method"] == "pearson"]
    for i, t in enumerate(TRAITS):
        for j, target in enumerate(("SR", "BC")):
            ax = axes[i, j]
            target_r = None
            for kind, color in (("random", "#888888"), ("shuffled", "#2a9d8f")):
                arr = null_arrays[f"pooled|pearson|{t}|{target}|{kind}"]
                ax.hist(arr, bins=40, alpha=0.55, color=color, label=f"{kind} null")
                ax.axvline(np.percentile(arr, 95), color=color, ls="--", lw=1)
                r = next(x for x in h4 if x["trait"] == t and f"{t}_{target})" in x["component"]
                         and kind in x["component"])
                target_r = r["estimate"]
            ax.axvline(target_r, color="crimson", lw=2, label=f"trait vector r = {target_r:.2f}")
            ax.set_title(f"{t}: r({t}_IP, {t}_{target}) vs nulls (pooled)")
            ax.set_xlabel("Pearson r")
            ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_steering(steer: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, measure, ylabel in (
        (axes[0], "bc", "BC P(high-trait option) — gate v2 (primary)"),
        (axes[1], "sr_balanced", "SR keying-balanced score (secondary)"),
    ):
        g = steer[steer["measure"] == measure]
        for t, gt in g.groupby("trait"):
            ax.plot(gt["norm_frac"], gt["score"], marker="o", label=t)
        ax.set_xlabel("steering coefficient (× mean ‖h‖ at L*)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.suptitle("Steering validation on the baseline assistant (Phase 4)")
    return _save(fig, path)


def fig_dprime(dprime: pd.DataFrame, L_star: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dprime["layer"], dprime["dprime_E"], label="E")
    ax.plot(dprime["layer"], dprime["dprime_A"], label="A")
    ax.plot(dprime["layer"], dprime["dprime_mean"], ls="--", label="mean")
    ax.axvline(L_star, color="black", ls=":", label=f"L* = {L_star}")
    ax.axhline(2, color="grey", lw=0.5, ls="--", label="gate d′ = 2")
    ax.set_xlabel("layer (0-based block)")
    ax.set_ylabel("held-out d′")
    ax.set_title("Held-out d′ by layer (extraction set)")
    ax.legend(frameon=False)
    return _save(fig, path)


def write_main_figures(
    res: dict[str, Any],
    *,
    fig_dir: Path,
    L_star: int,
    steer: pd.DataFrame | None,
    dprime: pd.DataFrame | None,
) -> list[Path]:
    paths = [
        fig_mtmm(res["matrices"], res["sample_info"], fig_dir / "main_1_mtmm_heatmap.png"),
        fig_ip_scatter(res["samples"]["pooled"], fig_dir / "main_2_ip_scatter.png"),
        fig_layer_sweep(res["layer_sweep"], L_star, fig_dir / "main_3_layer_sweep.png"),
        fig_nulls(res["null_arrays"], res["rows"], fig_dir / "main_4_null_distributions.png"),
    ]
    if steer is not None:
        paths.append(fig_steering(steer, fig_dir / "main_5_steering_curves.png"))
    if dprime is not None:
        paths.append(fig_dprime(dprime, L_star, fig_dir / "main_6_dprime_by_layer.png"))
    return paths
