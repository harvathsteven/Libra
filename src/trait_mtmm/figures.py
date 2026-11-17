"""Figure generation for smoke and full analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_mtmm_heatmap(corr: pd.DataFrame, path: Path, title: str = "MTMM (smoke)") -> None:
    if corr.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "insufficient data", ha="center")
        ax.axis("off")
        _save(fig, path)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def fig_dprime_by_layer(dprime_df: pd.DataFrame, path: Path, L_star: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dprime_df["layer"], dprime_df["dprime_E"], label="E")
    ax.plot(dprime_df["layer"], dprime_df["dprime_A"], label="A")
    ax.plot(dprime_df["layer"], dprime_df["dprime_mean"], label="mean", linestyle="--")
    if L_star is not None:
        ax.axvline(L_star, color="black", linestyle=":", label=f"L*={L_star}")
    ax.set_xlabel("layer (0-based block)")
    ax.set_ylabel("held-out d'")
    ax.legend()
    ax.set_title("Held-out d' by layer (smoke)")
    _save(fig, path)


def fig_steering_curves(steer_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if steer_df.empty:
        ax.text(0.5, 0.5, "no steering data", ha="center")
        ax.axis("off")
    else:
        for trait, g in steer_df.groupby("trait"):
            ax.plot(g["coefficient"], g["sr_score"], marker="o", label=trait)
        ax.set_xlabel("steering coefficient")
        ax.set_ylabel("SR score")
        ax.legend()
        ax.set_title("Steering curves (smoke)")
    _save(fig, path)


def fig_ip_scatter(wide: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    pairs = [("E_IP", "E_SR", "E: IP vs SR"), ("E_IP", "E_BC", "E: IP vs BC")]
    for ax, (x, y, title) in zip(axes, pairs, strict=True):
        if x in wide.columns and y in wide.columns and len(wide) > 0:
            colors = wide.get("persona_set", "test")
            ax.scatter(wide[x], wide[y], c=pd.Categorical(colors).codes, cmap="tab10")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, "n/a", ha="center")
            ax.axis("off")
    _save(fig, path)


def write_smoke_figures(
    *,
    figures: Path,
    corr: pd.DataFrame,
    dprime_df: pd.DataFrame | None,
    steer_df: pd.DataFrame | None,
    wide: pd.DataFrame,
    L_star: int | None,
) -> list[Path]:
    paths = []
    p = figures / "smoke_mtmm_heatmap.png"
    fig_mtmm_heatmap(corr, p)
    paths.append(p)
    if dprime_df is not None:
        p = figures / "smoke_dprime_by_layer.png"
        fig_dprime_by_layer(dprime_df, p, L_star=L_star)
        paths.append(p)
    if steer_df is not None:
        p = figures / "smoke_steering_curves.png"
        fig_steering_curves(steer_df, p)
        paths.append(p)
    p = figures / "smoke_ip_scatter.png"
    fig_ip_scatter(wide, p)
    paths.append(p)
    # Placeholder panels so Phase 3 "all figures" smoke set is complete-shaped
    for name in ("smoke_null_distributions.png", "smoke_layer_sweep.png"):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"{name}\n(full analysis in Phase 7)", ha="center", va="center")
        ax.axis("off")
        p = figures / name
        _save(fig, p)
        paths.append(p)
    return paths
