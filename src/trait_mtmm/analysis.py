"""MTMM correlations and related analyses (Phase 7; smoke subset in Phase 3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SCORE_COLS = ["E_SR", "E_BC", "E_IP", "A_SR", "A_BC", "A_IP"]


def wide_condition_table(
    sr: pd.DataFrame,
    bc: pd.DataFrame,
    ip: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per persona with six trait-method scores."""

    def pivot(df: pd.DataFrame, method: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["persona_id", "persona_set"])
        p = df.pivot_table(
            index=["persona_id", "persona_set"],
            columns="trait",
            values="score",
            aggfunc="mean",
        ).reset_index()
        p.columns = [
            c if c in ("persona_id", "persona_set") else f"{c}_{method}"
            for c in p.columns
        ]
        return p

    out = pivot(sr, "SR")
    for part in (pivot(bc, "BC"), pivot(ip, "IP")):
        if out.empty:
            out = part
        elif not part.empty:
            out = out.merge(part, on=["persona_id", "persona_set"], how="outer")
    return out


def mtmm_correlation(wide: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SCORE_COLS if c in wide.columns]
    if len(cols) < 2 or len(wide) < 2:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    return wide[cols].corr(method="pearson")


def analysis_summary(wide: pd.DataFrame) -> dict[str, Any]:
    corr = mtmm_correlation(wide)
    return {
        "n_conditions": int(len(wide)),
        "columns": list(wide.columns),
        "mtmm_pearson": corr.replace({np.nan: None}).to_dict(),
    }
