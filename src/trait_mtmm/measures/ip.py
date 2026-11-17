"""Internal projection scoring onto trait vectors."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from trait_mtmm.model import (
    LoadedModel,
    format_chat,
    forward_with_hidden_states,
    last_token_residual,
)


def project(residual: torch.Tensor, vector: torch.Tensor) -> float:
    r = F.normalize(residual.float().flatten(), dim=0)
    v = F.normalize(vector.float().flatten(), dim=0)
    return float(torch.dot(r, v).item())


def run_ip_for_persona(
    loaded: LoadedModel,
    persona: dict[str, Any],
    probes: list[str],
    *,
    vectors: dict[str, torch.Tensor],
    layer_hidden_index: int,
    assistant_prefill: str | None = None,
) -> pd.DataFrame:
    """Mean projection over probes at the chosen residual layer index.

    `layer_hidden_index` matches `hidden_states` indexing (1 = after first block).
    `vectors` maps trait key 'E'/'A' to unit vectors.
    """
    system = persona.get("system_prompt")
    rows: list[dict[str, Any]] = []
    for qi, question in enumerate(probes):
        prompt = format_chat(
            loaded.tokenizer,
            system=system,
            user=question,
            assistant_prefill=assistant_prefill,
        )
        _, hidden = forward_with_hidden_states(loaded, prompt)
        residual = last_token_residual(hidden, layer_hidden_index)
        for trait, vec in vectors.items():
            rows.append(
                {
                    "persona_id": persona["persona_id"],
                    "persona_set": persona.get("set"),
                    "trait": trait,
                    "probe_index": qi,
                    "probe": question,
                    "projection": project(residual, vec),
                }
            )
    return pd.DataFrame(rows)


def aggregate_ip(probe_df: pd.DataFrame) -> pd.DataFrame:
    if probe_df.empty:
        return pd.DataFrame(
            columns=["persona_id", "persona_set", "trait", "score", "n_probes"]
        )
    return (
        probe_df.groupby(["persona_id", "persona_set", "trait"], dropna=False)
        .agg(score=("projection", "mean"), n_probes=("probe_index", "count"))
        .reset_index()
    )
