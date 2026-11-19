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
    n_transformer_layers,
)


def project(residual: torch.Tensor, vector: torch.Tensor) -> float:
    """IP score for one residual: cosine similarity with the trait vector.

    Both the residual and the vector are L2-normalized, so this is cos(h, v),
    the same quantity used for held-out d' during extraction.
    """
    r = F.normalize(residual.float().flatten(), dim=0)
    # Saved vectors load on CPU; residuals live on the model device (cuda on GPU runs).
    v = F.normalize(vector.to(device=r.device).float().flatten(), dim=0)
    return float(torch.dot(r, v).item())


def run_ip_for_persona(
    loaded: LoadedModel,
    persona: dict[str, Any],
    probes: list[str],
    *,
    vectors: dict[str, torch.Tensor],
    layer_hidden_index: int,
    assistant_prefill: str | None = None,
    return_residuals: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, torch.Tensor]:
    """Per-probe IP scores (cosine with each trait vector) at one residual layer.

    `layer_hidden_index` matches `hidden_states` indexing (1 = after first block).
    `vectors` maps trait key 'E'/'A' to unit vectors.

    With ``return_residuals=True`` also returns the post-block last-token
    residuals at every layer, float16 on CPU, shape [n_probes, n_layers, d];
    index ``[:, L]`` equals ``hidden_states[L + 1]``. These feed the H4 null
    distributions and the layer sweep.
    """
    system = persona.get("system_prompt")
    rows: list[dict[str, Any]] = []
    cached: list[torch.Tensor] = []
    n_layers = n_transformer_layers(loaded.model) if return_residuals else 0
    for qi, question in enumerate(probes):
        prompt = format_chat(
            loaded.tokenizer,
            system=system,
            user=question,
            assistant_prefill=assistant_prefill,
        )
        _, hidden = forward_with_hidden_states(loaded, prompt)
        residual = last_token_residual(hidden, layer_hidden_index)
        if return_residuals:
            cached.append(
                torch.stack(
                    [last_token_residual(hidden, i + 1) for i in range(n_layers)]
                ).to(torch.float16).cpu()
            )
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
    df = pd.DataFrame(rows)
    if return_residuals:
        return df, torch.stack(cached)
    return df


def cosine_scores(residuals: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Vectorized IP scores: cosine of each residual [..., d] with ``vector``."""
    r = F.normalize(residuals.float(), dim=-1)
    v = F.normalize(vector.to(device=r.device).float().flatten(), dim=0)
    return r @ v


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
