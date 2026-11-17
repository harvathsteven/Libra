"""Self-report (IPIP) scoring via next-token probabilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trait_mtmm.model import (
    LoadedModel,
    format_chat,
    forward_with_hidden_states,
    next_token_logits,
)
from trait_mtmm.scoring import (
    apply_reverse_keying,
    average_paired_scores,
    discover_option_token_ids,
    map_reversed_scale_score,
    score_likert,
)

LIKERT_LABELS_STANDARD = (
    "1 = Very Inaccurate\n"
    "2 = Moderately Inaccurate\n"
    "3 = Neither Inaccurate nor Accurate\n"
    "4 = Moderately Accurate\n"
    "5 = Very Accurate"
)
LIKERT_LABELS_REVERSED = (
    "1 = Very Accurate\n"
    "2 = Moderately Accurate\n"
    "3 = Neither Inaccurate nor Accurate\n"
    "4 = Moderately Inaccurate\n"
    "5 = Very Inaccurate"
)


def build_sr_user_prompt(item_text: str, *, scale_reversed: bool) -> str:
    labels = LIKERT_LABELS_REVERSED if scale_reversed else LIKERT_LABELS_STANDARD
    return (
        "Rate how accurately the following statement describes you.\n"
        f"Statement: {item_text}\n\n"
        f"{labels}\n\n"
        "Answer with a single number from 1 to 5."
    )


def score_sr_item(
    loaded: LoadedModel,
    item: dict[str, Any],
    *,
    system: str | None,
    assistant_prefill: str,
    option_token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    scores: list[float] = []
    masses: list[float] = []
    for scale_reversed in (False, True):
        user = build_sr_user_prompt(item["text"], scale_reversed=scale_reversed)
        prompt = format_chat(
            loaded.tokenizer,
            system=system,
            user=user,
            assistant_prefill=assistant_prefill,
        )
        logits, _ = forward_with_hidden_states(loaded, prompt)
        result = score_likert(next_token_logits(logits), option_token_ids)
        assert result.expected_value is not None
        mapped = map_reversed_scale_score(
            result.expected_value, scale_reversed=scale_reversed
        )
        scores.append(mapped)
        masses.append(result.option_mass)

    avg = average_paired_scores(scores[0], scores[1])
    keyed = apply_reverse_keying(avg, reverse=(item["keying"] == "-"))
    return {
        "item_id": item["item_id"],
        "trait": item["trait"],
        "facet": item["facet"],
        "keying": item["keying"],
        "score": keyed,
        "score_before_keying": avg,
        "option_mass": average_paired_scores(masses[0], masses[1]),
        "option_mass_standard": masses[0],
        "option_mass_reversed": masses[1],
    }


def run_sr_for_persona(
    loaded: LoadedModel,
    persona: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    assistant_prefill: str,
) -> pd.DataFrame:
    option_ids = discover_option_token_ids(loaded.tokenizer, ["1", "2", "3", "4", "5"])
    system = persona.get("system_prompt")
    rows = []
    for item in items:
        row = score_sr_item(
            loaded,
            item,
            system=system,
            assistant_prefill=assistant_prefill,
            option_token_ids=option_ids,
        )
        row["persona_id"] = persona["persona_id"]
        row["persona_set"] = persona.get("set")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sr(item_df: pd.DataFrame) -> pd.DataFrame:
    """Trait score = mean over items per persona × trait."""
    if item_df.empty:
        return pd.DataFrame(
            columns=["persona_id", "persona_set", "trait", "score", "mean_option_mass", "n_items"]
        )
    g = (
        item_df.groupby(["persona_id", "persona_set", "trait"], dropna=False)
        .agg(score=("score", "mean"), mean_option_mass=("option_mass", "mean"), n_items=("item_id", "count"))
        .reset_index()
    )
    return g
