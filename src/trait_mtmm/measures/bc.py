"""Behavioral choice (TRAIT) scoring via next-token probabilities."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from trait_mtmm.model import (
    LoadedModel,
    format_chat,
    forward_with_hidden_states,
    next_token_logits,
)
from trait_mtmm.scoring import (
    average_option_order_scores,
    discover_option_token_ids,
    score_forced_choice,
)

OPTIONS = ("A", "B", "C", "D")


def _option_texts(item: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (label_role, text) where label_role is high1/high2/low1/low2."""
    return [
        ("high1", item["response_high1"]),
        ("high2", item["response_high2"]),
        ("low1", item["response_low1"]),
        ("low2", item["response_low2"]),
    ]


def order_options(
    item: dict[str, Any],
    *,
    seed: int,
    reverse: bool,
) -> list[tuple[str, str]]:
    """Seeded shuffle of four options; reverse=True uses the exact reverse order."""
    texts = _option_texts(item)
    rng = random.Random(seed)
    order = list(range(4))
    rng.shuffle(order)
    if reverse:
        order = list(reversed(order))
    return [(OPTIONS[i], texts[j][0], texts[j][1]) for i, j in enumerate(order)]


def bc_order_seed(base_seed: int, item: dict[str, Any], index: int) -> int:
    """Stable per-item option-order seed (index = position in the full BC item list)."""
    return base_seed + sum(ord(c) for c in item["item_id"]) + index


def build_bc_user_prompt(item: dict[str, Any], ordered: list[tuple[str, str, str]]) -> str:
    lines = [item["question"].strip(), "", "Options:"]
    for letter, _role, text in ordered:
        lines.append(f"{letter}. {text}")
    lines.append("")
    lines.append("Answer with a single letter: A, B, C, or D.")
    return "\n".join(lines)


def score_bc_item(
    loaded: LoadedModel,
    item: dict[str, Any],
    *,
    system: str | None,
    assistant_prefill: str,
    option_token_ids: dict[str, list[int]],
    order_seed: int,
    steer_layer: int | None = None,
    steer_vector: Any = None,
    steer_coefficient: float = 0.0,
) -> dict[str, Any]:
    """Score one item in forward and reversed option order; optional steering."""
    scores: list[float] = []
    masses: list[float] = []
    for reverse in (False, True):
        ordered = order_options(item, seed=order_seed, reverse=reverse)
        high_letters = [letter for letter, role, _ in ordered if role.startswith("high")]
        user = build_bc_user_prompt(item, ordered)
        prompt = format_chat(
            loaded.tokenizer,
            system=system,
            user=user,
            assistant_prefill=assistant_prefill,
        )
        logits, _ = forward_with_hidden_states(
            loaded,
            prompt,
            steer_layer=steer_layer,
            steer_vector=steer_vector,
            steer_coefficient=steer_coefficient,
        )
        result = score_forced_choice(
            next_token_logits(logits),
            option_token_ids,
            high_trait_options=high_letters,
        )
        assert result.high_trait_probability is not None
        scores.append(result.high_trait_probability)
        masses.append(result.option_mass)

    return {
        "item_id": item["item_id"],
        "trait": item["trait"],
        "score": average_option_order_scores(scores[0], scores[1]),
        "option_mass": average_option_order_scores(masses[0], masses[1]),
        "option_mass_forward": masses[0],
        "option_mass_reversed": masses[1],
        "order_seed": order_seed,
    }


def run_bc_for_persona(
    loaded: LoadedModel,
    persona: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    assistant_prefill: str,
    base_seed: int,
) -> pd.DataFrame:
    option_ids = discover_option_token_ids(loaded.tokenizer, list(OPTIONS))
    system = persona.get("system_prompt")
    rows = []
    for i, item in enumerate(items):
        order_seed = bc_order_seed(base_seed, item, i)
        row = score_bc_item(
            loaded,
            item,
            system=system,
            assistant_prefill=assistant_prefill,
            option_token_ids=option_ids,
            order_seed=order_seed,
        )
        row["persona_id"] = persona["persona_id"]
        row["persona_set"] = persona.get("set")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_bc(item_df: pd.DataFrame) -> pd.DataFrame:
    if item_df.empty:
        return pd.DataFrame(
            columns=["persona_id", "persona_set", "trait", "score", "mean_option_mass", "n_items"]
        )
    return (
        item_df.groupby(["persona_id", "persona_set", "trait"], dropna=False)
        .agg(score=("score", "mean"), mean_option_mass=("option_mass", "mean"), n_items=("item_id", "count"))
        .reset_index()
    )
