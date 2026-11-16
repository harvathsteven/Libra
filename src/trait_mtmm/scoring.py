"""Option-token probability scoring (shared SR/BC primitive)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


def discover_option_token_ids(
    tokenizer: Any,
    options: Sequence[str],
) -> dict[str, list[int]]:
    """Map each option string to tokenizer IDs that decode (stripped) to that option.

    Covers bare tokens and common space-prefixed variants without inventing
    multi-token sequences: only single-token IDs whose stripped decode matches.
    """
    wanted = {opt: [] for opt in options}
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        # Prefer decode for special-space variants; fall back to raw string.
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        except Exception:
            decoded = token_str
        stripped = decoded.strip()
        if stripped in wanted:
            wanted[stripped].append(int(token_id))

    missing = [opt for opt, ids in wanted.items() if not ids]
    if missing:
        raise ValueError(f"No token IDs found for options: {missing}")
    # Stable unique IDs
    return {opt: sorted(set(ids)) for opt, ids in wanted.items()}


def option_mass_and_probs(
    logits: torch.Tensor,
    option_token_ids: dict[str, list[int]],
    options: Sequence[str],
) -> tuple[float, dict[str, float]]:
    """From next-token logits [vocab], return option mass and renormalized probs.

    option mass = sum of probabilities on all valid option token IDs (unnormalized
    share of the full distribution). Renormalized probs sum to 1 over options.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    probs = F.softmax(logits.float(), dim=-1)
    raw: dict[str, float] = {}
    for opt in options:
        ids = option_token_ids[opt]
        raw[opt] = float(probs[ids].sum().item())
    mass = float(sum(raw.values()))
    if mass <= 0.0:
        renorm = {opt: 0.0 for opt in options}
    else:
        renorm = {opt: raw[opt] / mass for opt in options}
    return mass, renorm


@dataclass(frozen=True)
class OptionScore:
    option_mass: float
    probs: dict[str, float]
    expected_value: float | None = None
    high_trait_probability: float | None = None


def score_likert(
    logits: torch.Tensor,
    option_token_ids: dict[str, list[int]],
    *,
    options: Sequence[str] = ("1", "2", "3", "4", "5"),
    values: Sequence[float] | None = None,
) -> OptionScore:
    """SR item: expected rating under renormalized digit distribution."""
    if values is None:
        values = [float(o) for o in options]
    if len(values) != len(options):
        raise ValueError("values must match options length")
    mass, probs = option_mass_and_probs(logits, option_token_ids, options)
    expected = sum(probs[o] * v for o, v in zip(options, values, strict=True))
    return OptionScore(option_mass=mass, probs=probs, expected_value=expected)


def apply_reverse_keying(score: float, *, reverse: bool, scale_max: float = 6.0) -> float:
    """IPIP reverse-keyed items: scale_max - score (for 1–5 ratings, 6 - score)."""
    if reverse:
        return scale_max - score
    return score


def map_reversed_scale_score(
    expected_value: float,
    *,
    scale_reversed: bool,
    scale_max: float = 6.0,
) -> float:
    """Map a score collected under a reversed response-label order back to standard.

    Standard: 1 = Very Inaccurate … 5 = Very Accurate.
    Reversed prompt: 1 = Very Accurate … 5 = Very Inaccurate.
    Mapping back uses the same 6 - x transform as reverse keying.
    """
    if scale_reversed:
        return scale_max - expected_value
    return expected_value


def average_paired_scores(a: float, b: float) -> float:
    return 0.5 * (a + b)


def score_forced_choice(
    logits: torch.Tensor,
    option_token_ids: dict[str, list[int]],
    *,
    options: Sequence[str] = ("A", "B", "C", "D"),
    high_trait_options: Iterable[str],
) -> OptionScore:
    """BC item: P(high-trait options) / P(all four) under renormalized mass."""
    mass, probs = option_mass_and_probs(logits, option_token_ids, options)
    high = set(high_trait_options)
    unknown = high - set(options)
    if unknown:
        raise ValueError(f"high_trait_options not in options: {unknown}")
    high_p = sum(probs[o] for o in options if o in high)
    return OptionScore(
        option_mass=mass,
        probs=probs,
        high_trait_probability=high_p,
    )


def average_option_order_scores(
    score_forward: float,
    score_reversed_order: float,
) -> float:
    """Average a BC item score with its exact reverse option-order twin."""
    return average_paired_scores(score_forward, score_reversed_order)
