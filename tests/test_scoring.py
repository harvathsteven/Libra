"""Unit tests for option-probability scoring (CPU, no model required)."""

from __future__ import annotations

import torch

from trait_mtmm.scoring import (
    apply_reverse_keying,
    average_option_order_scores,
    average_paired_scores,
    map_reversed_scale_score,
    option_mass_and_probs,
    score_forced_choice,
    score_likert,
)


class FakeTokenizer:
    """Minimal tokenizer stub with a tiny controllable vocab."""

    def __init__(self) -> None:
        # ids: 0=junk, 1="1", 2=" 1", 3="A", 4=" B", 5="2", 6="C", 7="D"
        self._id_to_str = {
            0: "x",
            1: "1",
            2: " 1",
            3: "A",
            4: " B",
            5: "2",
            6: "C",
            7: "D",
            8: "3",
            9: "4",
            10: "5",
        }

    def get_vocab(self) -> dict[str, int]:
        return {s: i for i, s in self._id_to_str.items()}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._id_to_str[i] for i in ids)


def test_discover_option_token_ids_strips_space() -> None:
    from trait_mtmm.scoring import discover_option_token_ids

    tok = FakeTokenizer()
    ids = discover_option_token_ids(tok, ["1", "A", "B"])
    assert set(ids["1"]) == {1, 2}
    assert set(ids["A"]) == {3}
    assert set(ids["B"]) == {4}


def test_renormalization_and_option_mass() -> None:
    # vocab size 11; put mass on option tokens 1 and 5 ("1" and "2")
    logits = torch.full((11,), -1e9)
    logits[1] = 0.0  # "1"
    logits[5] = 0.0  # "2"
    logits[0] = 0.0  # junk also equal — so option mass < 1
    option_ids = {"1": [1], "2": [5], "3": [8], "4": [9], "5": [10]}
    mass, probs = option_mass_and_probs(logits, option_ids, ["1", "2", "3", "4", "5"])
    assert 0.0 < mass < 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert abs(probs["1"] - 0.5) < 1e-5
    assert abs(probs["2"] - 0.5) < 1e-5


def test_score_likert_expected_value() -> None:
    logits = torch.full((11,), -1e9)
    logits[1] = 0.0  # "1"
    logits[10] = 0.0  # "5"
    option_ids = {"1": [1], "2": [5], "3": [8], "4": [9], "5": [10]}
    result = score_likert(logits, option_ids)
    assert abs(result.expected_value - 3.0) < 1e-5
    assert result.option_mass > 0.9


def test_reverse_keying_and_scale_order() -> None:
    assert apply_reverse_keying(2.0, reverse=True) == 4.0
    assert apply_reverse_keying(2.0, reverse=False) == 2.0
    assert map_reversed_scale_score(2.0, scale_reversed=True) == 4.0
    assert map_reversed_scale_score(2.0, scale_reversed=False) == 2.0
    assert average_paired_scores(2.0, 4.0) == 3.0


def test_forced_choice_and_order_average() -> None:
    logits = torch.full((11,), -1e9)
    logits[3] = 0.0  # A
    logits[4] = 0.0  # B
    logits[6] = -1e9  # C absent effectively
    option_ids = {"A": [3], "B": [4], "C": [6], "D": [7]}
    # Make A and B equal and dominate
    logits[6] = -1e9
    logits[7] = -1e9
    result = score_forced_choice(
        logits, option_ids, high_trait_options=["A", "B"]
    )
    assert abs(result.high_trait_probability - 1.0) < 1e-5
    assert average_option_order_scores(0.8, 0.6) == 0.7
