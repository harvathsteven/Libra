"""Phase 1 model tests on the CPU smoke model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from trait_mtmm.model import (
    format_chat,
    forward_with_hidden_states,
    last_token_residual,
    load_model,
    n_transformer_layers,
    next_token_logits,
    seed_everything,
    supports_system_role,
)
from trait_mtmm.scoring import discover_option_token_ids, score_likert


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_cfg() -> dict:
    with open(ROOT / "configs" / "smoke.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def loaded(smoke_cfg: dict):
    seed_everything(smoke_cfg["seed"])
    m = smoke_cfg["model"]
    return load_model(
        m["id"],
        m["revision"],
        dtype=m["dtype"],
        device=m["device"],
    )


def test_chat_formatting_includes_system_and_prefill(loaded) -> None:
    assert supports_system_role(loaded.tokenizer)
    text = format_chat(
        loaded.tokenizer,
        system="You are concise.",
        user="Pick a number.",
        assistant_prefill="Answer:",
    )
    assert "You are concise." in text
    assert "Pick a number." in text
    assert text.endswith("Answer:")


def test_option_token_discovery_on_real_tokenizer(loaded) -> None:
    ids = discover_option_token_ids(loaded.tokenizer, ["1", "2", "3", "4", "5"])
    for opt in ["1", "2", "3", "4", "5"]:
        assert ids[opt], f"missing tokens for {opt}"
        for tid in ids[opt]:
            assert loaded.tokenizer.decode([tid]).strip() == opt

    letter_ids = discover_option_token_ids(loaded.tokenizer, ["A", "B", "C", "D"])
    for opt in ["A", "B", "C", "D"]:
        assert letter_ids[opt]


def test_zero_steering_vector_changes_nothing(loaded) -> None:
    prompt = format_chat(
        loaded.tokenizer,
        system="You are a careful assistant.",
        user="Answer with a single digit from 1 to 5.",
        assistant_prefill="Answer:",
    )
    logits_a, hidden_a = forward_with_hidden_states(loaded, prompt)
    hidden_dim = hidden_a[-1].shape[-1]
    zero = torch.zeros(hidden_dim, dtype=loaded.dtype, device=loaded.device)
    logits_b, hidden_b = forward_with_hidden_states(
        loaded,
        prompt,
        steer_layer=0,
        steer_vector=zero,
        steer_coefficient=1.0,
    )
    assert torch.allclose(logits_a, logits_b, atol=0.0, rtol=0.0)
    assert torch.allclose(hidden_a[-1], hidden_b[-1], atol=0.0, rtol=0.0)


def test_determinism_two_runs_identical(loaded, smoke_cfg: dict) -> None:
    prompt = format_chat(
        loaded.tokenizer,
        user="Reply with one letter: A, B, C, or D.",
        assistant_prefill="Answer:",
    )
    seed_everything(smoke_cfg["seed"])
    logits_1, _ = forward_with_hidden_states(loaded, prompt)
    seed_everything(smoke_cfg["seed"])
    logits_2, _ = forward_with_hidden_states(loaded, prompt)
    assert torch.equal(logits_1, logits_2)


def test_hidden_states_and_likert_scoring_pipeline(loaded) -> None:
    prompt = format_chat(
        loaded.tokenizer,
        user=(
            "How accurately does this describe you: 'I am talkative.' "
            "Answer with a single number 1-5."
        ),
        assistant_prefill="Answer:",
    )
    logits, hidden = forward_with_hidden_states(loaded, prompt)
    assert n_transformer_layers(loaded.model) + 1 == len(hidden)
    residual = last_token_residual(hidden, layer=1)
    assert residual.shape[-1] == hidden[1].shape[-1]

    option_ids = discover_option_token_ids(
        loaded.tokenizer, ["1", "2", "3", "4", "5"]
    )
    result = score_likert(next_token_logits(logits), option_ids)
    assert 0.0 <= result.option_mass <= 1.0
    assert result.expected_value is not None
    assert 1.0 <= result.expected_value <= 5.0 or result.option_mass == 0.0
