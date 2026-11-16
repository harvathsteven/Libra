"""Phase 0 acceptance: smoke model loads and returns hidden states on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trait_mtmm.model import (
    format_chat,
    forward_with_hidden_states,
    last_token_residual,
    load_model,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_cfg() -> dict:
    with open(ROOT / "configs" / "smoke.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def loaded(smoke_cfg: dict):
    m = smoke_cfg["model"]
    return load_model(
        m["id"],
        m["revision"],
        dtype=m["dtype"],
        device=m["device"],
    )


def test_smoke_model_returns_hidden_states(loaded) -> None:
    prompt = format_chat(
        loaded.tokenizer,
        system="You are a careful assistant.",
        user="Reply with a single digit: 1",
        assistant_prefill="Answer:",
    )
    logits, hidden = forward_with_hidden_states(loaded, prompt)
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert len(hidden) >= 2
    residual = last_token_residual(hidden, layer=1)
    assert residual.ndim == 1
    assert residual.numel() > 0
