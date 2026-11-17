"""Fast unit tests for vector helpers (no model)."""

from __future__ import annotations

import numpy as np
import torch

from trait_mtmm.vectors import adjective_triples, choose_L_star, dprime, mean_difference_vector


def test_choose_L_star_prefers_middle_on_tie() -> None:
    # Equal d' everywhere -> middle layer
    dprimes = {"E": [1.0, 1.0, 1.0, 1.0, 1.0], "A": [1.0, 1.0, 1.0, 1.0, 1.0]}
    assert choose_L_star(dprimes) == 2


def test_choose_L_star_maximizes_mean() -> None:
    dprimes = {
        "E": [0.1, 0.2, 5.0, 0.2, 0.1],
        "A": [0.1, 0.2, 4.0, 0.2, 0.1],
    }
    assert choose_L_star(dprimes) == 2


def test_mean_difference_and_dprime() -> None:
    high = [torch.tensor([1.0, 0.0]), torch.tensor([1.1, 0.0])]
    low = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.1])]
    v = mean_difference_vector(high, low)
    assert abs(float(v.norm()) - 1.0) < 1e-5
    dp = dprime(np.array([2.0, 2.2]), np.array([0.0, 0.1]))
    assert dp > 0


def test_adjective_triples_deterministic() -> None:
    words = [f"w{i}" for i in range(8)]
    a = adjective_triples(words, n=3, seed=42)
    b = adjective_triples(words, n=3, seed=42)
    assert a == b
    assert len(a) == 3
    assert len(a[0]) == 3
