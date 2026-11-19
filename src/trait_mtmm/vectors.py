"""Trait vector extraction, layer selection, and validation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from trait_mtmm.model import (
    LoadedModel,
    format_chat,
    forward_with_hidden_states,
    last_token_residual,
    n_transformer_layers,
)
from trait_mtmm.scoring import discover_option_token_ids
from trait_mtmm.model import next_token_logits


def adjective_triples(adjectives: list[str], *, n: int, seed: int) -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(seed)
    if len(adjectives) < 3:
        raise ValueError("Need at least 3 adjectives for triples")
    # Sample without replacement when possible
    combos = list(itertools.combinations(sorted(adjectives), 3))
    rng.shuffle(combos)
    return [tuple(c) for c in combos[:n]]


def extraction_system_prompt(adj1: str, adj2: str, adj3: str) -> str:
    return f"You are {adj1}, {adj2}, and {adj3}."


def collect_residuals(
    loaded: LoadedModel,
    *,
    system: str,
    questions: list[str],
) -> list[tuple[torch.Tensor, ...]]:
    """For each question, return all-layer last-token residuals (CPU float32)."""
    out: list[tuple[torch.Tensor, ...]] = []
    for q in questions:
        prompt = format_chat(loaded.tokenizer, system=system, user=q)
        _, hidden = forward_with_hidden_states(loaded, prompt)
        # Store post-block residuals only: indices 1..N
        n_layers = n_transformer_layers(loaded.model)
        layers = tuple(
            last_token_residual(hidden, i + 1).float().cpu() for i in range(n_layers)
        )
        out.append(layers)
    return out


def mean_difference_vector(
    high: list[torch.Tensor],
    low: list[torch.Tensor],
) -> torch.Tensor:
    h = torch.stack(high).mean(dim=0)
    l = torch.stack(low).mean(dim=0)
    v = h - l
    return F.normalize(v.float(), dim=0)


def dprime(high_proj: np.ndarray, low_proj: np.ndarray) -> float:
    mu_h, mu_l = high_proj.mean(), low_proj.mean()
    var_h, var_l = high_proj.var(ddof=1), low_proj.var(ddof=1)
    pooled = np.sqrt(0.5 * (var_h + var_l))
    if pooled < 1e-12:
        return 0.0
    return float((mu_h - mu_l) / pooled)


def choose_L_star(dprimes: dict[str, list[float]]) -> int:
    """Layer index (0-based over transformer blocks) maximizing mean d' across traits.

    On ties, prefer the layer closest to the middle of the network.
    """
    traits = list(dprimes.keys())
    n_layers = len(dprimes[traits[0]])
    mid = (n_layers - 1) / 2.0
    best_layer = 0
    best_key: tuple[float, float] | None = None
    for layer in range(n_layers):
        mean_dp = float(np.mean([dprimes[t][layer] for t in traits]))
        # Maximize mean d'; on ties minimize distance to middle
        key = (mean_dp, -abs(layer - mid))
        if best_key is None or key > best_key:
            best_key = key
            best_layer = layer
    return best_layer



TRAIT_NAMES = {"E": "extraversion", "A": "agreeableness"}


def collect_extraction_activations(
    loaded: LoadedModel,
    adjectives: dict[str, Any],
    extraction_questions: list[str],
    *,
    seed: int,
    n_triples: int = 6,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the extraction prompts and return all-layer last-token residuals.

    Returns ``{trait: {pole: {"acts": Tensor[n, n_layers, d] float32,
    "is_train": list[bool], "triple": list, "question_index": list}}}``.
    The RNG call order matches the Phase 4 extraction exactly (E high, E low,
    A high, A low), so the split and vectors are reproducible from the seed.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict[str, Any]]] = {"E": {}, "A": {}}
    for trait, name in TRAIT_NAMES.items():
        for pole in ("high", "low"):
            words = adjectives[name][pole]["extraction"]
            pole_seed = seed + (0 if trait == "E" else 100) + (0 if pole == "high" else 10)
            triples = adjective_triples(words, n=n_triples, seed=pole_seed)
            prompts = [
                {
                    "triple": triple,
                    "question": q,
                    "question_index": qi,
                    "system": extraction_system_prompt(*triple),
                }
                for triple in triples
                for qi, q in enumerate(extraction_questions)
            ]
            out[trait][pole] = {"prompts": prompts}

    # 80/20 split: a prompt is train only if both its adjective triple and its
    # question are in the train subsets; every other prompt is held out.
    for trait in ("E", "A"):
        for pole in ("high", "low"):
            prompts = out[trait][pole].pop("prompts")
            triples = sorted({p["triple"] for p in prompts})
            questions = sorted({p["question"] for p in prompts})
            rng.shuffle(triples)
            rng.shuffle(questions)
            n_t_train = max(1, int(0.8 * len(triples)))
            n_q_train = max(1, int(0.8 * len(questions)))
            train_triples = set(triples[:n_t_train])
            train_questions = set(questions[:n_q_train])
            is_train = [
                p["triple"] in train_triples and p["question"] in train_questions
                for p in prompts
            ]
            # Ensure a non-empty holdout (only reachable on tiny smoke configs):
            # move the last train prompt to holdout, as in the original code.
            if not any(not t for t in is_train) and any(is_train):
                last = max(i for i, t in enumerate(is_train) if t)
                is_train[last] = False
            layer_stacks = [
                torch.stack(
                    collect_residuals(loaded, system=p["system"], questions=[p["question"]])[0]
                )
                for p in prompts
            ]
            out[trait][pole].update(
                {
                    "acts": torch.stack(layer_stacks),  # [n, n_layers, d]
                    "is_train": is_train,
                    "triple": [p["triple"] for p in prompts],
                    "question_index": [p["question_index"] for p in prompts],
                }
            )
    return out


def split_activations(
    acts: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, list[list[torch.Tensor]]]], dict[str, dict[str, list[list[torch.Tensor]]]]]:
    """Convert stacked activations to per-layer train/holdout lists."""
    train: dict[str, dict[str, list[list[torch.Tensor]]]] = {}
    hold: dict[str, dict[str, list[list[torch.Tensor]]]] = {}
    for trait, poles in acts.items():
        train[trait], hold[trait] = {}, {}
        for pole, rec in poles.items():
            a = rec["acts"].float()
            n_layers = a.shape[1]
            tr = [i for i, t in enumerate(rec["is_train"]) if t]
            ho = [i for i, t in enumerate(rec["is_train"]) if not t]
            train[trait][pole] = [[a[i, layer] for i in tr] for layer in range(n_layers)]
            hold[trait][pole] = [[a[i, layer] for i in ho] for layer in range(n_layers)]
    return train, hold


def save_extraction_activations(acts: dict[str, Any], path: Path) -> None:
    """Save extraction activations (float16) for the label-shuffled null (H4)."""
    blob = {
        trait: {
            pole: {**rec, "acts": rec["acts"].to(torch.float16).cpu()}
            for pole, rec in poles.items()
        }
        for trait, poles in acts.items()
    }
    torch.save(blob, path)


def vectors_from_activations(
    acts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[torch.Tensor]]:
    """Recompute per-layer mean-difference unit vectors from the train split."""
    train, _ = split_activations(acts)
    out: dict[str, list[torch.Tensor]] = {}
    for trait in train:
        n_layers = len(train[trait]["high"])
        out[trait] = [
            mean_difference_vector(train[trait]["high"][layer], train[trait]["low"][layer])
            for layer in range(n_layers)
        ]
    return out


def extract_and_validate(
    loaded: LoadedModel,
    adjectives: dict[str, Any],
    extraction_questions: list[str],
    *,
    seed: int,
    n_triples: int = 6,
    out_dir: Path,
    sr_items: list[dict[str, Any]] | None = None,
    bc_items: list[dict[str, Any]] | None = None,
    steer_norm_fracs: tuple[float, ...] | None = None,
    assistant_prefill: str = "",
    sr_format: str = "letters",
    enforce_gates: bool = True,
    min_dprime: float = 2.0,
    min_steer_spearman: float = 0.9,
    min_option_mass: float = 0.80,
) -> dict[str, Any]:
    """Run extraction, choose L*, write vectors and validation reports.

    Returns a summary dict including L_star and whether gates passed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_layers = n_transformer_layers(loaded.model)

    acts = collect_extraction_activations(
        loaded, adjectives, extraction_questions, seed=seed, n_triples=n_triples
    )
    save_extraction_activations(acts, out_dir / "extraction_activations.pt")
    train_acts, hold_acts = split_activations(acts)

    vectors_by_layer: dict[str, list[torch.Tensor]] = {"E": [], "A": []}
    dprimes: dict[str, list[float]] = {"E": [], "A": []}
    for trait in ("E", "A"):
        for layer_i in range(n_layers):
            vec = mean_difference_vector(
                train_acts[trait]["high"][layer_i],
                train_acts[trait]["low"][layer_i],
            )
            vectors_by_layer[trait].append(vec)
            high_p = np.array(
                [
                    float(torch.dot(F.normalize(x, dim=0), vec).item())
                    for x in hold_acts[trait]["high"][layer_i]
                ]
            )
            low_p = np.array(
                [
                    float(torch.dot(F.normalize(x, dim=0), vec).item())
                    for x in hold_acts[trait]["low"][layer_i]
                ]
            )
            dprimes[trait].append(dprime(high_p, low_p))

    L_star = choose_L_star(dprimes)
    vectors = {
        "E": vectors_by_layer["E"][L_star],
        "A": vectors_by_layer["A"][L_star],
    }
    cosine_EA = float(
        torch.dot(vectors["E"], vectors["A"]).item()
    )  # already unit

    # Save vectors
    torch.save(
        {
            "L_star": L_star,
            "vectors": {k: v.cpu() for k, v in vectors.items()},
            "vectors_by_layer": {
                k: [vv.cpu() for vv in vs] for k, vs in vectors_by_layer.items()
            },
            "dprimes": dprimes,
            "cosine_EA_at_L_star": cosine_EA,
        },
        out_dir / "trait_vectors.pt",
    )
    dprime_df = pd.DataFrame(
        {
            "layer": list(range(n_layers)),
            "dprime_E": dprimes["E"],
            "dprime_A": dprimes["A"],
            "dprime_mean": [
                0.5 * (dprimes["E"][i] + dprimes["A"][i]) for i in range(n_layers)
            ],
        }
    )
    dprime_df.to_parquet(out_dir / "dprime_by_layer.parquet", index=False)

    sep_ok = all(dprimes[t][L_star] >= min_dprime for t in ("E", "A"))

    # Steering validation (gate v2, DECISIONS 2026-11-17): baseline assistant,
    # coefficients scaled to ||h|| at L*, criterion = BC forced choice.
    steer_report: dict[str, Any] = {"skipped": bc_items is None}
    steer_ok = True
    if bc_items is not None:
        steer_ok, steer_report, steer_df = validate_steering(
            loaded,
            vectors,
            L_star=L_star,
            bc_items=bc_items,
            sr_items=sr_items,
            assistant_prefill=assistant_prefill,
            sr_format=sr_format,
            seed=seed,
            norm_fracs=steer_norm_fracs,
            min_spearman=min_steer_spearman,
            min_option_mass=min_option_mass,
        )
        steer_df.to_parquet(out_dir / "steering_curves.parquet", index=False)

    summary = {
        "L_star": L_star,
        "n_layers": n_layers,
        "dprime_at_L_star": {t: dprimes[t][L_star] for t in ("E", "A")},
        "cosine_EA_at_L_star": cosine_EA,
        "separation_passed": sep_ok,
        "steering_passed": steer_ok if bc_items is not None else None,
        "steering": steer_report,
        "gates_enforced": enforce_gates,
        "all_gates_passed": bool(sep_ok and (steer_ok if bc_items is not None else True)),
    }
    (out_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if enforce_gates and not summary["all_gates_passed"]:
        raise RuntimeError(
            f"Vector validation failed: {json.dumps(summary, indent=2)}"
        )
    return summary


# --- Steering gate v2 -------------------------------------------------------
# Coefficients are fractions of the mean last-token residual norm at L*
# (unit vectors x raw +/-2 were ~2% of ||h|| on Phi-3.5 and did not steer).
STEER_NORM_FRACS: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
# 5-point Spearman takes values {1.0, 0.9, ...}; scipy can return 0.8999999999999998.
SPEARMAN_TOL = 1e-9


def spearman_passes(rho: float | None, threshold: float, tol: float = SPEARMAN_TOL) -> bool:
    if rho is None or np.isnan(rho):
        return False
    return float(rho) >= threshold - tol


def keying_balanced_mean(scores: list[float], keyings: list[str]) -> float:
    """Mean of the +keyed mean and the -keyed mean (equal weight per keying)."""
    groups = [
        [s for s, k in zip(scores, keyings, strict=True) if k == key] for key in ("+", "-")
    ]
    means = [float(np.mean(g)) for g in groups if g]
    return float(np.mean(means)) if means else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    if len(set(np.round(y, 12))) < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).correlation)


def residual_norm_at_layer(loaded: LoadedModel, prompts: list[str], layer_index: int) -> float:
    """Mean L2 norm of the last-token residual after block `layer_index`."""
    norms = []
    for prompt in prompts:
        _, hidden = forward_with_hidden_states(loaded, prompt)
        norms.append(float(last_token_residual(hidden, layer_index + 1).float().norm()))
    return float(np.mean(norms))


def validate_steering(
    loaded: LoadedModel,
    vectors: dict[str, torch.Tensor],
    *,
    L_star: int,
    bc_items: list[dict[str, Any]],
    sr_items: list[dict[str, Any]] | None,
    assistant_prefill: str,
    sr_format: str,
    seed: int,
    norm_fracs: tuple[float, ...] | None = None,
    min_spearman: float = 0.9,
    min_option_mass: float = 0.80,
) -> tuple[bool, dict[str, Any], pd.DataFrame]:
    """Steering gate v2.

    Gating criterion per trait: Spearman(coefficient, mean BC high-trait probability)
    >= min_spearman, with mean BC option mass >= min_option_mass. BC is forced choice
    between high/low responses (order-averaged), so acquiescence cannot drive it.
    SR (both label orders, keying-balanced) is computed at the same coefficients as a
    secondary, non-gating report.
    """
    from trait_mtmm.measures.bc import (
        OPTIONS as BC_OPTIONS,
        bc_order_seed,
        build_bc_user_prompt,
        order_options,
        score_bc_item,
    )
    from trait_mtmm.measures.sr import SR_FORMATS, score_sr_item

    fracs = tuple(norm_fracs) if norm_fracs is not None else STEER_NORM_FRACS
    tok = loaded.tokenizer
    bc_ids = discover_option_token_ids(tok, list(BC_OPTIONS))
    indexed_bc = list(enumerate(bc_items))  # index = position in full list (pilot seeds)

    # ||h|| reference: unsteered BC prompts, first 4 items per trait, forward order.
    ref_prompts = []
    for trait in ("E", "A"):
        for i, item in [x for x in indexed_bc if x[1]["trait"] == trait][:4]:
            ordered = order_options(item, seed=bc_order_seed(seed, item, i), reverse=False)
            ref_prompts.append(
                format_chat(
                    tok,
                    user=build_bc_user_prompt(item, ordered),
                    assistant_prefill=assistant_prefill or None,
                )
            )
    h_norm = residual_norm_at_layer(loaded, ref_prompts, L_star)

    def steer_kwargs(trait: str, coef: float) -> dict[str, Any]:
        if abs(coef) < 1e-12:
            return {}
        return {
            "steer_layer": L_star,
            "steer_vector": vectors[trait],
            "steer_coefficient": float(coef),
        }

    report: dict[str, Any] = {
        "skipped": False,
        "version": 2,
        "criterion": "bc",
        "h_norm_at_L_star": h_norm,
        "norm_fracs": list(fracs),
        "min_spearman": min_spearman,
        "spearman_tol": SPEARMAN_TOL,
        "sr_format": sr_format,
        "assistant_prefill": assistant_prefill,
    }
    rows: list[dict[str, Any]] = []
    all_ok = True
    for trait in ("E", "A"):
        t_bc = [(i, it) for i, it in indexed_bc if it["trait"] == trait]
        scale = 1.0
        for _ in range(8):
            bc_rows = []
            for frac in fracs:
                coef = frac * scale * h_norm
                scores, masses = [], []
                for i, item in t_bc:
                    r = score_bc_item(
                        loaded,
                        item,
                        system=None,
                        assistant_prefill=assistant_prefill,
                        option_token_ids=bc_ids,
                        order_seed=bc_order_seed(seed, item, i),
                        **steer_kwargs(trait, coef),
                    )
                    scores.append(r["score"])
                    masses.append(r["option_mass"])
                bc_rows.append(
                    {
                        "trait": trait,
                        "measure": "bc",
                        "norm_frac": frac * scale,
                        "coefficient": coef,
                        "scale": scale,
                        "score": float(np.mean(scores)),
                        "mean_option_mass": float(np.mean(masses)),
                        "n_items": len(scores),
                    }
                )
            bc_mass = float(np.mean([r["mean_option_mass"] for r in bc_rows]))
            if bc_mass >= min_option_mass or scale < 0.01:
                break
            scale *= 0.5
        rows.extend(bc_rows)
        bc_rho = _spearman([r["coefficient"] for r in bc_rows], [r["score"] for r in bc_rows])
        bc_ok = spearman_passes(bc_rho, min_spearman)
        mass_ok = bc_mass >= min_option_mass
        trait_report: dict[str, Any] = {
            "bc_spearman": bc_rho,
            "bc_delta": bc_rows[-1]["score"] - bc_rows[0]["score"],
            "bc_mean_option_mass": bc_mass,
            "scale": scale,
            "passed": bool(bc_ok and mass_ok),
        }

        # Secondary (non-gating): SR, both label orders, keying-balanced.
        t_sr = [it for it in (sr_items or []) if it["trait"] == trait]
        if t_sr:
            fmt = SR_FORMATS[sr_format]
            sr_ids = discover_option_token_ids(tok, list(fmt["options"]))
            sr_rows = []
            for frac in fracs:
                coef = frac * scale * h_norm
                res = [
                    score_sr_item(
                        loaded,
                        item,
                        system=None,
                        assistant_prefill=assistant_prefill,
                        option_token_ids=sr_ids,
                        sr_format=sr_format,
                        **steer_kwargs(trait, coef),
                    )
                    for item in t_sr
                ]
                sr_rows.append(
                    {
                        "trait": trait,
                        "measure": "sr_balanced",
                        "norm_frac": frac * scale,
                        "coefficient": coef,
                        "scale": scale,
                        "score": keying_balanced_mean(
                            [r["score"] for r in res], [r["keying"] for r in res]
                        ),
                        "raw_agree": float(np.mean([r["score_before_keying"] for r in res])),
                        "mean_option_mass": float(np.mean([r["option_mass"] for r in res])),
                        "n_items": len(res),
                    }
                )
            rows.extend(sr_rows)
            coefs = [r["coefficient"] for r in sr_rows]
            trait_report.update(
                {
                    "sr_balanced_spearman": _spearman(coefs, [r["score"] for r in sr_rows]),
                    "sr_balanced_delta": sr_rows[-1]["score"] - sr_rows[0]["score"],
                    "sr_raw_agree_spearman": _spearman(coefs, [r["raw_agree"] for r in sr_rows]),
                    "sr_gating": False,
                }
            )
        report[trait] = trait_report
        all_ok = all_ok and trait_report["passed"]

    return all_ok, report, pd.DataFrame(rows)


def load_vectors(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
