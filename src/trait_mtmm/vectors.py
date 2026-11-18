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
from trait_mtmm.scoring import discover_option_token_ids, score_likert
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


def extract_and_validate(
    loaded: LoadedModel,
    adjectives: dict[str, Any],
    extraction_questions: list[str],
    *,
    seed: int,
    n_triples: int = 6,
    out_dir: Path,
    sr_items: list[dict[str, Any]] | None = None,
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
    rng = np.random.default_rng(seed)
    n_layers = n_transformer_layers(loaded.model)

    # Build prompts per trait/pole
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        "E": {"high": [], "low": []},
        "A": {"high": [], "low": []},
    }
    trait_map = {"E": "extraversion", "A": "agreeableness"}

    for trait, name in trait_map.items():
        for pole in ("high", "low"):
            words = adjectives[name][pole]["extraction"]
            pole_seed = seed + (0 if trait == "E" else 100) + (0 if pole == "high" else 10)
            triples = adjective_triples(words, n=n_triples, seed=pole_seed)
            for triple in triples:
                for qi, q in enumerate(extraction_questions):
                    records[trait][pole].append(
                        {
                            "triple": triple,
                            "question": q,
                            "question_index": qi,
                            "system": extraction_system_prompt(*triple),
                        }
                    )

    # 80/20 split by adjective triple and by question
    train_acts: dict[str, dict[str, list[list[torch.Tensor]]]] = {
        t: {"high": [[] for _ in range(n_layers)], "low": [[] for _ in range(n_layers)]}
        for t in ("E", "A")
    }
    hold_acts: dict[str, dict[str, list[list[torch.Tensor]]]] = {
        t: {"high": [[] for _ in range(n_layers)], "low": [[] for _ in range(n_layers)]}
        for t in ("E", "A")
    }

    for trait in ("E", "A"):
        for pole in ("high", "low"):
            prompts = records[trait][pole]
            triples = sorted({p["triple"] for p in prompts})
            questions = sorted({p["question"] for p in prompts})
            rng.shuffle(triples)
            rng.shuffle(questions)
            n_t_train = max(1, int(0.8 * len(triples)))
            n_q_train = max(1, int(0.8 * len(questions)))
            train_triples = set(triples[:n_t_train])
            train_questions = set(questions[:n_q_train])

            for p in prompts:
                is_train = p["triple"] in train_triples and p["question"] in train_questions
                # If both sets empty edge case, treat as train
                layers_list = collect_residuals(
                    loaded, system=p["system"], questions=[p["question"]]
                )[0]
                bucket = train_acts if is_train else hold_acts
                # If holdout empty, keep a few for holdout forcibly
                for layer_i, residual in enumerate(layers_list):
                    bucket[trait][pole][layer_i].append(residual)

            # Ensure holdout non-empty per pole/layer
            for layer_i in range(n_layers):
                if not hold_acts[trait][pole][layer_i] and train_acts[trait][pole][layer_i]:
                    hold_acts[trait][pole][layer_i].append(
                        train_acts[trait][pole][layer_i].pop()
                    )

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

    # Steering validation on baseline (no persona), using SR if items provided
    steer_report: dict[str, Any] = {"skipped": sr_items is None}
    steer_ok = True
    if sr_items is not None and len(sr_items) > 0:
        coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]
        # Scale coefficients so option mass stays high; start from these and shrink if needed
        scale = 1.0
        from trait_mtmm.measures.sr import SR_FORMATS, build_sr_user_prompt

        fmt = SR_FORMATS[sr_format]
        option_ids = discover_option_token_ids(
            loaded.tokenizer, list(fmt["options"])
        )
        steer_report["sr_format"] = sr_format
        steer_report["assistant_prefill"] = assistant_prefill

        def _sr_mass_unsteered() -> float:
            """Baseline option mass with no steering (prompt-format check)."""
            masses = []
            for item in sr_items[: min(4, len(sr_items))]:
                user = build_sr_user_prompt(
                    item["text"], scale_reversed=False, sr_format=sr_format
                )
                prompt = format_chat(
                    loaded.tokenizer,
                    system=None,
                    user=user,
                    assistant_prefill=assistant_prefill or None,
                )
                logits, _ = forward_with_hidden_states(loaded, prompt)
                result = score_likert(
                    next_token_logits(logits),
                    option_ids,
                    options=fmt["options"],
                    values=fmt["values"],
                )
                masses.append(result.option_mass)
            return float(np.mean(masses)) if masses else 0.0

        baseline_mass = _sr_mass_unsteered()
        steer_report["baseline_option_mass"] = baseline_mass
        if baseline_mass < min_option_mass:
            steer_report["failure_reason"] = (
                f"SR prompt format yields option mass {baseline_mass:.4f} "
                f"< {min_option_mass} even without steering "
                f"(sr_format={sr_format!r}, prefill={assistant_prefill!r}). "
                "Change scoring.sr_format / assistant_prefill (dev only) and re-run extract."
            )
            steer_ok = False
        else:

            def run_curve(trait: str, scale_: float) -> pd.DataFrame:
                rows = []
                vec = vectors[trait]
                for coef in coefficients:
                    item_scores = []
                    masses = []
                    for item in sr_items:
                        if item["trait"] != trait:
                            continue
                        user = build_sr_user_prompt(
                            item["text"],
                            scale_reversed=False,
                            sr_format=sr_format,
                        )
                        prompt = format_chat(
                            loaded.tokenizer,
                            system=None,
                            user=user,
                            assistant_prefill=assistant_prefill or None,
                        )
                        c = float(coef * scale_)
                        if abs(c) < 1e-12:
                            logits, _ = forward_with_hidden_states(loaded, prompt)
                        else:
                            logits, _ = forward_with_hidden_states(
                                loaded,
                                prompt,
                                steer_layer=L_star,
                                steer_vector=vec,
                                steer_coefficient=c,
                            )
                        result = score_likert(
                            next_token_logits(logits),
                            option_ids,
                            options=fmt["options"],
                            values=fmt["values"],
                        )
                        assert result.expected_value is not None
                        keyed = apply_reverse_keying(
                            result.expected_value, reverse=(item["keying"] == "-")
                        )
                        item_scores.append(keyed)
                        masses.append(result.option_mass)
                    rows.append(
                        {
                            "trait": trait,
                            "coefficient": coef * scale_,
                            "raw_coefficient": coef,
                            "scale": scale_,
                            "sr_score": float(np.mean(item_scores))
                            if item_scores
                            else float("nan"),
                            "mean_option_mass": float(np.mean(masses))
                            if masses
                            else float("nan"),
                        }
                    )
                return pd.DataFrame(rows)

            steer_frames = []
            for trait in ("E", "A"):
                # Shrink scale until mean option mass across curve >= threshold
                scale = 1.0
                for _ in range(8):
                    curve = run_curve(trait, scale)
                    if (
                        curve["mean_option_mass"].mean() >= min_option_mass
                        or scale < 0.01
                    ):
                        break
                    scale *= 0.5
                steer_frames.append(curve)
                spearman = stats.spearmanr(
                    curve["coefficient"], curve["sr_score"]
                ).correlation
                if spearman is None or np.isnan(spearman):
                    spearman = 0.0
                trait_ok = float(spearman) >= min_steer_spearman
                steer_ok = steer_ok and trait_ok
                steer_report[trait] = {
                    "spearman": float(spearman),
                    "scale": scale,
                    "mean_option_mass": float(curve["mean_option_mass"].mean()),
                    "passed": trait_ok,
                }
            steer_df = pd.concat(steer_frames, ignore_index=True)
            steer_df.to_parquet(out_dir / "steering_curves.parquet", index=False)

    summary = {
        "L_star": L_star,
        "n_layers": n_layers,
        "dprime_at_L_star": {t: dprimes[t][L_star] for t in ("E", "A")},
        "cosine_EA_at_L_star": cosine_EA,
        "separation_passed": sep_ok,
        "steering_passed": steer_ok if sr_items is not None else None,
        "steering": steer_report,
        "gates_enforced": enforce_gates,
        "all_gates_passed": bool(sep_ok and (steer_ok if sr_items is not None else True)),
    }
    (out_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if enforce_gates and not summary["all_gates_passed"]:
        raise RuntimeError(
            f"Vector validation failed: {json.dumps(summary, indent=2)}"
        )
    return summary


def load_vectors(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
