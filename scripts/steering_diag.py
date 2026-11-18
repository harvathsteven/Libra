"""Phase 4 steering diagnostic (dev only; not a gate, not a prompt-format attempt).

Explains the Attempt 2 steering failure (E rho=0.8999..., A rho=-1.0) by scoring
every SR item under steering in BOTH label orders and splitting by keying:

  +/-          keyed SR mean over forward / reverse-keyed items
  balanced     mean of the + and - columns (cancels uniform acquiescence)
  gate_metric  what the failing gate measured (standard order only, all items)
  raw_agree    unkeyed mean rating (acquiescence: "Accurate" to everything)
  mass         mean option mass

Two coefficient sets: "raw" = the gate's [-2..2] on a unit vector, and
"x||h||" = fractions of the mean residual norm at L*.

If no vectors exist (fresh clone), re-extracts them first (same seed -> same L*),
skipping the steering gate.

Usage:  python scripts/steering_diag.py --config configs/default.yaml
Writes: results/extract/steering_diag.parquet, steering_diag_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trait_mtmm.measures.sr import SR_FORMATS, build_sr_user_prompt  # noqa: E402
from trait_mtmm.model import (  # noqa: E402
    format_chat,
    forward_with_hidden_states,
    load_model,
    next_token_logits,
    seed_everything,
)
from trait_mtmm.provenance import load_config, results_dir, stimuli_dir  # noqa: E402
from trait_mtmm.scoring import (  # noqa: E402
    apply_reverse_keying,
    discover_option_token_ids,
    map_reversed_scale_score,
    score_likert,
)
from trait_mtmm.stimuli_io import load_adjectives, load_ipip, load_questions  # noqa: E402
from trait_mtmm.vectors import extract_and_validate, load_vectors  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--norm-fracs", default="0.1,0.2", help="positive fractions of ||h||")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    stim = stimuli_dir(cfg)
    out = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    out.mkdir(parents=True, exist_ok=True)
    sr_format = str(cfg["scoring"].get("sr_format", "letters"))
    prefill = cfg["scoring"].get("assistant_prefill") or None

    m = cfg["model"]
    loaded = load_model(m["id"], m["revision"], dtype=m["dtype"], device=m["device"])
    tok = loaded.tokenizer

    pt = out / "trait_vectors.pt"
    if not pt.exists():
        print(f"{pt} missing -> re-extracting vectors (steering gate skipped)")
        questions = load_questions(stim / "extraction_questions.txt")
        n_triples = 6
        if is_smoke:
            s = cfg["smoke"]
            questions = questions[: int(s.get("max_extraction_questions", 2))]
            n_triples = int(s.get("n_triples", 2))
        summary = extract_and_validate(
            loaded,
            load_adjectives(stim),
            questions,
            seed=cfg["seed"],
            n_triples=n_triples,
            out_dir=out,
            sr_items=None,
            enforce_gates=False,
        )
        print("extract:", {k: summary[k] for k in ("L_star", "dprime_at_L_star", "cosine_EA_at_L_star")})

    saved = load_vectors(pt)
    L = int(saved["L_star"])
    items = load_ipip(stim)
    fmt = SR_FORMATS[sr_format]
    opt_ids = discover_option_token_ids(tok, list(fmt["options"]))

    def prompt(it: dict, rev: bool) -> str:
        user = build_sr_user_prompt(it["text"], scale_reversed=rev, sr_format=sr_format)
        return format_chat(tok, user=user, assistant_prefill=prefill)

    norms = [
        forward_with_hidden_states(loaded, prompt(it, False))[1][L + 1][0, -1].float().norm().item()
        for it in items[:8]
    ]
    h_norm = float(np.mean(norms))
    print(f"L*={L}  mean ||h|| at last token = {h_norm:.1f}  -> coef 2 is {2 / h_norm:.2%} of residual norm")

    fracs = [float(x) for x in args.norm_fracs.split(",")]
    frac_set = sorted({0.0, *fracs, *(-f for f in fracs)})
    coef_sets = {
        "raw": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "x||h||": [f * h_norm for f in frac_set],
    }

    rows = []
    for trait in ("E", "A"):
        vec = saved["vectors"][trait].to(loaded.device)
        trait_items = [x for x in items if x["trait"] == trait]
        for label, coefs in coef_sets.items():
            for c in coefs:
                for it in trait_items:
                    for rev in (False, True):
                        kw = (
                            dict(steer_layer=L, steer_vector=vec, steer_coefficient=float(c))
                            if c
                            else {}
                        )
                        logits, _ = forward_with_hidden_states(loaded, prompt(it, rev), **kw)
                        r = score_likert(
                            next_token_logits(logits),
                            opt_ids,
                            options=fmt["options"],
                            values=fmt["values"],
                        )
                        raw = map_reversed_scale_score(r.expected_value, scale_reversed=rev)
                        rows.append(
                            dict(
                                trait=trait,
                                set=label,
                                coef=round(float(c), 3),
                                item_id=it["item_id"],
                                keying=it["keying"],
                                rev=rev,
                                raw=raw,
                                keyed=apply_reverse_keying(raw, reverse=it["keying"] == "-"),
                                mass=r.option_mass,
                            )
                        )
            print(f"done {trait} {label}")

    df = pd.DataFrame(rows)
    g = ["trait", "set", "coef"]
    t = df.pivot_table(index=g, columns="keying", values="keyed", aggfunc="mean")
    t["balanced"] = t[["+", "-"]].mean(axis=1)
    t["gate_metric"] = df[~df.rev].groupby(g).keyed.mean()
    t["raw_agree"] = df.groupby(g).raw.mean()
    t["mass"] = df.groupby(g).mass.mean()

    df.to_parquet(out / "steering_diag.parquet", index=False)
    t.round(4).to_csv(out / "steering_diag_table.csv")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print(f"\nWrote {out / 'steering_diag.parquet'} and {out / 'steering_diag_table.csv'}")


if __name__ == "__main__":
    main()
