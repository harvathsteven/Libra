"""CLI entrypoint: extract, pilot, main, smoke, analyze, figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from trait_mtmm.analysis import analysis_summary, mtmm_correlation, wide_condition_table
from trait_mtmm.figures import write_smoke_figures
from trait_mtmm.measures.bc import aggregate_bc, run_bc_for_persona
from trait_mtmm.measures.ip import aggregate_ip, run_ip_for_persona
from trait_mtmm.measures.sr import aggregate_sr, run_sr_for_persona
from trait_mtmm.model import load_model, seed_everything
from trait_mtmm.provenance import (
    figures_dir,
    load_config,
    provenance_dict,
    results_dir,
    stimuli_dir,
    write_json,
)
from trait_mtmm.stimuli_io import (
    load_adjectives,
    load_bc,
    load_ipip,
    load_personas,
    load_questions,
)
from trait_mtmm.vectors import extract_and_validate, load_vectors


def _smoke_limits(cfg: dict[str, Any]) -> dict[str, int]:
    s = cfg.get("smoke") or {}
    return {
        "max_sr_items": int(s.get("max_sr_items", 2)),
        "max_bc_items": int(s.get("max_bc_items", 2)),
        "max_probes": int(s.get("max_probes", 2)),
        "max_personas": int(s.get("max_personas", 2)),
        "n_triples": int(s.get("n_triples", 2)),
        "max_extraction_questions": int(s.get("max_extraction_questions", 2)),
    }


def cmd_extract(cfg: dict[str, Any], *, enforce_gates: bool | None = None) -> dict[str, Any]:
    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    if enforce_gates is None:
        enforce_gates = not is_smoke

    stim = stimuli_dir(cfg)
    out = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    out.mkdir(parents=True, exist_ok=True)

    loaded = load_model(
        cfg["model"]["id"],
        cfg["model"]["revision"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )
    adjectives = load_adjectives(stim)
    questions = load_questions(stim / "extraction_questions.txt")
    if is_smoke:
        questions = questions[: limits["max_extraction_questions"]]
        n_triples = limits["n_triples"]
    else:
        n_triples = 6

    sr_items = load_ipip(
        stim, max_items=limits.get("max_sr_items") if is_smoke else None
    )
    # Only E+A items already; for steering use items matching each trait

    summary = extract_and_validate(
        loaded,
        adjectives,
        questions,
        seed=cfg["seed"],
        n_triples=n_triples,
        out_dir=out,
        sr_items=sr_items,
        assistant_prefill=cfg["scoring"]["assistant_prefill"],
        enforce_gates=enforce_gates,
    )
    # Persist L* into a sidecar for later stages (config file unchanged)
    write_json(
        out / "provenance.json",
        {**provenance_dict(cfg, stage="extract"), "summary": summary},
    )
    print(json.dumps({"extract": summary}, indent=2))
    return summary


def _load_L_star_and_vectors(cfg: dict[str, Any]) -> tuple[int, dict[str, torch.Tensor], Path]:
    is_smoke = "smoke" in cfg
    extract_dir = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    pt = extract_dir / "trait_vectors.pt"
    if not pt.exists():
        raise FileNotFoundError(f"Missing {pt}; run extract first")
    blob = load_vectors(pt)
    L_star = int(blob["L_star"])
    if cfg.get("layer", {}).get("L_star") is not None:
        L_star = int(cfg["layer"]["L_star"])
    vectors = {k: v for k, v in blob["vectors"].items()}
    return L_star, vectors, extract_dir


def cmd_pilot(cfg: dict[str, Any]) -> Path:
    """Run SR/BC/IP on personas (dev, or smoke subset of test)."""
    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    stim = stimuli_dir(cfg)
    out = results_dir(cfg) / ("pilot_smoke" if is_smoke else "pilot")
    out.mkdir(parents=True, exist_ok=True)

    loaded = load_model(
        cfg["model"]["id"],
        cfg["model"]["revision"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )
    L_star, vectors, _ = _load_L_star_and_vectors(cfg)
    # hidden_states index = block_index + 1
    layer_hidden_index = L_star + 1

    which = "test" if is_smoke else "dev"
    personas = load_personas(
        stim,
        which=which,
        max_personas=limits.get("max_personas") if is_smoke else None,
        include_baseline=False,
    )
    sr_items = load_ipip(
        stim, max_items=limits.get("max_sr_items") if is_smoke else None
    )
    bc_items = load_bc(
        stim, max_items=limits.get("max_bc_items") if is_smoke else None
    )
    probes = load_questions(stim / "probe_questions.txt")
    if is_smoke:
        probes = probes[: limits["max_probes"]]

    prefill = cfg["scoring"]["assistant_prefill"]
    sr_parts, bc_parts, ip_parts = [], [], []
    for persona in personas:
        sr_parts.append(
            run_sr_for_persona(loaded, persona, sr_items, assistant_prefill=prefill)
        )
        bc_parts.append(
            run_bc_for_persona(
                loaded,
                persona,
                bc_items,
                assistant_prefill=prefill,
                base_seed=cfg["seed"],
            )
        )
        ip_parts.append(
            run_ip_for_persona(
                loaded,
                persona,
                probes,
                vectors=vectors,
                layer_hidden_index=layer_hidden_index,
            )
        )

    sr_item = pd.concat(sr_parts, ignore_index=True)
    bc_item = pd.concat(bc_parts, ignore_index=True)
    ip_probe = pd.concat(ip_parts, ignore_index=True)
    sr_agg = aggregate_sr(sr_item)
    bc_agg = aggregate_bc(bc_item)
    ip_agg = aggregate_ip(ip_probe)
    wide = wide_condition_table(sr_agg, bc_agg, ip_agg)

    sr_item.to_parquet(out / "sr_item.parquet", index=False)
    bc_item.to_parquet(out / "bc_item.parquet", index=False)
    ip_probe.to_parquet(out / "ip_probe.parquet", index=False)
    sr_agg.to_parquet(out / "sr_condition.parquet", index=False)
    bc_agg.to_parquet(out / "bc_condition.parquet", index=False)
    ip_agg.to_parquet(out / "ip_condition.parquet", index=False)
    wide.to_parquet(out / "condition_scores.parquet", index=False)

    summary = analysis_summary(wide)
    write_json(
        out / "provenance.json",
        {
            **provenance_dict(cfg, stage="pilot"),
            "L_star": L_star,
            "n_personas": len(personas),
            "analysis": summary,
        },
    )
    print(json.dumps({"pilot": {"n_personas": len(personas), "L_star": L_star, **summary}}, indent=2))
    return out


def cmd_figures(cfg: dict[str, Any]) -> list[Path]:
    is_smoke = "smoke" in cfg
    pilot_dir = results_dir(cfg) / ("pilot_smoke" if is_smoke else "pilot")
    extract_dir = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    fig_dir = figures_dir(cfg)

    wide = pd.read_parquet(pilot_dir / "condition_scores.parquet")
    corr = mtmm_correlation(wide)
    corr.to_csv(pilot_dir / "mtmm_pearson.csv")

    dprime_df = None
    steer_df = None
    L_star = None
    dp_path = extract_dir / "dprime_by_layer.parquet"
    if dp_path.exists():
        dprime_df = pd.read_parquet(dp_path)
    st_path = extract_dir / "steering_curves.parquet"
    if st_path.exists():
        steer_df = pd.read_parquet(st_path)
    summary_path = extract_dir / "extraction_summary.json"
    if summary_path.exists():
        L_star = json.loads(summary_path.read_text(encoding="utf-8")).get("L_star")

    if is_smoke:
        paths = write_smoke_figures(
            figures=fig_dir,
            corr=corr,
            dprime_df=dprime_df,
            steer_df=steer_df,
            wide=wide,
            L_star=L_star,
        )
    else:
        paths = write_smoke_figures(
            figures=fig_dir,
            corr=corr,
            dprime_df=dprime_df,
            steer_df=steer_df,
            wide=wide,
            L_star=L_star,
        )
    print(json.dumps({"figures": [str(p) for p in paths]}, indent=2))
    return paths


def cmd_smoke(cfg: dict[str, Any]) -> None:
    """End-to-end Phase 3 smoke: extract → pilot → figures."""
    if "smoke" not in cfg:
        raise SystemExit("smoke command requires configs/smoke.yaml")
    cmd_extract(cfg, enforce_gates=False)
    cmd_pilot(cfg)
    cmd_figures(cfg)
    out = results_dir(cfg) / "pilot_smoke" / "condition_scores.parquet"
    print(f"Smoke complete. Condition table: {out}")


def cmd_main(cfg: dict[str, Any]) -> None:
    raise SystemExit(
        "main run is Phase 6 (after preregistration). Use pilot/smoke until then."
    )


def cmd_analyze(cfg: dict[str, Any]) -> None:
    is_smoke = "smoke" in cfg
    pilot_dir = results_dir(cfg) / ("pilot_smoke" if is_smoke else "pilot")
    wide = pd.read_parquet(pilot_dir / "condition_scores.parquet")
    summary = analysis_summary(wide)
    write_json(pilot_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trait-mtmm")
    parser.add_argument(
        "command",
        choices=["extract", "pilot", "main", "analyze", "figures", "smoke"],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "extract": lambda: cmd_extract(cfg),
        "pilot": lambda: cmd_pilot(cfg),
        "main": lambda: cmd_main(cfg),
        "analyze": lambda: cmd_analyze(cfg),
        "figures": lambda: cmd_figures(cfg),
        "smoke": lambda: cmd_smoke(cfg),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
