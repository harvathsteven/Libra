"""CLI entrypoint: extract, extract-acts, pilot, diagnose, phase4, main, smoke, analyze, figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from trait_mtmm.analysis import analysis_summary, mtmm_correlation, wide_condition_table
from trait_mtmm.config_utils import set_L_star
from trait_mtmm.diagnose import diagnose_pilot
from trait_mtmm.figures import write_smoke_figures
from trait_mtmm.measures.bc import aggregate_bc, run_bc_for_persona
from trait_mtmm.measures.ip import aggregate_ip, run_ip_for_persona
from trait_mtmm.measures.sr import aggregate_sr, run_sr_for_persona
from trait_mtmm.model import load_model, seed_everything
from trait_mtmm.provenance import (
    figures_dir,
    load_config,
    provenance_dict,
    repo_root,
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
    bc_items = load_bc(
        stim, max_items=limits.get("max_bc_items") if is_smoke else None
    )

    summary = extract_and_validate(
        loaded,
        adjectives,
        questions,
        seed=cfg["seed"],
        n_triples=n_triples,
        out_dir=out,
        sr_items=sr_items,
        bc_items=bc_items,
        assistant_prefill=cfg["scoring"].get("assistant_prefill") or "",
        sr_format=str(cfg["scoring"].get("sr_format", "letters")),
        enforce_gates=enforce_gates,
    )
    vec_sha = sha256_file(out / "trait_vectors.pt")
    summary["trait_vectors_sha256"] = vec_sha
    write_json(
        out / "provenance.json",
        {**provenance_dict(cfg, stage="extract"), "summary": summary},
    )
    print(f"trait_vectors.pt sha256 {vec_sha}")
    # Freeze L* into the active config when gates pass (Phase 4).
    if summary.get("all_gates_passed") and not is_smoke:
        cfg_path = Path(cfg["_config_path"])
        set_L_star(cfg_path, int(summary["L_star"]))
        print(f"Wrote layer.L_star={summary['L_star']} to {cfg_path}")
    print(json.dumps({"extract": summary}, indent=2))
    return summary


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_extract_acts(cfg: dict[str, Any]) -> dict[str, Any]:
    """Re-run extraction forward passes and save activations only.

    For the existing Phase 4 extract: writes `extraction_activations.pt` and a
    check that vectors recomputed from it match the pinned `trait_vectors.pt`.
    Does not overwrite vectors, rerun the gates, or change the config.
    """
    from trait_mtmm.vectors import (
        collect_extraction_activations,
        save_extraction_activations,
        vectors_from_activations,
    )

    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    stim = stimuli_dir(cfg)
    out = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    L_star, _, _ = _load_L_star_and_vectors(cfg)  # also verifies the pin
    blob = load_vectors(out / "trait_vectors.pt")

    loaded = load_model(
        cfg["model"]["id"],
        cfg["model"]["revision"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )
    questions = load_questions(stim / "extraction_questions.txt")
    if is_smoke:
        questions = questions[: limits["max_extraction_questions"]]
    n_triples = limits["n_triples"] if is_smoke else 6
    acts = collect_extraction_activations(
        loaded, load_adjectives(stim), questions, seed=cfg["seed"], n_triples=n_triples
    )
    save_extraction_activations(acts, out / "extraction_activations.pt")
    recomputed = vectors_from_activations(acts)
    cos_by_layer = {
        t: [
            float(torch.dot(recomputed[t][i], blob["vectors_by_layer"][t][i].float()).item())
            for i in range(len(recomputed[t]))
        ]
        for t in ("E", "A")
    }
    report = {
        "L_star": L_star,
        "cosine_at_L_star": {t: cos_by_layer[t][L_star] for t in ("E", "A")},
        "min_cosine_any_layer": {t: min(cos_by_layer[t]) for t in ("E", "A")},
        "matches_pinned": all(cos_by_layer[t][L_star] >= 0.999 for t in ("E", "A")),
        "cosine_by_layer": cos_by_layer,
    }
    write_json(
        out / "extraction_activations_check.json",
        {**provenance_dict(cfg, stage="extract-acts"), **report},
    )
    print(json.dumps({"extract_acts": {k: v for k, v in report.items() if k != "cosine_by_layer"}}, indent=2))
    return report


def _load_L_star_and_vectors(
    cfg: dict[str, Any], *, require_pin: bool = False
) -> tuple[int, dict[str, torch.Tensor], Path]:
    """Load trait vectors; verify `vectors.sha256` from the config when set.

    `require_pin=True` (main run) refuses to run without a pinned hash.
    """
    is_smoke = "smoke" in cfg
    extract_dir = results_dir(cfg) / ("extract_smoke" if is_smoke else "extract")
    pt = extract_dir / "trait_vectors.pt"
    if not pt.exists():
        raise FileNotFoundError(f"Missing {pt}; run extract first")
    pinned = (cfg.get("vectors") or {}).get("sha256")
    if require_pin and not pinned:
        raise SystemExit("vectors.sha256 is not set in the config; pin the vectors first")
    if pinned:
        actual = sha256_file(pt)
        if actual != pinned:
            raise SystemExit(
                f"{pt} sha256 {actual} does not match pinned vectors.sha256 {pinned}"
            )
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

    prefill = cfg["scoring"].get("assistant_prefill") or ""
    sr_format = str(cfg["scoring"].get("sr_format", "letters"))
    residual_dir = out / "ip_residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    sr_parts, bc_parts, ip_parts = [], [], []
    for persona in personas:
        sr_parts.append(
            run_sr_for_persona(
                loaded,
                persona,
                sr_items,
                assistant_prefill=prefill,
                sr_format=sr_format,
            )
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
        ip_df, residuals = run_ip_for_persona(
            loaded,
            persona,
            probes,
            vectors=vectors,
            layer_hidden_index=layer_hidden_index,
            return_residuals=True,
        )
        ip_parts.append(ip_df)
        torch.save(
            {"persona_id": persona["persona_id"], "residuals": residuals},
            residual_dir / f"{persona['persona_id']}.pt",
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
            "trait_vectors_sha256": sha256_file(
                results_dir(cfg) / ("extract_smoke" if is_smoke else "extract") / "trait_vectors.pt"
            ),
            "n_personas": len(personas),
            "assistant_prefill": prefill,
            "analysis": summary,
        },
    )
    print(json.dumps({"pilot": {"n_personas": len(personas), "L_star": L_star, **summary}}, indent=2))
    return out


def cmd_diagnose(cfg: dict[str, Any]) -> dict[str, Any]:
    is_smoke = "smoke" in cfg
    pilot_dir = results_dir(cfg) / ("pilot_smoke" if is_smoke else "pilot")
    report = diagnose_pilot(
        pilot_dir,
        min_option_mass=float(cfg["scoring"]["min_option_mass_dev"]),
    )
    print(json.dumps({"diagnose": report}, indent=2))
    if not report["phase4_acceptance_passed"]:
        print(
            "Phase 4 diagnostics FAILED. Adjust scoring.assistant_prefill in the "
            "config (dev only), re-run pilot, and log the change in DECISIONS.md. "
            "Stop after two reasonable prompt-format attempts."
        )
    return report


def cmd_phase4(cfg: dict[str, Any]) -> None:
    """GPU Phase 4: extract (gates on) → pilot (dev) → diagnose."""
    if "smoke" in cfg:
        raise SystemExit("phase4 requires configs/default.yaml (not smoke)")
    summary = cmd_extract(cfg, enforce_gates=True)
    if not summary.get("all_gates_passed"):
        raise SystemExit("Extract validation failed; not running pilot.")
    # Reload config in case L* was written
    cfg = load_config(cfg["_config_path"])
    cmd_pilot(cfg)
    report = cmd_diagnose(cfg)
    if not report["phase4_acceptance_passed"]:
        raise SystemExit(1)
    print("Phase 4 acceptance passed.")


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

    paths = write_smoke_figures(
        figures=fig_dir,
        corr=corr,
        dprime_df=dprime_df,
        steer_df=steer_df,
        wide=wide,
        L_star=L_star,
    )
    # Rename smoke_* only for smoke; for phase4 keep names under figures/phase4_*
    if not is_smoke:
        renamed = []
        for p in paths:
            new = fig_dir / p.name.replace("smoke_", "phase4_")
            p.replace(new)
            renamed.append(new)
        paths = renamed
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
        "main run is Phase 6 (after preregistration). Use phase4/pilot until then."
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
        choices=[
            "extract",
            "extract-acts",
            "pilot",
            "diagnose",
            "phase4",
            "main",
            "analyze",
            "figures",
            "smoke",
        ],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "extract": lambda: cmd_extract(cfg),
        "extract-acts": lambda: cmd_extract_acts(cfg),
        "pilot": lambda: cmd_pilot(cfg),
        "diagnose": lambda: cmd_diagnose(cfg),
        "phase4": lambda: cmd_phase4(cfg),
        "main": lambda: cmd_main(cfg),
        "analyze": lambda: cmd_analyze(cfg),
        "figures": lambda: cmd_figures(cfg),
        "smoke": lambda: cmd_smoke(cfg),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
