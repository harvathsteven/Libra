"""CLI entrypoint: extract, extract-acts, pilot, diagnose, phase4, main, phase7, report, smoke, analyze, figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
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


def _score_personas(
    cfg: dict[str, Any],
    personas: list[dict[str, Any]],
    out: Path,
    *,
    L_star: int,
    vectors: dict[str, torch.Tensor],
    resume: bool,
) -> dict[str, pd.DataFrame]:
    """Score SR/BC/IP for each persona, writing per-persona parts as it goes.

    Parts live in ``out/parts`` and residuals in ``out/ip_residuals``. A persona
    is complete once ``parts/<id>.done.json`` exists. With ``resume=True``
    complete personas are skipped, but only if they were written at the same
    git commit; otherwise the run refuses to mix code versions.
    """
    from trait_mtmm.provenance import git_commit

    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    stim = stimuli_dir(cfg)
    parts = out / "parts"
    residual_dir = out / "ip_residuals"
    if not resume:
        for d in (parts, residual_dir):
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
    parts.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    sr_items = load_ipip(stim, max_items=limits.get("max_sr_items") if is_smoke else None)
    bc_items = load_bc(stim, max_items=limits.get("max_bc_items") if is_smoke else None)
    probes = load_questions(stim / "probe_questions.txt")
    if is_smoke:
        probes = probes[: limits["max_probes"]]
    prefill = cfg["scoring"].get("assistant_prefill") or ""
    sr_format = str(cfg["scoring"].get("sr_format", "letters"))

    todo = []
    for persona in personas:
        done = parts / f"{persona['persona_id']}.done.json"
        if resume and done.exists():
            prev = json.loads(done.read_text(encoding="utf-8")).get("git_commit")
            if prev != commit:
                raise SystemExit(
                    f"{done} was written at {prev}, current commit is {commit}; "
                    "refusing to mix code versions (clear the run directory to restart)"
                )
            continue
        todo.append(persona)
    print(f"Scoring {len(todo)} of {len(personas)} personas ({len(personas) - len(todo)} already done)")

    loaded = None
    for n, persona in enumerate(todo, 1):
        if loaded is None:
            loaded = load_model(
                cfg["model"]["id"],
                cfg["model"]["revision"],
                dtype=cfg["model"]["dtype"],
                device=cfg["model"]["device"],
            )
        pid = persona["persona_id"]
        sr = run_sr_for_persona(
            loaded, persona, sr_items, assistant_prefill=prefill, sr_format=sr_format
        )
        bc = run_bc_for_persona(
            loaded, persona, bc_items, assistant_prefill=prefill, base_seed=cfg["seed"]
        )
        ip, residuals = run_ip_for_persona(
            loaded,
            persona,
            probes,
            vectors=vectors,
            layer_hidden_index=L_star + 1,
            return_residuals=True,
        )
        sr.to_parquet(parts / f"{pid}.sr.parquet", index=False)
        bc.to_parquet(parts / f"{pid}.bc.parquet", index=False)
        ip.to_parquet(parts / f"{pid}.ip.parquet", index=False)
        torch.save({"persona_id": pid, "residuals": residuals}, residual_dir / f"{pid}.pt")
        write_json(parts / f"{pid}.done.json", {"persona_id": pid, "git_commit": commit})
        print(f"[{n}/{len(todo)}] {pid} done")

    def gather(kind: str) -> pd.DataFrame:
        return pd.concat(
            [pd.read_parquet(parts / f"{p['persona_id']}.{kind}.parquet") for p in personas],
            ignore_index=True,
        )

    sr_item, bc_item, ip_probe = gather("sr"), gather("bc"), gather("ip")
    sr_agg, bc_agg, ip_agg = aggregate_sr(sr_item), aggregate_bc(bc_item), aggregate_ip(ip_probe)
    wide = wide_condition_table(sr_agg, bc_agg, ip_agg)
    tables = {
        "sr_item": sr_item,
        "bc_item": bc_item,
        "ip_probe": ip_probe,
        "sr_condition": sr_agg,
        "bc_condition": bc_agg,
        "ip_condition": ip_agg,
        "condition_scores": wide,
    }
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return tables


def cmd_pilot(cfg: dict[str, Any]) -> Path:
    """Run SR/BC/IP on personas (dev, or smoke subset of test)."""
    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    out = results_dir(cfg) / ("pilot_smoke" if is_smoke else "pilot")
    out.mkdir(parents=True, exist_ok=True)
    L_star, vectors, extract_dir = _load_L_star_and_vectors(cfg)
    personas = load_personas(
        stimuli_dir(cfg),
        which="test" if is_smoke else "dev",
        max_personas=limits.get("max_personas") if is_smoke else None,
        include_baseline=False,
    )
    tables = _score_personas(cfg, personas, out, L_star=L_star, vectors=vectors, resume=False)
    summary = analysis_summary(tables["condition_scores"])
    write_json(
        out / "provenance.json",
        {
            **provenance_dict(cfg, stage="pilot"),
            "L_star": L_star,
            "trait_vectors_sha256": sha256_file(extract_dir / "trait_vectors.pt"),
            "n_personas": len(personas),
            "assistant_prefill": cfg["scoring"].get("assistant_prefill") or "",
            "analysis": summary,
        },
    )
    print(json.dumps({"pilot": {"n_personas": len(personas), "L_star": L_star, **summary}}, indent=2))
    return out


def main_run_quality(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Data-quality summary for the main run. Deliberately reports no outcome
    statistics (no scores, correlations or SDs): those wait for the Phase 7
    analysis code (PREREGISTRATION.md §5)."""
    drop_mass = float(cfg["scoring"]["drop_item_option_mass"])
    drop_frac = float(cfg["scoring"]["drop_condition_item_frac"])
    report: dict[str, Any] = {
        "n_conditions": int(tables["condition_scores"]["persona_id"].nunique()),
        "n_by_set": tables["condition_scores"]["persona_set"].value_counts().to_dict(),
        "item_drop_threshold": drop_mass,
        "condition_drop_frac": drop_frac,
    }
    for measure in ("sr", "bc"):
        items = tables[f"{measure}_item"]
        low = items["option_mass"] < drop_mass
        per_cond = (
            items.assign(low=low)
            .groupby(["persona_id", "trait"])["low"]
            .mean()
        )
        report[measure] = {
            "mean_option_mass": float(items["option_mass"].mean()),
            "min_option_mass": float(items["option_mass"].min()),
            "n_items_below_threshold": int(low.sum()),
            "n_condition_scores_excluded": int((per_cond > drop_frac).sum()),
            "excluded": [f"{p}:{t}" for (p, t), v in per_cond.items() if v > drop_frac],
        }
    ip = tables["ip_probe"]
    report["ip"] = {"n_nonfinite": int((~np.isfinite(ip["projection"].to_numpy(dtype=float))).sum())}
    # Every condition must have every item / probe (one row per item, both orders inside).
    complete = {}
    for name, key in (("sr_item", "item_id"), ("bc_item", "item_id"), ("ip_probe", "probe_index")):
        df = tables[name]
        per = df.groupby(["persona_id", "trait"])[key].nunique().reset_index()
        expected = df.groupby("trait")[key].nunique()
        n_conditions = df["persona_id"].nunique()
        complete[name] = bool(
            (per[key].to_numpy() == per["trait"].map(expected).to_numpy()).all()
            and len(per) == n_conditions * len(expected)
        )
    report["rows_complete"] = complete
    return report


def cmd_main(cfg: dict[str, Any]) -> Path:
    """Phase 6 main run: 96 test personas + baseline (PREREGISTRATION.md §5).

    Same SR/BC/IP code path as the dev pilot, residual caching on, pinned
    vectors required, resumable per persona.
    """
    seed_everything(cfg["seed"])
    is_smoke = "smoke" in cfg
    limits = _smoke_limits(cfg) if is_smoke else {}
    out = results_dir(cfg) / ("main_smoke" if is_smoke else "main")
    out.mkdir(parents=True, exist_ok=True)
    L_star, vectors, extract_dir = _load_L_star_and_vectors(cfg, require_pin=not is_smoke)
    if not is_smoke:
        blob_L = int(load_vectors(extract_dir / "trait_vectors.pt")["L_star"])
        if L_star != 17 or blob_L != 17:
            raise SystemExit(f"L* must be 17 (prereg-v2); config {L_star}, vectors {blob_L}")
    personas = load_personas(stimuli_dir(cfg), which="test", include_baseline=True)
    if is_smoke:
        personas = personas[: limits["max_personas"]] + [
            p for p in personas if p.get("set") == "baseline"
        ]
    elif len(personas) != 97:
        raise SystemExit(f"Expected 97 test conditions, found {len(personas)}")
    tables = _score_personas(cfg, personas, out, L_star=L_star, vectors=vectors, resume=True)
    quality = main_run_quality(tables, cfg)
    write_json(out / "main_quality.json", quality)
    write_json(
        out / "provenance.json",
        {
            **provenance_dict(cfg, stage="main"),
            "L_star": L_star,
            "trait_vectors_sha256": sha256_file(extract_dir / "trait_vectors.pt"),
            "n_personas": len(personas),
            "assistant_prefill": cfg["scoring"].get("assistant_prefill") or "",
        },
    )
    print(json.dumps({"main_quality": quality}, indent=2))
    print(f"Main run complete: {out}")
    return out


def _git_tracked_changes() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root(), text=True
    ).strip()


def cmd_phase7(cfg: dict[str, Any]) -> Path:
    """Phase 7 preregistered analysis of the main run (PREREGISTRATION.md §2–§3).

    Refuses to run with uncommitted changes to tracked files, so the analysis
    code that produced the results is always the recorded commit (prereg §5).
    """
    from trait_mtmm.figures_main import write_main_figures
    from trait_mtmm.mtmm import CORR_METHODS, SAMPLES, run_analysis

    if "smoke" in cfg:
        raise SystemExit("phase7 runs on the main-run results only")
    dirty = _git_tracked_changes()
    if dirty:
        raise SystemExit(f"Commit the analysis code first; tracked changes:\n{dirty}")
    L_star, _, extract_dir = _load_L_star_and_vectors(cfg, require_pin=True)
    main_dir = results_dir(cfg) / "main"
    main_prov = json.loads((main_dir / "provenance.json").read_text(encoding="utf-8"))
    pin = cfg["vectors"]["sha256"]
    if main_prov.get("trait_vectors_sha256") != pin or main_prov.get("is_smoke"):
        raise SystemExit("results/main was not produced with the pinned vectors")
    if L_star != 17 or int(main_prov.get("L_star", -1)) != 17:
        raise SystemExit("L* must be 17 (prereg-v2)")

    stim = stimuli_dir(cfg)
    personas = load_personas(stim, which="test", include_baseline=True)
    ipip = load_ipip(stim)
    bc = load_bc(stim)
    n_items = {
        "SR": {t: sum(r["trait"] == t for r in ipip) for t in ("E", "A")},
        "BC": {t: sum(r["trait"] == t for r in bc) for t in ("E", "A")},
    }
    a = cfg["analysis"]
    sc = cfg["scoring"]
    res = run_analysis(
        main_dir=main_dir,
        vectors_blob=load_vectors(extract_dir / "trait_vectors.pt"),
        extraction_acts=torch.load(extract_dir / "extraction_activations.pt", weights_only=False),
        personas=personas,
        n_items=n_items,
        L_star=L_star,
        B=int(a["bootstrap_n"]),
        seed=int(a["bootstrap_seed"]),
        n_null=int(a["null_vectors"]),
        alpha=float(a["alpha"]),
        drop_mass=float(sc["drop_item_option_mass"]),
        drop_frac=float(sc["drop_condition_item_frac"]),
    )

    out = results_dir(cfg) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(out / "hypothesis_components.csv", index=False)
    for name in SAMPLES:
        for m in CORR_METHODS:
            res["matrices"][name][m].to_csv(out / f"mtmm_{m}_{name}.csv")
        res["samples"][name].to_csv(out / f"listwise_{name}.csv", index=False)
    pd.DataFrame(res["manipulation_check"]).to_csv(out / "manipulation_check.csv", index=False)
    pd.DataFrame(res["disattenuated"]).to_csv(out / "disattenuated.csv", index=False)
    pd.DataFrame(res["acquiescence"]).to_csv(out / "acquiescence.csv", index=False)
    pd.DataFrame(res["layer_sweep"]).to_csv(out / "layer_sweep.csv", index=False)
    res["exclusion_log"].to_csv(out / "exclusion_log.csv", index=False)
    np.savez_compressed(out / "null_correlations.npz", **res["null_arrays"])
    write_json(
        out / "analysis.json",
        {
            "prereg": "prereg-v2",
            "sample_info": res["sample_info"],
            "exclusions": res["exclusions"],
            "checks": res["checks"],
            "verdicts": res["verdicts"],
            "outcome_patterns": res["outcome_patterns"],
            "reliability": res["reliability"],
            "baseline": res["baseline"],
        },
    )
    write_json(
        out / "provenance.json",
        {
            **provenance_dict(cfg, stage="phase7"),
            "main_run_git_commit": main_prov.get("git_commit"),
            "trait_vectors_sha256": sha256_file(extract_dir / "trait_vectors.pt"),
            "extraction_activations_sha256": sha256_file(extract_dir / "extraction_activations.pt"),
            "bootstrap_n": int(a["bootstrap_n"]),
            "bootstrap_seed": int(a["bootstrap_seed"]),
            "null_vectors": int(a["null_vectors"]),
        },
    )
    steer_p = extract_dir / "steering_curves.parquet"
    dp_p = extract_dir / "dprime_by_layer.parquet"
    figs = write_main_figures(
        res,
        fig_dir=figures_dir(cfg),
        L_star=L_star,
        steer=pd.read_parquet(steer_p) if steer_p.exists() else None,
        dprime=pd.read_parquet(dp_p) if dp_p.exists() else None,
    )
    for v in res["verdicts"]:
        if v["sample"] == "pooled" and v["corr_method"] == "pearson":
            print(f"{v['hypothesis']} {v['trait']}: {'holds' if v['holds'] else 'does not hold'}")
    print(f"Phase 7 analysis written to {out}; figures: {[str(p) for p in figs]}")
    return out


def cmd_report(cfg: dict[str, Any]) -> Path:
    """Phase 8: generate report tables (and exploratory checks) and fill REPORT.md blocks."""
    from trait_mtmm.report import build_tables, exploratory, fill_report

    if "smoke" in cfg:
        raise SystemExit("report runs on the main-run results only")
    L_star, _, extract_dir = _load_L_star_and_vectors(cfg, require_pin=True)
    analysis_dir = results_dir(cfg) / "analysis"
    listwise = pd.read_csv(analysis_dir / "listwise_pooled.csv")
    expl = exploratory(
        main_dir=results_dir(cfg) / "main",
        extract_dir=extract_dir,
        listwise=listwise,
        L_star=L_star,
        n_null=int(cfg["analysis"]["null_vectors"]),
    )
    write_json(analysis_dir / "exploratory.json", {**expl, "git_commit": provenance_dict(cfg, stage="report")["git_commit"]})
    tables = build_tables(analysis_dir, extract_dir, expl)
    (analysis_dir / "report_tables.md").write_text(
        "\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()) + "\n", encoding="utf-8"
    )
    report = repo_root() / "REPORT.md"
    used = fill_report(report, tables) if report.exists() else []
    print(json.dumps({"tables": list(tables), "filled_in_REPORT.md": used}, indent=2))
    return analysis_dir


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
            "phase7",
            "report",
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
        "phase7": lambda: cmd_phase7(cfg),
        "report": lambda: cmd_report(cfg),
        "analyze": lambda: cmd_analyze(cfg),
        "figures": lambda: cmd_figures(cfg),
        "smoke": lambda: cmd_smoke(cfg),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
