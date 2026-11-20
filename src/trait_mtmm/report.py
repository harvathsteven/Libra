"""Phase 8: report tables generated from Phase 7 outputs, plus labeled exploratory checks.

`trait-mtmm report` reads `results/analysis/` (and `results/main`, `results/extract`),
writes `results/analysis/report_tables.md` and `exploratory.json`, and replaces each
`<!-- BEGIN:name -->` … `<!-- END:name -->` block in REPORT.md with the generated
table, so every number in a table comes from committed code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from trait_mtmm.mtmm import (
    SAMPLES,
    SHUFFLE_SEEDS,
    TRAITS,
    VARS,
    load_mbar,
    point_corr,
    shuffled_null_vectors,
    train_activations,
    unit_vectors_by_layer,
)


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")


def _p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"= {x:.3f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{_f(lo)}, {_f(hi)}]"


def _yes(b: bool) -> str:
    return "pass" if b else "fail"


def _cell(x: Any) -> str:
    return str(x).replace("|", "\\|")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(_cell(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- exploratory


def exploratory(
    *, main_dir: Path, extract_dir: Path, listwise: pd.DataFrame, L_star: int, n_null: int
) -> dict[str, Any]:
    """EXPLORATORY (not preregistered): diagnostics for the H4 nulls."""
    blob = torch.load(extract_dir / "trait_vectors.pt", weights_only=False)
    vbl = unit_vectors_by_layer(blob)
    acts = torch.load(extract_dir / "extraction_activations.pt", weights_only=False)
    out: dict[str, Any] = {"label": "exploratory; not preregistered"}

    align = {}
    for t in TRAITS:
        X, labels = train_activations(acts[t], L_star)
        V = shuffled_null_vectors(X, labels, n_null, SHUFFLE_SEEDS[t])
        c = np.abs(V @ vbl[t][L_star])
        align[t] = {
            "mean_abs_cos_with_trait_vector": float(c.mean()),
            "frac_abs_cos_gt_0.5": float((c > 0.5).mean()),
        }
    out["shuffled_null_alignment"] = align

    mbar = load_mbar(main_dir / "ip_residuals", listwise["persona_id"])
    M = np.stack([mbar[p][L_star] for p in listwise["persona_id"]])
    Mc = M - M.mean(axis=0)
    U, S, _ = np.linalg.svd(Mc, full_matrices=False)
    share = S**2 / (S**2).sum()
    pcs = U[:, :3] * S[:3]
    is_implicit = (listwise["persona_set"] == "implicit").to_numpy(float)
    pc_corr = {}
    for k in range(3):
        pc_corr[f"PC{k + 1}"] = {
            **{v: point_corr(pcs[:, k], listwise[v].to_numpy(float), "pearson") for v in VARS},
            "is_implicit": point_corr(pcs[:, k], is_implicit, "pearson"),
        }
    out["residual_pca_at_L_star"] = {
        "variance_share_pc1_3": [float(x) for x in share[:3]],
        "variance_share_top10_cumulative": float(share[:10].sum()),
        "pc_correlations": pc_corr,
        "note": "PC signs are arbitrary; magnitudes are what matter.",
    }
    return out


# ---------------------------------------------------------------- tables


def build_tables(analysis_dir: Path, extract_dir: Path, expl: dict[str, Any]) -> dict[str, str]:
    a = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(analysis_dir / "hypothesis_components.csv")
    t: dict[str, str] = {}

    # Samples
    info = a["sample_info"]
    t["samples"] = _md(
        pd.DataFrame(
            [
                {
                    "Sample": s,
                    "Conditions": info[s]["n_conditions"],
                    "Listwise N": info[s]["n_listwise"],
                    "Excluded": len(info[s]["excluded_ids"]),
                    "Bootstrap clusters": info[s]["n_clusters"],
                }
                for s in SAMPLES
            ]
        )
    )

    # MTMM matrices
    for s in SAMPLES:
        m = pd.read_csv(analysis_dir / f"mtmm_pearson_{s}.csv", index_col=0)
        df = pd.DataFrame(
            [{"": i, **{c: ("1" if i == c else _f(m.loc[i, c])) for c in VARS}} for i in VARS]
        )
        t[f"mtmm_{s}"] = _md(df)

    # H1–H3 primary components, pooled, Pearson
    p = rows[(rows["sample"] == "pooled") & (rows["corr_method"] == "pearson")]
    comp = []
    for hyp in ("H1", "H2", "H3"):
        sub = p[(p["hypothesis"] == hyp) & (p["role"] == "primary")]
        for (trait, component), g in sub.groupby(["trait", "component"], sort=False):
            c = g[g["bootstrap"] == "condition"].iloc[0]
            k = g[g["bootstrap"] == "cluster"].iloc[0]
            row = {
                "H": hyp,
                "Trait": trait,
                "Statistic": component.replace("-", "−"),
                "Estimate": _f(c["estimate"]),
                "95% CI (condition)": _ci(c["ci_low"], c["ci_high"]),
                "95% CI (cluster)": _ci(k["ci_low"], k["ci_high"]),
            }
            if hyp == "H3":
                row["Result"] = (
                    f"bootstrap p {_p(c['p_boot'])} (condition), {_p(k['p_boot'])} (cluster); Holm p {_p(c['p_holm'])}; "
                    + ("significant" if bool(c["passes"]) else "not significant")
                )
            else:
                row["Result"] = _yes(bool(c["passes"]) and bool(k["passes"]))
            comp.append(row)
    t["h123_pooled"] = _md(pd.DataFrame(comp))

    # H4
    h4 = rows[(rows["hypothesis"] == "H4") & (rows["sample"] == "pooled")]
    t["h4_pooled"] = _md(
        pd.DataFrame(
            [
                {
                    "Corr": r["corr_method"],
                    "Comparison": r["component"],
                    "Target r": _f(r["estimate"]),
                    "Null p95": _f(r["null_p95"]),
                    "Share of null below target": _f(r["frac_null_below"]),
                    "Result": _yes(bool(r["passes"])),
                }
                for _, r in h4.iterrows()
            ]
        )
    )

    # H5
    h5 = rows[rows["hypothesis"] == "H5"]
    out5 = []
    for (method, trait), g in h5.groupby(["corr_method", "trait"], sort=False):
        c = g[g["bootstrap"] == "condition"].iloc[0]
        k = g[g["bootstrap"] == "cluster"].iloc[0]
        out5.append(
            {
                "Corr": method,
                "Trait": trait,
                "r explicit": _f(c["r_explicit"]),
                "r implicit": _f(c["r_implicit"]),
                "Δ": _f(c["estimate"]),
                "95% CI (condition)": _ci(c["ci_low"], c["ci_high"]),
                "95% CI (cluster)": _ci(k["ci_low"], k["ci_high"]),
                "Result": _yes(bool(c["passes"]) and bool(k["passes"])),
            }
        )
    t["h5"] = _md(pd.DataFrame(out5))

    # Verdict summary across samples and correlation methods
    v = pd.DataFrame(a["verdicts"])
    summ = []
    for s in (*SAMPLES, "explicit_vs_implicit"):
        for method in ("pearson", "spearman"):
            sub = v[(v["sample"] == s) & (v["corr_method"] == method)]
            if sub.empty:
                continue
            row = {"Sample": s + (" (primary)" if s == "pooled" else ""), "Corr": method}
            for hyp in ("H1", "H2", "H3", "H4", "H5"):
                for trait in TRAITS:
                    x = sub[(sub["hypothesis"] == hyp) & (sub["trait"] == trait)]
                    if not x.empty:
                        row[f"{hyp} {trait}"] = "holds" if bool(x["holds"].iloc[0]) else "–"
            summ.append(row)
    t["verdict_summary"] = _md(pd.DataFrame(summ).fillna(""))

    # Outcome patterns
    t["outcome_patterns"] = _md(
        pd.DataFrame(
            [{"Trait": k, "Matching §3.4 rows": "; ".join(vv) if vv else "none"} for k, vv in a["outcome_patterns"].items()]
        )
    )

    # Reliability
    rel = a["reliability"]
    t["reliability"] = _md(
        pd.DataFrame(
            [
                {"Variable": var, "Index": rel["pooled"][var]["kind"].replace("_", " "),
                 **{s: _f(rel[s][var]["value"]) for s in SAMPLES}}
                for var in VARS
            ]
        )
    )

    # Manipulation check
    mc = pd.read_csv(analysis_dir / "manipulation_check.csv")
    same = mc[mc["same_trait"]].pivot_table(index="score", columns="sample", values="pearson_r")
    cross = mc[~mc["same_trait"]].pivot_table(index="score", columns="sample", values="pearson_r")
    t["manipulation"] = _md(
        pd.DataFrame(
            [
                {"Score": var, **{f"{s} same": _f(same.loc[var, s]) for s in SAMPLES},
                 **{f"{s} cross": _f(cross.loc[var, s]) for s in SAMPLES}}
                for var in VARS
            ]
        )
    )

    # Acquiescence
    acq = pd.read_csv(analysis_dir / "acquiescence.csv")
    t["acquiescence"] = _md(
        pd.DataFrame(
            [{"Sample": r["sample"], "Index": r["index"], "Mean": _f(r["mean"]),
              "r with E_SR": _f(r["r_with_E_SR"]), "r with A_SR": _f(r["r_with_A_SR"])}
             for _, r in acq.iterrows()]
        )
    )

    # Disattenuated (pooled, validity diagonal)
    d = pd.read_csv(analysis_dir / "disattenuated.csv")
    d = d[d["sample"] == "pooled"]
    t["disattenuated"] = _md(
        pd.DataFrame(
            [{"Pair": r["pair"].replace("~", " – "), "r": _f(r["r"]), "Disattenuated r": _f(r["r_disattenuated"])}
             for _, r in d.iterrows()]
        )
    )

    # Layer sweep summary
    ls = pd.read_csv(analysis_dir / "layer_sweep.csv")
    ls = ls[(ls["sample"] == "pooled") & (ls["corr_method"] == "pearson")]
    piv = ls.pivot_table(index="layer", columns=["trait", "target"], values="r")
    keep = [0, 4, 8, 11, 12, 14, 17, 20, 24, 28, 31]
    t["layer_sweep"] = _md(
        pd.DataFrame(
            [{"Layer": f"{L}{' (L*)' if L == 17 else ''}",
              **{f"r({tr}_IP, {tr}_{tg})": _f(piv.loc[L, (tr, tg)]) for tr in TRAITS for tg in ("SR", "BC")}}
             for L in keep if L in piv.index]
        )
    )

    # Baseline
    b = a["baseline"][0]
    t["baseline"] = _md(pd.DataFrame([{var: _f(b[var], 3) for var in VARS}]))

    # Phase 4 summary
    ex = json.loads((extract_dir / "extraction_summary.json").read_text(encoding="utf-8"))
    st = ex["steering"]
    t["phase4"] = _md(
        pd.DataFrame(
            [
                {
                    "Trait": tr,
                    "Held-out d′ at L*": _f(ex["dprime_at_L_star"][tr]),
                    "Steering ρ (BC)": _f(st[tr]["bc_spearman"]),
                    "BC change −0.2→+0.2": _f(st[tr]["bc_delta"], 3),
                    "BC option mass": _f(st[tr]["bc_mean_option_mass"], 3),
                    "SR keying-balanced ρ (secondary)": _f(st[tr]["sr_balanced_spearman"]),
                }
                for tr in TRAITS
            ]
        )
    ) + f"\n\nL* = {ex['L_star']} of {ex['n_layers']} blocks; cos(v_E, v_A) at L* = {_f(ex['cosine_EA_at_L_star'])}."

    # Null-distribution spread and exploratory diagnostics
    z = np.load(analysis_dir / "null_correlations.npz")
    t["null_spread"] = _md(
        pd.DataFrame(
            [
                {"Null": kind, "Trait": tr, "Target": tg,
                 "Mean r": _f(float(z[f"pooled|pearson|{tr}|{tg}|{kind}"].mean())),
                 "SD r": _f(float(z[f"pooled|pearson|{tr}|{tg}|{kind}"].std())),
                 "p95": _f(float(np.percentile(z[f"pooled|pearson|{tr}|{tg}|{kind}"], 95)))}
                for kind in ("random", "shuffled") for tr in TRAITS for tg in ("SR", "BC")
            ]
        )
    )
    al = expl["shuffled_null_alignment"]
    t["expl_shuffle"] = _md(
        pd.DataFrame(
            [{"Trait": tr, "Mean |cos(shuffled vector, v_T)|": _f(al[tr]["mean_abs_cos_with_trait_vector"]),
              "Share with |cos| > 0.5": _f(al[tr]["frac_abs_cos_gt_0.5"])} for tr in TRAITS]
        )
    )
    pca = expl["residual_pca_at_L_star"]
    t["expl_pca"] = _md(
        pd.DataFrame(
            [{"Component": k,
              "Variance share": _f(pca["variance_share_pc1_3"][i]),
              **{f"|r| {var}": _f(abs(pca["pc_correlations"][k][var])) for var in (*VARS, "is_implicit")}}
             for i, k in enumerate(("PC1", "PC2", "PC3"))]
        )
    ) + f"\n\nTop 10 components together: {_f(pca['variance_share_top10_cumulative'])} of between-persona variance."
    return t


BLOCK = re.compile(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", re.S)


def fill_report(report_path: Path, tables: dict[str, str]) -> list[str]:
    """Replace each marked block; error if REPORT.md names a table that does not exist."""
    text = report_path.read_text(encoding="utf-8")
    used: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in tables:
            raise KeyError(f"REPORT.md block {name!r} has no generated table")
        used.append(name)
        return f"<!-- BEGIN:{name} -->\n{tables[name]}\n<!-- END:{name} -->"

    report_path.write_text(BLOCK.sub(sub, text), encoding="utf-8")
    return used
