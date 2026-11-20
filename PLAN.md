# PLAN: Trait MTMM Pilot

A zero-cost construct-validity study of LLM personality measures, written as a work plan for an agentic coding model.

## 0. How to use this plan

You are an agent implementing an open-source research project for a human owner. Work through the phases in order. Each phase lists tasks, acceptance criteria, and in some cases a human gate.

Operating rules:

- Budget is $0. Do not call paid APIs. Do not rent compute. Use CPU for development and a free 16 GB GPU (Kaggle or Colab free tier) for the main run.
- Never fabricate data or results. Every number in the report must be produced by committed code from committed inputs. A null or messy result is an acceptable outcome and must be reported as it is.
- Never tune on the test set. All prompt-format and layer choices are made on the dev personas and the extraction set only (see Phase 5). After the preregistration commit, the analysis is frozen.
- Log decisions. Record every choice this plan leaves open, and every assumption in this plan that turns out to be wrong, in DECISIONS.md with a date and a one-line reason. Record any departure from the preregistration in DEVIATIONS.md.
- Stop at human gates. Gates are marked HUMAN GATE. Summarize the state, list what you need, and wait.
- Verify before relying. Model IDs, dataset IDs, licenses, and library APIs named here reflect the plan author's knowledge and a brief search in November 2026. Confirm each one before use. If one is unavailable, choose the closest substitute that meets the stated criteria and log it.
- Do not publish. Do not push to a public remote, create a public repository, or post anywhere without the owner's approval.
- Keep the owner's personal information out of the repository except for a name in LICENSE and CITATION.cff, which the owner supplies.

## 1. Objective

### 1.1 Research question

Three kinds of measurement are used to assign personality traits to language models:

- **SR (self-report):** the model rates questionnaire statements about itself.
- **BC (behavioral choice):** the model chooses among actions in concrete scenarios.
- **IP (internal projection):** the model's activations are projected onto a trait direction ("persona vector").

Prior work mostly studies these separately or in pairs. This project measures all three on the same model under the same persona conditions and asks:

1. Do the three methods converge on the same trait and discriminate between different traits?
2. Does the internal projection track what the model says about itself (SR), what it chooses to do (BC), both, or neither?

### 1.2 Approach in one paragraph

Take one small open-weight chat model. Apply 96 persona system prompts plus one baseline. Under each persona, compute SR, BC, and IP scores for two traits (Extraversion and Agreeableness). Every score comes from a single forward pass read at the next-token position, so there is no text generation, no judge model, and no human rating. Across personas, build a 6 by 6 multitrait-multimethod (MTMM) correlation matrix and test preregistered hypotheses.

### 1.3 Non-goals

- No claims about LLMs in general. One small model is a pilot.
- No claim that forced-choice answers equal real behavior.
- No steering product, no leaderboard, no fine-tuning.

## 2. Fixed design decisions

Do not change these without a logged deviation.

| Element | Decision |
|---------|----------|
| Traits | Extraversion (E), Agreeableness (A) |
| Model | One open-weight instruct model, 2B to 4B parameters (Phase 0 selects it) |
| Test conditions | 96 personas (32 explicit, 64 implicit) plus 1 no-persona baseline |
| Dev conditions | 8 separate personas, used only for piloting |
| SR instrument | IPIP-NEO-120, Extraversion and Agreeableness domains, 24 items each |
| BC instrument | 100 forced-choice scenario items per trait, sampled from TRAIT (fallback in 4.3) |
| IP instrument | Mean-difference trait vectors, projection at the last prompt token over 50 neutral probe prompts |
| Unit of analysis | The persona condition (N = 96) |
| Scoring | Next-token probabilities only. No sampling, no generation |

Statistical power, two-sided alpha .05, 80 percent power: the minimum detectable correlation is about .28 at N = 96, .34 at N = 64, and .48 at N = 32. State this in the report.

Forward-pass budget:

| Stage | Passes |
|-------|--------|
| Per condition: SR 48 items x 2 scale orders | 96 |
| Per condition: BC 200 items x 2 option orders | 400 |
| Per condition: IP 50 probes | 50 |
| Main run: 546 x 97 conditions | 52,962 |
| Dev pilot: 546 x 8 | 4,368 |
| Vector extraction | 480 |
| Steering validation | 960 |
| Total | about 59,000 |

Target: main run under 2 GPU-hours on a 16 GB GPU. Hard cap for the whole project: 10 GPU-hours.

## 3. Repository layout

```
trait-mtmm/
  README.md
  PLAN.md
  PREREGISTRATION.md
  DECISIONS.md
  DEVIATIONS.md
  REPORT.md
  LICENSE
  CITATION.cff
  pyproject.toml
  configs/
    default.yaml
    smoke.yaml
  stimuli/
    adjectives.yaml
    personas_test.jsonl
    personas_dev.jsonl
    ipip_neo_120_EA.csv
    bc_items.jsonl
    extraction_questions.txt
    probe_questions.txt
    banned_terms.txt
  src/trait_mtmm/
    model.py
    scoring.py
    measures/sr.py
    measures/bc.py
    measures/ip.py
    vectors.py
    run.py
    analysis.py
    figures.py
  tests/
  notebooks/
    run_kaggle.ipynb
  results/
  figures/
```

Engineering requirements:

- Python 3.11+, torch, transformers, datasets, numpy, pandas, scipy, pyarrow, matplotlib, pyyaml, pytest.
- Plain transformers with `output_hidden_states=True` and forward hooks.
- All randomness seeded from the config. Record model revision hash, dataset revision, library versions, and git commit in every results file.
- Every pipeline stage is a CLI command, caches output as parquet, and can resume.
- pytest must pass on CPU in under 5 minutes using `configs/smoke.yaml`.

## 4–10. Stimuli, measures, phases, analysis, deliverables

Full specifications for stimuli (Section 4), measurement (Section 5), phases and human gates (Section 6), analysis (Section 7), deliverables (Section 8), related work (Section 9), and definition of done (Section 10) are as stated in the owner brief that created this repository. Phase 0 verification outcomes and any plan corrections are logged in `DECISIONS.md`. Do not invent substitutes for those sections; implement them from the brief and log deviations.

### Human gates (quick reference)

| Gate | When | Owner provides |
|------|------|----------------|
| G0 | End of Phase 0 | Model confirmation; TRAIT HF token or fallback approval; code license; repo name; owner name for LICENSE/CITATION |
| G1 | End of Phase 5 | Approve preregistration before any test persona run |
| G2 | Phase 6 if no GPU | Run notebook; return results files |
| G3 | End of Phase 8 | Review before any public release |

### Phase status

| Phase | Status |
|-------|--------|
| 0 Verification and setup | Done (G0 passed; TRAIT verified) |
| 1 Core library | Done |
| 2 Stimuli | Done |
| 3 Measures and vectors | Done (smoke E2E) |
| 4 GPU extraction and dev pilot | Done (2026-11-17): gates v2 passed, L*=17; dev pilot acceptance passed |
| 5 Preregistration | Done (2026-11-19): G1 passed; frozen at tag `prereg-v2` (v1 superseded) |
| 6 Main run | Done (2026-11-20): G2 passed; 97 conditions, 0 exclusions |
| 7 Analysis | Done (2026-11-20): code `40433bf` committed before run; results in `results/analysis/`, verdicts in DECISIONS |
| 8 | Not started |
