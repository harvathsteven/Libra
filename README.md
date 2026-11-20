# Libra

Zero-cost construct-validity study of LLM personality measures: **self-report (SR)**, **behavioral choice (BC)**, and **internal projection (IP)** on one small open-weight chat model under shared persona conditions.

**Status:** Phases 0–8 complete. Preregistered at tag `prereg-v2`; main run and analysis done; results in `REPORT.md` (draft, awaiting owner review at G3). Do not publish.

## Research question

Do SR, BC, and IP converge on the same trait and discriminate between Extraversion and Agreeableness? Does the internal projection track what the model says (SR), what it chooses (BC), both, or neither?

## Design (summary)

| Element | Decision |
|---------|----------|
| Traits | Extraversion, Agreeableness |
| Model | `microsoft/Phi-3.5-mini-instruct` (rev `2fe19245…`) |
| Smoke model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Conditions | 96 personas + 1 baseline; 8 dev personas |
| Scoring | Next-token probabilities only; no generation |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest   # fast CPU tests (excludes slow E2E by default)
```

For TRAIT rebuilds only: put a read HF token in gitignored `.hf_token` (or `HF_TOKEN`), after accepting the gate at https://huggingface.co/datasets/mirlab/TRAIT.

## CLI

```bash
# Phase 3 smoke (CPU; numbers are meaningless — do not report)
python -m trait_mtmm.run smoke --config configs/smoke.yaml

# Individual stages (same config)
python -m trait_mtmm.run extract --config configs/smoke.yaml
python -m trait_mtmm.run pilot --config configs/smoke.yaml
python -m trait_mtmm.run figures --config configs/smoke.yaml
```

Smoke outputs: `results/pilot_smoke/condition_scores.parquet` and `figures/smoke_*.png` (local only; gitignored).

Primary-model config: `configs/default.yaml` (Phase 4+).

## Phase 4 (GPU)

No local GPU was available in development. Use Kaggle/Colab free tier (16 GB):

1. Open `notebooks/run_kaggle.ipynb` on a GPU runtime.
2. Point it at this repo (public clone, private token secret, or uploaded zip).
3. Run stage `phase4` with `configs/default.yaml`.
4. Download `libra_phase4_results.zip` and merge `results/`, `figures/`, and updated `configs/default.yaml` back here.
5. Confirm `results/extract/extraction_summary.json` → `all_gates_passed` and `results/pilot/phase4_diagnostics.json` → `phase4_acceptance_passed`.

```bash
# On the GPU machine / notebook:
python -m trait_mtmm.run phase4 --config configs/default.yaml
# Or stepwise:
python -m trait_mtmm.run extract --config configs/default.yaml
python -m trait_mtmm.run pilot --config configs/default.yaml
python -m trait_mtmm.run diagnose --config configs/default.yaml
```

## Layout

See `PLAN.md` and `DECISIONS.md`. Package code: `src/trait_mtmm/`. Stimuli: `stimuli/`.

## Citation

See `CITATION.cff` (Steven Harvath).

## License

MIT — Copyright (c) 2026 Steven Harvath.
