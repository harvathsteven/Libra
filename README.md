# Libra

Zero-cost construct-validity study of LLM personality measures: **self-report (SR)**, **behavioral choice (BC)**, and **internal projection (IP)** on one small open-weight chat model under shared persona conditions.

**Status:** Phases 0–3 complete (G0 passed; stimuli + measures + smoke E2E). Phase 4 next (GPU extraction / dev pilot). Do not publish. Do not run the main test set until after preregistration (`prereg-v1`).

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

Primary-model config: `configs/default.yaml` (used from Phase 4 onward).

## Layout

See `PLAN.md` and `DECISIONS.md`. Package code: `src/trait_mtmm/`. Stimuli: `stimuli/`.

## Citation

See `CITATION.cff` (Steven Harvath).

## License

MIT — Copyright (c) 2026 Steven Harvath.
