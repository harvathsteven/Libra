# Libra

Zero-cost construct-validity study of LLM personality measures: **self-report (SR)**, **behavioral choice (BC)**, and **internal projection (IP)** on one small open-weight chat model under shared persona conditions.

**Status:** Gate G0 passed. Phase 1 next. Do not publish. Do not run the main test set.

## Research question

Do SR, BC, and IP converge on the same trait and discriminate between Extraversion and Agreeableness? Does the internal projection track what the model says (SR), what it chooses (BC), both, or neither?

## Design (summary)

| Element | Decision |
|---------|----------|
| Traits | Extraversion, Agreeableness |
| Model | `microsoft/Phi-3.5-mini-instruct` |
| Conditions | 96 personas + 1 baseline; 8 dev personas |
| Scoring | Next-token probabilities only; no generation |

## Repository layout

See `PLAN.md` and `DECISIONS.md`. Code lives under `src/trait_mtmm/`.

## Reproduce (after G0)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest  # CPU smoke tests; < 5 minutes target once Phase 1 is done
```

## Citation

See `CITATION.cff` (Steven Harvath).

## License

MIT — Copyright (c) 2026 Steven Harvath.
