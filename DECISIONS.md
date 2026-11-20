# DECISIONS

Dated log of choices left open by the plan, assumptions that proved wrong, and human-gate outcomes.

---

## 2026-11-16 — Phase 0 verification

### Primary model (confirmed G0)

- **Choice:** `microsoft/Phi-3.5-mini-instruct`
- **Revision:** `2fe192450127e6a83f7441aef6e3ca586c338b77`
- **Parameters:** ~3.82B
- **License:** MIT
- **Gated:** No
- **Reason:** Owner confirmed at Gate G0. Ungated MIT instruct model in the 2B–4B band.

### Candidate comparison (verified via Hugging Face API, 2026-11-16)

| Model | Params | License | Gated | Notes |
|-------|--------|---------|-------|-------|
| `microsoft/Phi-3.5-mini-instruct` | 3.82B | MIT | No | **Recommended** |
| `Qwen/Qwen2.5-3B-Instruct` | 3.09B | Qwen Research (non-commercial) | No | Research OK; redistribution of results OK under research license; commercial use restricted |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.21B | Llama 3.2 Community | Manual | Requires Meta accept |
| `google/gemma-2-2b-it` | 2.61B | Gemma | Manual | Requires Google accept |

### Smoke model

- **Choice:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Revision:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **License:** Apache-2.0
- **Gated:** No
- **Chat template:** Yes (ChatML)
- **Reason:** Exists, tiny, CPU-friendly, has chat template and hidden states.

### TRAIT dataset

- **IDs:** `mirlab/TRAIT` (canonical in paper); `snupilab/TRAIT` is the same revision (`8b31c078cb897c3917d2ee48735d0c15030680e0`).
- **Gated:** `auto` (Hugging Face account + accept terms required).
- **License (card text):** CC-BY-4.0 for annotations. API `license` field is unset; treat card text as authoritative and re-check before redistributing items.
- **Fields:** `personality`, `question`, `response_high1`, `response_high2`, `response_low1`, `response_low2`.
- **Splits:** load with `load_dataset("mirlab/TRAIT", split="Extraversion")` (and `Agreeableness`).
- **Paper:** Lee et al., arXiv:2406.14703, Findings of NAACL 2025.
- **Action for G0:** Owner must accept TRAIT terms and supply an HF token, or approve the 4.3 fallback (agent-authored items).

### IPIP-NEO-120

- **Source:** https://ipip.ori.org/30FacetNEO-PI-RItems.htm
- **Citation:** Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory. *Journal of Research in Personality*, 51, 78–89.
- **License:** Public domain (IPIP).
- **Status:** Confirmed available. Extraversion and Agreeableness domains each have 24 items (6 facets × 4). Item text and keying will be taken from ipip.ori.org in Phase 2.

### Literature search — prior three-method MTMM?

Searched for studies combining self-report, behavioral choice, and activation-based measures in one MTMM analysis (2026-11-16).

**No exact match found.** Closest related work:

| Work | What it does | Gap vs this pilot |
|------|--------------|-------------------|
| Serapio-García et al. (arXiv:2307.00184) | MTMM across two questionnaires (IPIP-NEO × BFI) | No behavioral forced-choice; no activation projection |
| Han et al. (arXiv:2509.03730) | SR vs downstream behavior dissociation | No activation/IP method; not full MTMM |
| arXiv:2606.09843 | LLM-native SR vs human-rated open-ended behavior | No persona-vector IP; cross-model, not within-persona MTMM |
| ActTraitBench (arXiv:2605.29791) | Knowledge–decision gap (SR vs behavior) | No IP |
| PVNI / ACL Findings 2026 | Internal activations for trait evaluation | Not a three-method MTMM with SR+BC+IP under shared personas |
| PERSONA (arXiv:2602.15669) | Contrastive trait vectors / steering | Control focus, not MTMM validity |
| Chen et al. persona vectors (arXiv:2507.21509) | Activation persona vectors | Monitoring/control, not SR+BC MTMM |

**Owner note for G0:** This pilot appears novel as a three-method (SR / BC / IP) MTMM under shared persona conditions. Position accordingly; not a direct replication.

### Section 9 reference verification

| Plan citation | Exists? | Plan claim accuracy |
|---------------|---------|---------------------|
| Serapio-García et al., arXiv:2307.00184 | Yes | Correct: uses MTMM across two questionnaires |
| Lee et al., TRAIT, arXiv:2406.14703 | Yes | Correct; NAACL 2025 Findings |
| Han et al., arXiv:2509.03730 | Yes | Correct: self-report and behavior dissociate |
| arXiv:2606.09843 | Yes | Claim OK with nuance: title is about an LLM-native instrument; self-report does not predict human-rated open-ended behavior (mean r≈.09) |
| PERSONA, arXiv:2602.15669, ICLR 2026 | Yes | Correct: trait vectors via contrastive activation analysis |
| arXiv:2602.15847 | Yes | Correct: Big Five steering vectors geometrically entangled |
| Lu et al., Assistant Axis, arXiv:2601.10387 | Yes | Correct |
| Beyond Static Personas (IRIS), arXiv:2604.13846, ACL 2026 Findings | Yes | Correct (framework name IRIS / IRiS) |
| Zeng et al., arXiv:2607.13162 | Yes | Correct: persona-vector auditing |
| Campbell & Fiske (1959) | Yes | MTMM classic |
| Johnson (2014) IPIP-NEO-120 | Yes | Public domain instrument |
| Goldberg (1992) Big Five markers | Yes | Psychological Assessment, 4(1), 26–42 |

### Smoke model acceptance (Phase 0)

- **2026-11-16:** `pytest tests/test_smoke_model.py` passed on CPU with `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Model loads, chat-formats, and returns `output_hidden_states` residuals. Library versions at test: torch 2.15.0, transformers 5.23.0.

### Repository name and code license (confirmed G0)

- **Repository / project name:** Libra
- **Code license:** MIT
- **Copyright / citation author:** Steven Harvath
- **Python package name:** `libra` (import path remains `trait_mtmm` per plan layout)

### HUMAN GATE G0 — passed (2026-11-16)

Owner confirmed: Phi-3.5-mini-instruct; TRAIT via HF token (stored only in local gitignored `.hf_token`, never committed); MIT; name Libra; author Steven Harvath.

**TRAIT access:** Verified 2026-11-16. `load_dataset("mirlab/TRAIT", split="Extraversion")` returns 1000 rows with fields `personality`, `question`, `response_high1`, `response_high2`, `response_low1`, `response_low2`. Agreeableness split also 1000 rows. HF user `sharvath`. Token kept only in gitignored `.hf_token` (and HF hub cache after login). Never commit tokens.

**Env note:** A stale `HF_TOKEN` env var overrides `.hf_token` / `hf auth login`. Unset it if loads fail with 401 after rotation.

### Phase 1 — core library (2026-11-16)

- Implemented `model.py` (load, chat format, hidden states, residual-add steering hooks) and `scoring.py` (option-token discovery, renormalization, SR expected value, reverse keying, BC high-trait probability, order averaging).
- Acceptance: `pytest tests/` passes on CPU with smoke model.

### Phase 4 — GPU extract + dev pilot (2026-11-16)

- **Local GPU:** none (cuda/mps unavailable). Agent cannot run Phi-3.5 extract/pilot locally.
- **Hand-off:** `notebooks/run_kaggle.ipynb` clones/installs and runs `phase4` (`extract` → `pilot` → `diagnose`) on a free 16 GB GPU.
- **CLI:** `python -m trait_mtmm.run phase4 --config configs/default.yaml` (also `extract`, `pilot`, `diagnose` separately).
- On successful extract, `layer.L_star` is written into `configs/default.yaml`.
- Prompt-format retries (max 2): edit `scoring.assistant_prefill` / SR wording on extract+dev only, re-run, log each attempt here.

#### 2026-11-17 — first Kaggle extract result

- **Separation passed** at `L_star=17` (32 layers): held-out d′ E≈7.11, A≈15.37; cos(E,A)≈0.23.
- **Steering failed:** mean SR option mass ≈1e−9 under prefill `"Answer:"` (digits get essentially no next-token mass on Phi-3.5-mini). Spearman E≈−1, A≈0.1 — not interpretable with near-zero mass.
- **Attempt 1 (prompt format):** set `assistant_prefill: ""` and tighten SR instruction to “exactly one digit…”. Re-run on GPU → still ~0 digit mass (`baseline_option_mass` ≈ 7e−12).
- **Attempt 2 (prompt format):** switch SR to **letter options A–E** (same 5-point Likert values 1–5), keep empty prefill (`scoring.sr_format: letters`). Digits appear to be a bad first-token for Phi-3.5-mini under this chat template; letters match the BC response style. This is the second and final planned prompt-format attempt before stopping if validation still fails.

#### 2026-11-17 — Attempt 2 result (STOP)

- **Option mass:** fixed. `baseline_option_mass` ≈ 0.999; steered mean mass ≈ 0.999–1.0 at scale 1.0. Letters are usable as next-token options on Phi-3.5-mini.
- **Separation:** still passes (`L_star=17`; d′ E≈7.11, A≈15.37; cos≈0.23).
- **Steering Spearman:** **failed.** E ≈ 0.900 (reported `0.8999999999999998`, so `>= 0.9` fails on float); A ≈ **−1.0** (perfectly anti-monotonic — high +A steering lowers keyed SR). Scale stayed 1.0 (mass never forced shrink).
- **Bugfix mid-run:** `NameError: apply_reverse_keying` in `vectors.py` (import dropped); fixed in `63601d2`, then re-ran to this summary.
- **Decision:** **Stop.** Two prompt-format attempts exhausted. Do **not** flip the A vector, relax the Spearman threshold, or invent a third prompt. Phase 4 extract gate did not pass; pilot was not run.
- **Owner choices from here:** (1) accept Phase 4 failure and redesign extraction/steering (new phase / prereg change); (2) inspect `results/extract/steering_curves.parquet` on Kaggle for the A inversion pattern; (3) only if owner explicitly authorizes a scoped code bug-hunt (not a third format tweak).

#### 2026-11-17 — Steering diagnostic (owner-authorized; not a gate or prompt attempt)

- **Hypotheses:** (a) E fails only on float: with 5 coefficients ρ ∈ {1.0, 0.9, 0.8, …}; 0.9 = one adjacent swap, returned as `0.8999999999999998` < 0.9. (b) A inversion is acquiescence × keying imbalance: IPIP A items are 7 `+` / 17 `−` (E is 18 `+` / 6 `−`), so a uniform shift toward "Accurate" raises keyed E and lowers keyed A. (c) The gate scores standard label order only, unlike `score_sr_item` (both orders). (d) ±2 on a unit vector may be tiny relative to ‖h‖ at L*.
- **Tool:** `scripts/steering_diag.py` — both orders, split by keying, raw and ‖h‖-scaled coefficients. No gate code changed; fixes wait on its output.
- **Attempt 2 `steering_curves.parquet` (downloaded):** across coef −2 → +2, keyed SR moved E 3.915 → 3.923 (Δ≈0.008) and A 4.330 → 4.300 (Δ≈−0.030) on the 1–5 scale; mass ≈0.9996–0.9999 throughout. Vectors are unit-norm, so ±2 is a negligible push at L*=17. The gate's Spearman ranked differences at fp16-noise scale (E's one swap: 3.9213 vs 3.9210). **Reading:** the steering gate did not test steering; the A ρ=−1 shows a slightly negative slope under near-zero perturbation, not that the vector is inverted. Hypothesis (d) is primary; (a)–(c) still apply once coefficients are meaningful.
- **Notebook:** `STAGE = "steering_diag"` runs the diagnostic; cell 2 now fetches/resets an existing clone instead of silently reusing stale code; cell 5 prints a download link and the diag table.

#### 2026-11-17 — Steering diagnostic result (Kaggle, Phi-3.5, L*=17)

- Mean last-token ‖h‖ at L* = 101.5, so raw coef 2 = 1.97% of ‖h‖. At ±0.1/±0.2 ‖h‖ option mass stays ≥ 0.997.
- **E:** keying-balanced, both orders: 3.268 → 3.378 → 3.482 → 3.515 → 3.524 (ρ = 1.0). The old gate metric (standard order only) on the same coefficients: 3.530 → 3.813 → 3.921 → 3.916 → 3.900 (ρ = 0.6). Single-order scoring masks the effect.
- **A:** raw agreement (unkeyed) 2.284 → 2.428 → 2.501 → 2.558 → 2.788 (ρ = 1.0): the A direction mainly drives acquiescence. Forward-keyed A items are near ceiling at baseline (4.48/5), so acquiescence lands on reverse-keyed items (keyed 4.53 → 3.88) and keying-balanced A still falls (4.396 → 4.448 → 4.396 → 4.325 → 4.145, ρ ≈ −0.8). −A steering lowers forward-keyed items as expected (4.48 → 4.26).
- **Reading:** A vector is not inverted; on an at-ceiling baseline assistant its SR effect is acquiescence (an SR method artifact).

#### 2026-11-17 — OWNER DECISION: steering gate v2 (fixed before the next GPU run)

Owner chose "A + B1". This rule is fixed **before** any run of it and will be reported whatever the outcome. Changed after seeing dev results (forking-paths risk acknowledged); still pre-preregistration, so logged here, not in DEVIATIONS.md.

- **Population:** baseline assistant (no persona system prompt), L* chosen by held-out d′ as before.
- **Coefficients:** {−0.2, −0.1, 0, +0.1, +0.2} × mean last-token ‖h‖ at L*, where ‖h‖ is measured on unsteered BC prompts (first 4 items per trait, forward order). Existing shrink-by-half rule if option mass < 0.80.
- **Gating criterion (per trait):** Spearman(coefficient, mean BC high-trait probability over all 100 BC items of that trait, forward + reversed option order averaged, pilot order seeds) ≥ **0.9** (unchanged threshold, compared with 1e−9 float tolerance), and mean BC option mass ≥ 0.80. Both E and A must pass. BC is forced choice between high/low responses, so acquiescence cannot drive it.
- **Secondary (non-gating):** SR at the same coefficients, both label orders, keying-balanced; plus raw-agreement Spearman (acquiescence). Reported in `extraction_summary.json` and `steering_curves.parquet` (`measure` = `bc` / `sr_balanced`).
- **Not changed:** threshold 0.9, vector sign, SR/BC prompt formats, L* rule, separation gate (d′ ≥ 2).
- **If v2 fails:** stop; Phase 4 fails and goes to redesign. No further gate revisions on this dev data.
- **Code:** `validate_steering` in `src/trait_mtmm/vectors.py`; smoke E2E passes (numbers meaningless).

#### 2026-11-17 — Gate v2 result (Kaggle, `aa5a2a6`): PASSED

- **Separation:** L* = 17; d′ E 7.11, A 15.37; cos(E,A) 0.23. ‖h‖ at L* (BC prompts) = 104.6.
- **E:** BC ρ = 1.0, Δ(−0.2→+0.2 ‖h‖) = +0.216, mass 0.989 → pass. Secondary SR balanced ρ = 1.0 (Δ +0.267); raw-agree ρ = 1.0.
- **A:** BC ρ = 1.0, Δ = +0.074, mass 0.991 → pass. Secondary SR balanced ρ = −0.7 (Δ −0.264); raw-agree ρ = 1.0 (acquiescence, as in the diagnostic).
- **all_gates_passed: true**; `layer.L_star: 17` frozen in `configs/default.yaml`.
- **Report:** A steers behavior (BC) but its SR effect is acquiescence-dominated at an at-ceiling baseline; carry this forward as an SR method-variance note. A's BC effect is ~⅓ of E's.
- **Pilot crash (bug, not a gate):** `ip.project` dotted a CPU-loaded vector with a CUDA residual. Fixed by moving the vector to the residual's device; CPU smoke could not catch it.

#### 2026-11-17 — Phase 4 dev pilot + diagnose (Kaggle, `7b0ac20`): ACCEPTANCE PASSED

- **Option mass:** SR 0.9999, BC 0.9986 (threshold 0.80). **Floor/ceiling:** 0 pinned conditions on any SR/BC column. **Variance (SD across 8 dev personas):** E_SR 0.373, A_SR 0.187, E_BC 0.022, A_BC 0.051, E_IP 0.020, A_IP 0.106 — all pass. `phase4_acceptance_passed: true`.
- **Dev MTMM (Pearson, n = 8 — pipeline check only, no inference):** monotrait–heteromethod E: SR–BC 0.83, SR–IP 0.14, BC–IP 0.22; A: SR–BC 0.31, SR–IP 0.32, BC–IP 0.99. Largest heterotrait: E_IP–A_SR −0.81.
- **Dev design caveat:** intended E and A levels are correlated r = −0.57 across the 8 dev personas (test sets are fully crossed: explicit and implicit r = 0.00). Heterotrait values above are confounded by design and must not inform the preregistration.
- **Carry into Phase 5 (preregistration):** (1) A_SR is acquiescence-prone (gate diagnostic); prereg how SR is scored and whether an acquiescence index is reported. (2) BC was the steering-validation criterion for the vectors, so IP–BC convergence (esp. A 0.99) is not fully independent evidence; prereg how it is interpreted. (3) E_IP has low between-persona SD (0.020); note as a risk for E IP validity.
- **Next:** Phase 5 preregistration → HUMAN GATE G1 before any test persona run.

### Phase 7 — analysis (2026-11-20)

- **Code before data:** `src/trait_mtmm/mtmm.py` implements prereg §2–§3, `src/trait_mtmm/figures_main.py` implements §3.5, and the CLI command is `phase7`. All three were committed before being run on `results/main`. `phase7` refuses to run if tracked files have uncommitted changes, or if the main run's vectors do not match the pin.
- **Tests (`tests/test_phase7_analysis.py`, synthetic data and stimuli only).** They check:
  - the weighted correlations against Pearson/Spearman on expanded resamples;
  - the cluster weights (explicit singletons give the condition bootstrap exactly);
  - the 12 cue clusters (sizes 5–6, one quadrant each);
  - the §2 thresholds at 4/5 of 24 and 20/21 of 100;
  - Holm, bootstrap p and α;
  - that the null vectors are seeded;
  - that the data with signal support H1, H2 and H4 and the noise data support none of them.
- **Clarifications D1–D11** were logged in `DEVIATIONS.md` before the run.

### Phase 6 — main run (2026-11-19)

- **Wiring (after `prereg-v2`, as allowed by prereg §5):** `main` runs the 96 test personas plus baseline through `_score_personas`, the same SR/BC/IP path the dev pilot now uses. It caches residuals, requires the pinned vectors and refuses to run unless L* = 17.
- **Resume:** per persona, from `results/main/parts`; it refuses parts written at a different git commit.
- **No outcomes at run time:** `main` reports only data quality in `main_quality.json` (option mass, item and condition exclusion counts, completeness, non-finite IP). It prints no scores, correlations or SDs, so the Phase 7 analysis code is written before outcomes are seen.
- **Smoke test:** `tests/test_phase6_main.py`; all 31 tests pass.
- **G2:** owner runs `STAGE = "main"` on Kaggle with `trait_vectors.pt` uploaded as dataset `libra-vectors`.
- **G2 passed: main run on Kaggle (finished 2026-11-20T07:00 UTC).** The owner returned `libra_main_results.zip`, copied to `results/main/` (gitignored). Only provenance and data quality were checked:
  - commit `dab96bb` in `provenance.json` and in all 97 `parts/*.done.json`;
  - vectors sha256 `5e12b0ba…a402ca` matches the pin;
  - L* = 17, model revision `2fe19245…`, torch 2.11.0+cu130, transformers 5.20.2 (same as Phase 4);
  - bundled `default.yaml` identical to the repo.
  - `main_quality.json`: 97 conditions (32 explicit, 64 implicit, 1 baseline); rows complete; SR option mass mean 0.99995 (min 0.983); BC mean 0.998 (min 0.580); 0 items below 0.50; 0 condition scores excluded; 0 non-finite IP.
  - No scores or correlations were opened. Phase 7 analysis code is written and committed before it is run on these files (prereg §5).

### Phase 5 — Preregistration (2026-11-18)

- **prereg-v1 draft** (`d171e6e`, written by a separate Cursor session): tagged and pushed to the public remote before G1 review. Kept as a record and superseded; never used for any test run.
- **Review of v1 found:** (1) H4 nulls and the layer sweep had no inputs, because IP stored only the L* score and extraction activations were not saved; (2) IP was described as a projection, but the code computes cosine; (3) decision rules for H1/H2/H4/H5 and the missing-data rule were unspecified; (4) the extraction was described as 10 questions when there are 20 (480 passes); (5) the trait vectors were not pinned.
- **Code added before G1 (no test data touched):** IP caches all-layer last-token residuals (float16) per persona; extraction saves `extraction_activations.pt`; `extract-acts` command (and notebook stage) re-runs the extraction passes against the existing vectors without changing them; `vectors.sha256` pin, checked at load and required for the main run. The refactored extraction reproduces the old smoke vectors (cos ≥ 0.9999998 at every layer). Tests: `tests/test_prereg_v2_caching.py`; full suite passes (27 fast, 3 slow).
- **Stimulus finding:** implicit test personas encode only the quadrant, through one of 3 cue sentences per quadrant (12 cues across 64 personas). "very" vs "somewhat" labels were assigned by `k % 2` and are not in the text. Raised as a G1 decision; stimuli unchanged.
- **PREREGISTRATION.md v2 draft** written with explicit decision rules, a listwise rule, a pole-coded manipulation check for implicit personas, and a preregistered cluster-bootstrap robustness check. Not tagged. On approval → tag `prereg-v2`.
- **2026-11-19 — Phase 4 re-run on Kaggle at `675f7f6`** (owner ran `phase4`; extraction set and dev personas only). Re-run guard passed: L* = 17; d′ E 7.107, A 15.372; cos(E,A) 0.231; ‖h‖ at L* 104.6; BC ρ E 1.0 (Δ +0.216), A 1.0 (Δ +0.074); masses 0.989 / 0.991; secondary A SR balanced ρ −0.7, raw-agree 1.0. Dev diagnostics and dev MTMM identical to 2026-11-17 (SD E_IP 0.020; A BC–IP 0.99). Torch 2.11.0+cu130, transformers 5.20.2.
- **Local checks of the re-run:** vectors recomputed from `extraction_activations.pt` match `trait_vectors.pt` (cos 1.0 at all 32 layers); split 64 train / 56 holdout per pole; cached dev residuals at L* reproduce `ip_probe` scores (max |Δ| 2e−7).
- **Pinned vectors:** `vectors.sha256 = 5e12b0ba04f6b2a7f09355e9156795629b689bdf0739a1ae2c44af185ca402ca`. The re-run config was not merged: the bundled `default.yaml` had its comments stripped by `set_L_star` and its values match the repo.
- **HUMAN GATE G1 — passed (2026-11-19).** Owner: "use your prudence to make the best decisions and proceed to phase 6". Agent resolutions:
  - H1–H5 wording kept as in v1; not verified against the brief.
  - Verdicts require both the condition-level and the cluster bootstrap (12 implicit cue clusters), so they are conservative.
  - H2 uses absolute heterotrait values.
  - Implicit personas kept: a rebuild would be unpiloted stimuli.
  - Tagged `prereg-v2`. No test persona run before the tag.

### Phase 3 — measures and smoke E2E (2026-11-16)

- Implemented SR/BC/IP measures, vector extraction (mean-difference, L* by held-out d', steering curves), CLI (`extract`, `pilot`, `smoke`, `figures`, `analyze`).
- Smoke gates are not enforced (`enforce_gates=False`); smoke-model validation numbers are meaningless and must not be reported.
- Acceptance: `python -m trait_mtmm.run smoke --config configs/smoke.yaml` writes `results/pilot_smoke/condition_scores.parquet` and six `figures/smoke_*.png`. Fast pytest suite passes (`-m 'not slow'`).

### Phase 2 — stimuli (2026-11-16)

- **Adjectives:** Goldberg (1992) unipolar Factor I/II poles alone yield only 10/pole; supplemented with Saucier (1994) Mini-Markers and additional Goldberg bipolar-pole adjectives to reach 16/pole. Split extraction/test with seed 42; disjointness enforced by tests.
- **IPIP:** Extraversion + Agreeableness 48 items from ipip.ori.org (Johnson, 2014), public domain.
- **BC:** Sampled 100 Extraversion + 100 Agreeableness from `mirlab/TRAIT` with seed 42. License CC-BY-4.0 — full items committed in `stimuli/bc_items.jsonl` with rebuild script `scripts/build_bc_items.py`.
- **Personas:** 32 explicit + 64 implicit + baseline; 8 dev. Intended levels in `stimuli/persona_intended_levels.csv`.
- Acceptance: stimulus lint tests pass.

### Plan corrections logged

1. TRAIT HF org: paper and load path use `mirlab/TRAIT`; Hub also shows `snupilab/TRAIT` at the same SHA. Use `mirlab/TRAIT`.
2. TRAIT is gated (`auto`), as the plan stated; owner HF token required.
3. Qwen2.5-3B-Instruct is ungated but under Qwen Research License (non-commercial), not Apache-2.0.
4. No published three-method SR+BC+IP MTMM found as of 2026-11-16.
