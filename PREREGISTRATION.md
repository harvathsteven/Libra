# Preregistration — Trait MTMM Pilot (Libra)

**Tag:** `prereg-v1`  
**Date:** 2026-11-18  
**Status:** Awaiting HUMAN GATE G1 (owner approval). No test-persona run until approved.

This document freezes the design, scoring, exclusion rules, hypotheses, and analyses before any test-persona measurement. After the tagged commit, analysis code for the preregistered tests must not change except via a logged entry in `DEVIATIONS.md`.

Dev-pilot numbers (N = 8) and extraction diagnostics informed prompt format and the steering gate; they do **not** inform hypothesis outcomes. Heterotrait associations on the dev set are confounded by design (intended E–A r = −0.57 among the 8 dev personas) and must not be treated as evidence.

---

## 1. Frozen design

| Element | Decision |
|---------|----------|
| Traits | Extraversion (E), Agreeableness (A) |
| Model | `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Dtype / device (main run) | float16 / CUDA |
| Config | `configs/default.yaml` |
| Global seed | 42 |
| Test conditions | 96 personas (32 explicit, 64 implicit) + 1 no-persona baseline |
| Unit of analysis | Persona condition; primary N = 96 (baseline descriptive only) |
| Secondary samples | Explicit only (N = 32); implicit only (N = 64) |
| SR instrument | IPIP-NEO-120 E and A domains, 24 items each (`stimuli/ipip_neo_120_EA.csv`) |
| BC instrument | 100 TRAIT forced-choice items per trait (`stimuli/bc_items.jsonl`; `mirlab/TRAIT`, CC-BY-4.0) |
| IP instrument | Mean-difference persona vectors; projection of last-prompt-token residual at L* onto each unit vector, mean over 50 probe prompts |
| Scoring | Next-token probabilities only; no sampling; no generation |
| Layer L* | **17** (0-based transformer block index), chosen by maximizing mean held-out d′ across E and A (tie-break: closest to network middle). Frozen in `configs/default.yaml` on 2026-11-17 |
| Extraction | 6 adjective triples × 10 questions × high/low poles × 2 traits; 80/20 split by triple and question, seed 42 |
| Separation gate (passed) | Held-out d′ at L* ≥ 2.0 for each trait (observed: E ≈ 7.11, A ≈ 15.37; cos(E,A) ≈ 0.23) |

### 1.1 Final prompt formats (dev-only; frozen)

**SR (`scoring.sr_format: letters`, `assistant_prefill: ""`):**

- User asks how accurately the statement describes the model.
- Five IPIP labels mapped to letters **A–E** with values 1–5 (A = Very Inaccurate … E = Very Accurate in standard order).
- Instruction: respond with exactly one letter (A–E); no other text.
- Each item scored in **standard and reversed** label order; reversed expected value mapped back; average of the two; then reverse-keying (`6 − x`) when `keying == "-"`.
- Trait score = mean over 24 keyed item scores.
- Digits 1–5 were abandoned on Phi-3.5-mini after ~0 next-token option mass (two prompt-format attempts; logged in `DECISIONS.md`).

**BC:**

- Scenario + four options labeled **A–D**; answer with a single letter.
- Item score = P(high-trait options) / P(all four) under renormalized mass.
- Each item in one seeded option order and its exact reverse; average the two.
- Trait score = mean over 100 items.

**IP:**

- 50 neutral probes (`stimuli/probe_questions.txt`) with the persona system prompt.
- At L*, project last-prompt-token residual onto the unit E and A vectors; mean over probes.
- Last-token residuals at all layers cached (float16) for the layer sweep and null analyses.

### 1.2 Steering validation (frozen; supersedes plan §5.4 SR criterion)

Original plan §5.4 required SR Spearman ≥ 0.9 under residual steering. On this model, baseline SR for A is acquiescence-dominated (forward-keyed items near ceiling), so SR monotonicity is not a clean vector check. **Before** re-running the gate, the owner fixed gate **v2** (logged in `DECISIONS.md`, 2026-11-17):

- Population: baseline assistant (no persona).
- Coefficients: {−0.2, −0.1, 0, +0.1, +0.2} × mean last-token ‖h‖ at L* (‖h‖ from unsteered BC prompts).
- **Gating criterion:** Spearman(coefficient, mean BC high-trait probability over all 100 BC items of that trait, forward + reversed orders) ≥ 0.9 (float tolerance 1e−9), and mean BC option mass ≥ 0.80. Both traits must pass.
- Secondary (non-gating): keying-balanced SR and raw-agreement Spearman at the same coefficients.

**Gate v2 passed** (Kaggle, 2026-11-17): E and A BC ρ = 1.0; masses ≥ 0.98. Secondary: A SR balanced ρ ≈ −0.7 with raw-agree ρ = 1.0 (acquiescence).

**Interpretive consequence (preregistered):** the E/A vectors were selected partly by BC steering success. Monotrait **IP–BC** correlations are therefore **not fully independent** of the vector-validation procedure. They remain preregistered estimates but must be described with that caveat. SR and IP–SR are not the gating criterion.

### 1.3 SR acquiescence (secondary reporting)

Primary SR scores for all hypotheses use keyed trait means (§1.1).

**Secondary (not used for H1–H5 pass/fail):** for each condition and trait, report an **acquiescence / raw-agreement index** = mean of `score_before_keying` over the 24 items (average of both label orders, before reverse-keying). This indexes tendency to endorse “Accurate” regardless of keying. Motivated by Phase 4 diagnostics; labeled secondary / method-variance, not a primary outcome.

### 1.4 Known risks carried from Phase 4 (not outcome claims)

- A_SR is acquiescence-prone on an at-ceiling baseline assistant.
- E_IP showed low between-persona SD on the 8-persona dev pilot (≈ 0.020); low variance may limit E IP validity on test.
- These are risks to note in the report, not grounds to alter the hypotheses below.

---

## 2. Exclusion rules

Applied at analysis time from item-level parquet (thresholds frozen in `configs/default.yaml`):

1. Drop an **item** for a condition if its option mass is below **0.50**.
2. Drop a **condition** from a measure if more than **20%** of its items for that measure are dropped.
3. Report all exclusions (counts and IDs) in the results / report.
4. Primary MTMM analyses use conditions that survive exclusions for the measures involved; any pairwise complete-case rule beyond that is documented in the analysis output.

---

## 3. Analysis specification (from plan §7; verbatim hypotheses)

Six variables per condition: `E_SR`, `E_BC`, `E_IP`, `A_SR`, `A_BC`, `A_IP`.  
Primary sample: all 96 personas. Secondary samples: explicit only (32) and implicit only (64). The baseline is descriptive only.

### 3.1 Preliminary checks (§7.1)

- **Manipulation check:** correlation of each score with the intended trait level of the persona.
- **Reliability:** Cronbach’s alpha for SR and BC scales (conditions as rows, items as columns). Split-half reliability with Spearman–Brown correction for IP across probes.

### 3.2 Hypotheses (§7.2)

- **H1 (convergent validity):** for each trait, the three same-trait, different-method correlations are greater than zero.
- **H2 (discriminant validity):** for each trait, each same-trait, different-method correlation exceeds the different-trait, different-method correlations involving the same variables. Also report the comparison against different-trait, same-method correlations.
- **H3 (key contrast, two-sided):** for each trait, r(IP, SR) differs from r(IP, BC).
- **H4 (specificity):** for each trait, r(IP, SR) and r(IP, BC) for the trait vector exceed the 95th percentile of the null distributions in §3.3.
- **H5 (secondary):** r(SR, BC) is lower in the implicit set than in the explicit set.

### 3.3 Methods (§7.3)

- Pearson correlations as primary, Spearman as a robustness check.
- 95% confidence intervals from **10,000** bootstrap resamples over conditions (seed **42**). For H2, H3, and H5, bootstrap the difference between correlations and test whether the interval excludes zero.
- Holm correction across the two traits for H3.
- **Null vectors (H4):** (a) 1,000 random unit vectors at L*; (b) 1,000 label-shuffled mean-difference vectors computed from the cached extraction activations. Report the trait vector’s percentile in each null.
- **Layer sweep:** recompute the IP correlations at every layer and plot them. Robustness check only; L* remains the preregistered layer.
- Report correlations corrected for unreliability as a secondary result.

Config mirrors: `analysis.bootstrap_n: 10000`, `analysis.bootstrap_seed: 42`, `analysis.null_vectors: 1000`, `analysis.alpha: 0.05`.

### 3.4 Outcome interpretation (§7.4)

| Pattern | Interpretation |
|---------|----------------|
| IP tracks BC more than SR | The internal direction relates to choice more than to self-description; questionnaires are the weaker instrument |
| IP tracks SR more than BC | The vector captures self-description more than disposition to act |
| All three converge and discriminate | The trait construct holds for this model under these conditions |
| IP tracks neither, or fails H4 | A mean-difference vector is not a valid trait measure here |
| Convergence in the explicit set only | Agreement is driven by explicit trait instructions, not by a shared construct |

When interpreting “IP tracks BC,” apply the §1.2 independence caveat.

### 3.5 Figures (§7.5)

1. MTMM heatmap with the validity diagonals marked (pooled, explicit, implicit).
2. Scatter plots of IP against SR and IP against BC for each trait, colored by persona set.
3. Layer sweep of the IP correlations.
4. Null distributions with the trait vector marked.
5. Steering curve from vector validation (gate v2 BC curves; SR secondary curves).
6. Held-out d′ by layer.

Any analysis beyond §3 is labeled **exploratory**.

---

## 4. Power (design note)

Two-sided α = .05, 80% power: minimum detectable correlation ≈ **.28** at N = 96, **.34** at N = 64, **.48** at N = 32. State this in the report.

---

## 5. What is frozen vs what is not

**Frozen at `prereg-v1`:** model revision; stimuli; L*; prompt formats; exclusion thresholds; hypotheses H1–H5; bootstrap/null settings; outcome table; gate v2 definition (historical validation already completed).

**Not yet run:** SR/BC/IP on the 96 test personas + baseline (Phase 6).  
**Not yet executed:** Phase 7 analysis on test results.  
**Must not happen before G1 approval:** any test-persona forward pass or peeking at test scores.

---

## 6. Provenance checklist for the main run

Every results write must record: git commit (this tag or a descendant that only adds results), model id + revision, library versions (torch, transformers, etc.), config path, and seeds. Item-level and condition-level parquet under `results/`.

---

## 7. HUMAN GATE G1

Owner: review this file and the tagged tree. Approve to unlock Phase 6 (main run). Request changes before approval if anything should differ; amendments after the tag go in `DEVIATIONS.md`.
