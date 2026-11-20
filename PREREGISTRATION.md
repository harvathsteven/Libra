# Preregistration — Trait MTMM Pilot (Libra)

**Version:** v2 draft (supersedes the `prereg-v1` draft, commit `d171e6e`)
**Date:** 2026-11-18
**Status:** Awaiting HUMAN GATE G1. On approval this commit is tagged `prereg-v2`. No test persona is run before then.

`prereg-v1` was tagged and pushed before review. It stays as a public record of the first draft. It was never used for any test run and is superseded by this version. `DEVIATIONS.md` counts departures from `prereg-v2`.

Dev-pilot numbers (N = 8) and extraction diagnostics informed the prompt format and the steering gate only. They do not inform hypotheses or decision rules. On the dev set, intended E and A levels correlate r = −0.57, so dev heterotrait values are confounded by design and are not evidence.

Items marked **[G1 decision]** need the owner's choice before approval. Each one shows the drafted default.

---

## 1. Frozen design

| Element | Decision |
|---------|----------|
| Traits | Extraversion (E), Agreeableness (A) |
| Model | `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`, float16, CUDA |
| Config | `configs/default.yaml`, global seed 42 |
| Test conditions | 96 personas (32 explicit, 64 implicit) + 1 no-persona baseline (`stimuli/personas_test.jsonl`) |
| Unit of analysis | Persona condition. Primary sample = the 96 personas; baseline is descriptive only |
| Secondary samples | Explicit only (32), implicit only (64) |
| SR | IPIP-NEO-120 E and A domains, 24 items each (`stimuli/ipip_neo_120_EA.csv`) |
| BC | 100 TRAIT items per trait (`stimuli/bc_items.jsonl`; `mirlab/TRAIT`, CC-BY-4.0) |
| IP | Cosine similarity between the last-prompt-token residual and the trait vector, mean over 50 probes (`stimuli/probe_questions.txt`) |
| Scoring | Next-token probabilities only; no sampling, no generation |
| Layer L* | **17**: index of the transformer block counted from 0 (excluding embeddings). The IP residual is the output of that block (`hidden_states[18]`); steering hooks the same block |
| Trait vectors | `results/extract/trait_vectors.pt` from the 2026-11-19 Kaggle re-run of `phase4` at `675f7f6`, which passed gate v2 and reproduced the 2026-11-17 run. SHA-256 `5e12b0ba04f6b2a7f09355e9156795629b689bdf0739a1ae2c44af185ca402ca`, pinned in `configs/default.yaml` (`vectors.sha256`). The main run refuses to start without a matching file. Vectors are never re-extracted for the main run. The matching `extraction_activations.pt` comes from the same passes |

### 1.1 Extraction and L* (completed in Phase 4; recorded here)

- Per trait and pole: 6 adjective triples (from the extraction half of the adjective lists) × 20 extraction questions = 120 prompts, 480 forward passes in total.
- Split, seed 42: 4 of 6 triples and 16 of 20 questions are drawn as train. A prompt is train only if both its triple and its question are train, giving 64 train and 56 holdout prompts per pole.
- Vector at each layer = normalized (mean high − mean low) of train last-token residuals.
- Held-out d′ is computed on cosine scores. L* maximizes mean d′ over E and A, with ties broken toward the middle layer. Separation gate: d′ ≥ 2 for each trait. Observed: E 7.11, A 15.37; cos(E, A) = 0.23 at L*.

### 1.2 Prompt formats (frozen; chosen on extraction and dev data only)

**SR** (`sr_format: letters`, `assistant_prefill: ""`): the user asks how accurately the statement describes the model. Five IPIP labels map to letters A–E with values 1–5. The model is told to answer with exactly one letter. Each item is scored in standard and reversed label order, the reversed expected value is mapped back, and the two are averaged. Reverse-keyed items then become 6 − x. Trait score = mean of the 24 keyed item scores. Digit labels were dropped after two attempts left about 0 next-token mass on the digits (`DECISIONS.md`).

**BC**: scenario plus four options labeled A–D, answered with one letter. Item score = P(high-trait letters) / P(all four letters). Each item is scored in one seeded option order (seed = 42 + sum of the item_id character codes + item index) and in its exact reverse, and the two are averaged. Trait score = mean of the 100 items.

**IP**: each of the 50 probes is run with the persona system prompt. At L*, the score is cos(h, v_T) for each trait vector v_T. Trait score = mean over probes. Last-token residuals at **all** layers are cached as float16 (`results/<run>/ip_residuals/<persona_id>.pt`) for the null distributions and the layer sweep.

### 1.3 Steering validation (gate v2; frozen; supersedes the SR steering criterion)

The baseline assistant was steered with coefficients {−0.2, −0.1, 0, +0.1, +0.2} × mean last-token ‖h‖ at L*. To pass, each trait needed Spearman(coefficient, mean BC high-trait probability over 100 items, both option orders) ≥ 0.9 (float tolerance 1e−9) and mean BC option mass ≥ 0.80. The owner fixed this rule before running it (`DECISIONS.md`, 2026-11-17). **Passed:** E and A ρ = 1.0, masses ≥ 0.98. Secondary result: A SR keying-balanced ρ ≈ −0.7 with raw agreement ρ = 1.0, i.e. acquiescence.

**Preregistered caveat:** the vectors were accepted partly because steering them moved BC. Monotrait IP–BC correlations are therefore not fully independent of vector validation. They are reported as preregistered estimates, but any statement that "IP tracks BC" must carry this caveat. SR played no part in the gate.

### 1.4 Known risks carried from Phase 4 (not outcome claims)

- A_SR is acquiescence-prone: forward-keyed items sit near ceiling on the baseline assistant.
- E_IP had low between-persona SD on the dev pilot (0.020), which may limit its correlations.
- The E and A vectors overlap (cos 0.23), so part of the E_IP–A_IP correlation is built in. This bears on the same-method comparisons in H2 (secondary part) and is noted in the report.
- **Implicit persona structure (stimulus finding, 2026-11-18).** The text of the 64 implicit personas encodes only their quadrant (E pole × A pole), through one of 3 cue sentences per quadrant. That gives 12 distinct cues, each shared by 5–6 personas, and the cue index also fixes the template. The "very" and "somewhat" labels in `persona_intended_levels.csv` were assigned by alternation (`scripts/build_stimuli.py`, `k % 2`) and do not appear in the text. Consequences:
  - the manipulation check for implicit personas uses pole coding (§3.1);
  - implicit conditions are not independent with respect to trait content, so their effective N is below 64 and the power figures in §4 are optimistic for the implicit and pooled samples.

  The explicit personas do encode graded levels ("very" / "somewhat" plus a trait adjective).

### 1.5 SR acquiescence index (secondary)

Per condition and trait: mean of `score_before_keying` over the 24 items (both label orders averaged, before reverse-keying). It is reported with its correlations with E_SR and A_SR as method variance. It is not used for any H1–H5 verdict.

---

## 2. Exclusions and missing data

Thresholds are in `configs/default.yaml` (`drop_item_option_mass: 0.50`, `drop_condition_item_frac: 0.20`).

1. **SR item:** dropped for a condition if its option mass, averaged over the two label orders, is below 0.50. The trait score is the mean of the remaining keyed items.
2. **BC item:** dropped for a condition if its option mass, averaged over the two option orders, is below 0.50.
3. **Condition score:** a condition's SR or BC trait score is set missing if more than 20% of that scale's items are dropped (SR: 5 or more of 24; BC: 21 or more of 100).
4. **IP:** no option mass, so no item exclusion. A non-finite score is missing.
5. **All hypothesis tests use listwise-complete conditions on all six variables**, within each sample (pooled, explicit, implicit). N and excluded IDs are reported per sample.

---

## 3. Analysis

Variables per condition: `E_SR`, `E_BC`, `E_IP`, `A_SR`, `A_BC`, `A_IP`. T denotes a trait and T′ the other trait.

### 3.0 General inference rules

- Pearson r is primary. Spearman ρ is a robustness check: all of H1–H5 are recomputed with it and reported in a parallel table. Primary verdicts do not change.
- **Bootstrap:** B = 10,000 resamples of conditions with replacement, `numpy.random.default_rng(42)`. Within a sample, the same resample index matrix is used for every statistic, so differences between correlations are paired. Resamples where a correlation is undefined (zero variance) are dropped and counted, and the count is reported if above 1%.
- A **95% interval** is the 2.5th–97.5th percentile interval. "Above zero" means its lower bound is > 0.
- **Bootstrap p** (H3 only): p = min(1, 2 · min(#{d* ≤ 0}, #{d* ≥ 0}) / B).
- Conjunctive hypotheses (H1, H2, H4) are supported for a trait only if every component passes. That is an intersection–union test, so components are not multiplicity-corrected. Each component is still reported on its own.
- **[G1 decision] Clustering of implicit personas (§1.4).** Draft default: the condition-level bootstrap above is primary. A **cluster bootstrap** is preregistered as a robustness check. In it, the 12 implicit cue sentences are clusters, each explicit persona is its own cluster, and clusters are resampled with replacement (B = 10,000, seed 42). The report gives each verdict under both bootstraps and flags any disagreement. Alternative: make the cluster bootstrap primary for the implicit and pooled samples.

### 3.1 Preliminary checks (descriptive, no pass/fail)

- **Manipulation check:** Pearson r between each score and the intended level of the same trait.
  - Explicit: intended level 1–4 (`intended_*_numeric`).
  - Implicit and pooled: pole coding (low = 0, high = 1), because only the pole is in the implicit text.
  - Cross-trait values (e.g. E scores against intended A) are also reported.
- **Reliability** (on the listwise sample of each sample):
  - SR and BC: Cronbach's α, with conditions as rows and items as columns (24 keyed SR items; 100 BC items).
  - IP: split-half reliability (odd vs even probe index means), Spearman–Brown corrected.

### 3.2 Hypotheses and decision rules

Hypothesis wording is from the plan's §7.2, as transcribed in `prereg-v1`. **The owner should check it against the brief**, which is not in the repository. Decision rules were added in this version.

- **H1 (convergent validity):** for each trait, the three same-trait, different-method correlations are greater than zero.
  *Rule:* for T in {E, A}, each of r(T_SR, T_BC), r(T_SR, T_IP), r(T_BC, T_IP) has a 95% interval above zero. H1 holds for T if all three do.

- **H2 (discriminant validity):** for each trait, each same-trait, different-method correlation exceeds the different-trait, different-method correlations involving the same variables. Also report the comparison against different-trait, same-method correlations.
  *Rule:* for each validity pair (Mi, Mj) of trait T, compute d = r(T_Mi, T_Mj) − |r(T_Mi, T′_Mj)| and d = r(T_Mi, T_Mj) − |r(T′_Mi, T_Mj)|. That is 3 pairs × 2 comparisons = 6 comparisons per trait. Each passes if its paired-bootstrap 95% interval is above zero, and H2 holds for T if all 6 pass. **[G1 decision]** Draft default: heterotrait values are taken as absolute values, so a strong negative heterotrait correlation does not count in favor of discriminant validity. Alternative: signed values, as in Campbell & Fiske (1959).
  *Secondary (reported, no verdict):* the same differences against the different-trait, same-method correlations r(T_Mi, T′_Mi) and r(T_Mj, T′_Mj). For IP, note the vector overlap from §1.4.

- **H3 (key contrast, two-sided):** for each trait, r(IP, SR) differs from r(IP, BC).
  *Rule:* d_T = r(T_IP, T_SR) − r(T_IP, T_BC) gets a paired bootstrap p and a 95% interval. Holm correction is applied over {E, A} at α = .05. The sign of d_T gives the direction. Apply the §1.3 caveat when interpreting.

- **H4 (specificity):** for each trait, r(IP, SR) and r(IP, BC) for the trait vector exceed the 95th percentile of the null distributions.
  *Nulls:* a null IP score replaces v_T with a null vector u: mean over probes of cos(h at L*, u), computed from the cached residuals on the same listwise sample.
  - (a) **Random:** 1,000 unit vectors u ~ N(0, I) normalized, `default_rng(42)`. The same set is used for both traits.
  - (b) **Label-shuffled:** for trait T, the 128 train extraction activations at L* (`results/extract/extraction_activations.pt`) have their high and low labels permuted, keeping 64 and 64. The vector is the normalized mean difference. 1,000 permutations, `default_rng(43)` for E and `default_rng(44)` for A.

  *Rule:* each of r(T_IP, T_SR) and r(T_IP, T_BC) must exceed the 95th percentile (`numpy.percentile`, linear) of the signed null correlations in both nulls. That is 4 comparisons per trait, and H4 holds for T if all 4 pass. The fraction of null r strictly below each target is reported.

- **H5 (secondary):** r(SR, BC) is lower in the implicit set than in the explicit set.
  *Rule:* per trait, Δ_T = r_explicit(T_SR, T_BC) − r_implicit(T_SR, T_BC). A stratified bootstrap resamples each set independently (B = 10,000, seed 42). H5 holds for T if the 95% interval of Δ_T is above zero. Secondary, so not corrected; both traits are reported.

### 3.3 Further preregistered analyses (no verdicts)

- **Disattenuated correlations:** r / √(rel_i · rel_j), using the §3.1 reliabilities. Not computed if either reliability is ≤ 0. Values above 1 are reported as they are, with a note.
- **Layer sweep:** for each layer ℓ = 0…31, IP scores are computed with `vectors_by_layer[T][ℓ]` and the cached residuals at ℓ, then correlated with T_SR and T_BC and plotted. This is a robustness check only; L* = 17 stays the preregistered layer.

### 3.4 Outcome interpretation

| Pattern | Interpretation |
|---------|----------------|
| IP tracks BC more than SR (H3 significant, d_T < 0) | The internal direction relates to choice more than to self-description; questionnaires are the weaker instrument. Subject to the §1.3 caveat |
| IP tracks SR more than BC (H3 significant, d_T > 0) | The vector captures self-description more than disposition to act |
| All three converge and discriminate (H1 and H2 hold) | The trait construct holds for this model under these conditions |
| IP tracks neither, or fails H4 | A mean-difference vector is not a valid trait measure here |
| Convergence in the explicit set only | Agreement is driven by explicit trait instructions, not by a shared construct |

### 3.5 Figures

1. MTMM heatmap with validity diagonals marked (pooled, explicit, implicit).
2. IP vs SR and IP vs BC scatter plots per trait, colored by persona set.
3. Layer sweep of the IP correlations.
4. Null distributions (both nulls) with the trait-vector correlation marked.
5. Steering curves from gate v2 (BC primary, SR secondary).
6. Held-out d′ by layer.

Any analysis not listed in §3 is labeled **exploratory** in the report.

---

## 4. Power

Two-sided α = .05 and 80% power give a minimum detectable correlation of about .28 at N = 96, .34 at N = 64, and .48 at N = 32. These figures assume independent conditions. §1.4 means the effective N for the implicit and pooled samples is lower, so they are optimistic there. The report states both.

---

## 5. Phase 6 and 7 procedure

- **Phase 6 (main run):** the `main` command runs the 96 test personas and the baseline through the same SR/BC/IP code path as the dev pilot (`cmd_pilot`), with residual caching on and `require_pin=True`. The only Phase 6 code change allowed is that wiring.
- **Phase 7 (analysis):** analysis code is written after this tag and must implement §2–§3 exactly. Any ambiguity or required change is logged in `DEVIATIONS.md` before the code that depends on it is run on test results.
- **Before G1 (done 2026-11-19; extraction set and dev personas only, no test data).** Re-running the Phase 4 workflow (`phase4`: extract with gates on → dev pilot → diagnose) on this commit writes `trait_vectors.pt` and `extraction_activations.pt` from the same passes, and the provenance records the vectors' SHA-256. That hash goes in `vectors.sha256`. Alternatively, `extract-acts` adds activations to an existing vectors file and checks that they match (cos ≥ 0.999 at L*).
- **Re-run guard:** the re-run is accepted only if both gates pass and the selected layer is still 17. If the re-run selects a different layer, or a gate fails, stop and report to the owner; L* = 17 stays frozen and nothing is re-tuned. Re-run numbers (d′, steering ρ, dev diagnostics) are logged in `DECISIONS.md` next to the 2026-11-17 values.

## 6. Provenance

Every results file records the git commit (`prereg-v2` or a descendant that only adds the Phase 6 wiring and results), model id and revision, library versions, config path, seeds, and the trait-vector SHA-256. Item-level and condition-level parquet go under `results/`.

## 7. HUMAN GATE G1

Owner, before approving:

1. Check the H1–H5 wording against the brief's §7.
2. Resolve the two **[G1 decision]** items: the clustering bootstrap (§3.0) and absolute vs signed heterotrait values (§3.2 H2).
3. Decide whether the implicit persona structure (§1.4) is acceptable as is, or whether the implicit personas should be rebuilt with graded cues before the test run. Rebuilding is a stimulus change; it is allowed now because no test data exists yet.
4. ~~Supply the vectors' SHA-256.~~ Done 2026-11-19: re-run passed the guard (L* = 17, both gates); hash pinned (§1).

On approval: commit, tag `prereg-v2`, then Phase 6.
