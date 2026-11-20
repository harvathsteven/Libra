# DEVIATIONS

Departures from the preregistration (`PREREGISTRATION.md`) after the `prereg-v2` tag (`prereg-v1` was a superseded draft).

## Departures

None yet.

## Clarifications (not departures)

Prereg §5 requires that any ambiguity in §2–§3 is logged here before the code that depends on it is run on test results. These were written with the Phase 7 code (`src/trait_mtmm/mtmm.py`), and committed before `phase7` was run on `results/main`. Each fixes a detail the prereg leaves open; none changes a rule.

- **D1 — Resample draws.**
  - The rows of each sample (pooled, explicit, implicit) are sorted by `persona_id`.
  - Condition-level bootstrap: one call `rng.integers(0, n, size=(B, n))`, with `rng = numpy.random.default_rng(42)`. Row b is resample b.
  - Cluster bootstrap: clusters are sorted by label, then one call `rng.integers(0, C, size=(B, C))` on a fresh `default_rng(42)`.
  - Each sample and each bootstrap kind starts its own `default_rng(42)`. Pearson and Spearman use the same resample matrices.
  - Resamples are computed as frequency weights, which is exactly equivalent to correlating the expanded resample. For explicit personas, the cluster bootstrap is therefore identical to the condition-level one, as §3.0 states.
- **D2 — Missing item rows.** A condition × trait with fewer item rows than the scale (24 SR, 100 BC) counts the missing rows as dropped items. No effect here: the main run's rows are complete.
- **D3 — Acquiescence index (§1.5).** It is the mean of `score_before_keying` over the items kept under §2. It is missing when that condition's SR score is missing.
- **D4 — Undefined resamples.** Percentile intervals and bootstrap p values use the defined resamples only, so the p denominator is the number of defined resamples. For a difference of correlations, a resample is undefined if any correlation in it is undefined. Counts are reported for every statistic.
- **D5 — Null vectors (H4).**
  - (a) Random: one call `rng.standard_normal((1000, d))` from `default_rng(42)`, with rows normalized.
  - (b) Label-shuffled: the train activations at L* are stacked high-pole first, then low-pole, in extraction order (64 + 64), and upcast to float64. Each of the 1,000 permutations is a successive `rng.permutation(labels)` from `default_rng(43)` for E and `default_rng(44)` for A.
  - Null IP is the probe mean of cos(h, u), computed as (probe mean of h/‖h‖) · u, which is algebraically identical. It comes from the float16 residual cache.
  - The target correlations use the run's own IP scores (`ip_probe`). The code refuses to run if the cache does not reproduce those scores to within 1e−3.
- **D6 — H5 draws.** Each bootstrap kind uses one `default_rng(42)`. The explicit resamples are drawn first (B × 32), then the implicit ones: B × 64 conditions, or B × 12 cue clusters for the cluster kind.
- **D7 — §3.4 outcome table, operational definitions.** These apply to the pooled sample and the Pearson verdicts:
  - "IP tracks BC (SR) more": H3 rejects after Holm, with d_T < 0 (or > 0).
  - "All three converge and discriminate": H1 and H2 both hold for T.
  - "IP tracks neither": neither r(T_IP, T_SR) nor r(T_IP, T_BC) passes its H1 component. The row also matches if H4 fails.
  - "Convergence in the explicit set only": H1 holds in the explicit sample but not in the implicit sample.
- **D8 — Cronbach's α with dropped items.** α uses only items with no missing value across the sample's listwise conditions, and the number of items used is reported. No effect here: no items were dropped.
- **D9 — Scope of the secondary analyses.**
  - Manipulation checks (Pearson), reliabilities and acquiescence correlations use each sample's listwise conditions.
  - H1–H4 are also computed in the explicit and implicit samples under the same rules, and reported as secondary. Verdicts are taken from the pooled sample.
  - Holm for H3 is applied separately within each sample and correlation method.
  - The Spearman robustness table includes H4, using Spearman null correlations.
- **D10 — H4 has no bootstrap.** Its verdict is therefore the same under both bootstrap schemes. "Exceed" means strictly greater than the 95th percentile.
- **D11 — H3 verdict.** "Holds" means that the Holm step-down at α = .05 rejects d_T = 0 for that trait. It uses p = max(p_condition, p_cluster). Intervals from both bootstraps are reported as well.
