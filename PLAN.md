# PLAN: Trait MTMM Pilot

A zero-cost construct-validity study of LLM personality measures, written as a work plan for an agentic coding model.

## 0. How to use this plan

You are an agent implementing an open-source research project for a human owner. Work through the phases in order. Each phase lists tasks, acceptance criteria, and in some cases a human gate.

Operating rules:

1. **Budget is $0.** Do not call paid APIs. Do not rent compute. Use CPU for development and a free 16 GB GPU (Kaggle or Colab free tier) for the main run.
2. **Never fabricate data or results.** Every number in the report must be produced by committed code from committed inputs. A null or messy result is an acceptable outcome and must be reported as it is.
3. **Never tune on the test set.** All prompt-format and layer choices are made on the dev personas and the extraction set only (see Phase 5). After the preregistration commit, the analysis is frozen.
4. **Log decisions.** Record every choice this plan leaves open, and every assumption in this plan that turns out to be wrong, in `DECISIONS.md` with a date and a one-line reason. Record any departure from the preregistration in `DEVIATIONS.md`.
5. **Stop at human gates.** Gates are marked **HUMAN GATE**. Summarize the state, list what you need, and wait.
6. **Verify before relying.** Model IDs, dataset IDs, licenses, and library APIs named here reflect the plan author's knowledge and a brief search in November 2026. Confirm each one before use. If one is unavailable, choose the closest substitute that meets the stated criteria and log it.
7. **Do not publish.** Do not push to a public remote, create a public repository, or post anywhere without the owner's approval.
8. Keep the owner's personal information out of the repository except for a name in `LICENSE` and `CITATION.cff`, which the owner supplies.

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
|---|---|
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
|---|---|
| Per condition: SR 48 items x 2 scale orders | 96 |
| Per condition: BC 200 items x 2 option orders | 400 |
| Per condition: IP 50 probes | 50 |
| Main run: 546 x 97 conditions | 52,962 |
| Dev pilot: 546 x 8 | 4,368 |
| Vector extraction | 480 |
| Steering validation | 960 |
| **Total** | **about 59,000** |

Target: main run under 2 GPU-hours on a 16 GB GPU. Hard cap for the whole project: 10 GPU-hours.

## 3. Repository layout

```
trait-mtmm/
  README.md
  PLAN.md                  # this file
  PREREGISTRATION.md
  DECISIONS.md
  DEVIATIONS.md
  REPORT.md
  LICENSE
  CITATION.cff
  pyproject.toml
  configs/
    default.yaml           # model id and revision, seeds, layer, paths
    smoke.yaml             # tiny model, tiny stimulus subset
  stimuli/
    adjectives.yaml        # extraction and test lists, disjoint
    personas_test.jsonl
    personas_dev.jsonl
    ipip_neo_120_EA.csv    # item text, trait, facet, keying
    bc_items.jsonl         # built by script if license forbids committing
    extraction_questions.txt
    probe_questions.txt
    banned_terms.txt
  src/trait_mtmm/
    model.py               # loading, chat formatting, hidden-state capture, hooks
    scoring.py             # option-token probability scoring
    measures/sr.py
    measures/bc.py
    measures/ip.py
    vectors.py             # extraction, validation, steering
    run.py                 # CLI: extract, pilot, main
    analysis.py            # MTMM, bootstrap, nulls, reliability
    figures.py
  tests/
  notebooks/
    run_kaggle.ipynb
  results/                 # parquet outputs, committed if small
  figures/
```

Engineering requirements:

- Python 3.11 or later, `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `scipy`, `pyarrow`, `matplotlib`, `pyyaml`, `pytest`. Use plain `transformers` with `output_hidden_states=True` and forward hooks. Do not add an interpretability framework unless plain hooks fail.
- All randomness seeded from the config. Record model revision hash, dataset revision, library versions, and git commit in every results file.
- Every pipeline stage is a CLI command, caches its output as parquet, and can resume.
- `pytest` must pass on CPU in under 5 minutes using `configs/smoke.yaml`.

## 4. Stimuli specification

### 4.1 Trait adjectives

Assemble at least 16 adjectives per pole per trait (high E, low E, high A, low A) from published Big Five adjective marker lists such as Goldberg (1992). Split each pole's list in half with a fixed seed into an **extraction list** and a **test list**. The two lists must share no word. A unit test enforces this.

### 4.2 SR items

Use the Extraversion and Agreeableness items of the IPIP-NEO-120 (Johnson, 2014): 6 facets per domain, 4 items per facet, 24 items per domain. The instrument is public domain; take item text and keying from ipip.ori.org and cite it. Store text, trait, facet, and keying direction.

### 4.3 BC items

Primary source: TRAIT (`mirlab/TRAIT` on Hugging Face; paper arXiv:2406.14703). Each item has a scenario question with two high-trait and two low-trait response options. The dataset is gated and requires accepting terms with a Hugging Face account, so this needs the owner (see Gate G0).

- Sample 100 Extraversion and 100 Agreeableness items with a fixed seed.
- Check the license. If redistribution is not permitted, commit only item identifiers and a build script.

Fallback if TRAIT is unavailable or its terms are incompatible: author 60 items per trait yourself. Each item is a concrete everyday scenario with two high-trait and two low-trait actions, spread evenly over the six IPIP-NEO facets of the trait, containing no banned terms (4.6). Log this as a major limitation, because agent-written items have no external validation.

### 4.4 Personas

All personas are system prompts written in the second person.

**Explicit set (32).** 4 levels of E by 4 levels of A by 2 wording variants. Levels are very low, somewhat low, somewhat high, very high. Build each prompt from the **test** adjective list only, for example: "You are a person who is very {E adjective} and somewhat {A adjective}."

**Implicit set (64).** 4 quadrants (high or low E by high or low A) by 16 personas. Each is a 60 to 90 word biographical sketch: occupation, age range, daily routines, and concrete habits that imply the intended quadrant. Vary occupation, age, and setting widely within each quadrant. No banned terms (4.6).

**Baseline (1).** No persona. Use the model's default system prompt or none.

**Dev set (8).** 2 explicit and 6 implicit personas built the same way and not included in the test set.

Store an intended E level and A level with each persona for the manipulation check.

### 4.5 Extraction and probe prompts

- `extraction_questions.txt`: 20 neutral user questions unrelated to social behavior (for example, how to store bread, how tides work).
- `probe_questions.txt`: 50 different neutral user questions. No overlap with the extraction questions. A unit test enforces this.
- Extraction system prompts: for each trait and pole, 6 prompts of the form "You are {adj1}, {adj2}, and {adj3}." using the **extraction** adjective list only. This gives 6 x 20 = 120 prompts per pole per trait.

### 4.6 Banned terms

`banned_terms.txt` contains every adjective in both lists, the trait names and their common derivatives (extravert, introvert, agreeable, and similar), and distinctive content phrases from the IPIP items. A lint test fails if any implicit persona, probe question, or fallback BC item contains a banned term. The purpose is to prevent the three measures from correlating only because they share words.

## 5. Measurement specification

### 5.1 Option-probability scoring (shared primitive)

Given a chat-formatted prompt ending at the point where the assistant's reply begins, take the next-token distribution. For each valid option (digits 1 to 5, or letters A to D), sum the probability over all token IDs whose decoded, stripped text equals the option. Renormalize across the valid options.

Also record **option mass**: the total unnormalized probability on valid options. Requirement: mean option mass of at least .80 on the dev personas for both SR and BC. If it is lower, change the prompt format (for example, add an assistant prefill such as "Answer:") and re-measure on dev only. Log the final format.

If the model's chat template has no system role, prepend the persona to the user turn and log it.

### 5.2 SR score

Prompt: persona, then a user turn asking how accurately the statement describes the model, with the five IPIP response labels numbered 1 to 5 and an instruction to answer with a single number.

- Item score = expected value of the rating under the renormalized distribution.
- Run each item twice: standard scale order and reversed scale order (1 = Very Accurate). Map the reversed version back and average the two.
- Reverse-keyed items: 6 minus the score.
- Trait score per condition = mean over the 24 items.

### 5.3 BC score

Prompt: persona, then a user turn with the scenario, four options labeled A to D, and an instruction to answer with a single letter.

- Item score = probability on the two high-trait options divided by probability on all four.
- Run each item twice: one seeded random option order and its exact reverse. Average the two.
- Trait score per condition = mean over the 100 items.

### 5.4 Trait vectors

1. Run the extraction prompts (4.5). Capture the residual stream at every layer at the last prompt token, which is the position where the first reply token is predicted.
2. Split the extraction set 80/20 by adjective triple and by question, with a fixed seed.
3. For each layer and trait: vector = mean(high pole) minus mean(low pole) on the training split, normalized to unit length.
4. On the held-out split, compute d-prime of the projections between poles for each layer and trait.
5. Choose a single layer **L\*** that maximizes the mean d-prime across both traits. On ties, prefer the layer closest to the middle of the network. Fix L\* in the config before the main run.
6. Report the cosine similarity between the E and A vectors at L\*. Non-orthogonality is expected and is a finding, not an error.

Validation, both required before continuing:

- **Separation:** held-out d-prime at L\* of at least 2.0 for each trait.
- **Steering:** on the baseline condition, add coefficient x vector at L\* at all positions for 5 coefficients symmetric around zero. Choose the coefficient scale so option mass stays at or above .80. The target-trait SR score must increase monotonically with the coefficient (Spearman correlation of at least .9). Report the off-target trait shift as well.

If either check fails, stop and report. Do not proceed with a vector that fails.

### 5.5 IP score

For each condition, run the 50 probe prompts with the persona. At L\*, project the last-prompt-token residual onto each unit trait vector. Trait score per condition = mean projection over the 50 probes. Cache last-token residuals for all layers as float16 for the layer sweep and the null analysis.

Secondary variant: projections computed on the SR and BC prompts themselves. Report it separately, because it shares stimuli with the other two methods.

## 6. Phases

### Phase 0: Verification and setup

Tasks:

1. Select the model. Criteria: open weights; 2B to 4B parameters; chat or instruct tuned; license allows research use and publishing results; loads in half precision within 16 GB; exposes hidden states through `transformers`. Candidates to check first: Phi-3.5-mini-instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Gemma 2 2B instruct, or a current equivalent. Prefer an ungated model. Pin the revision hash.
2. Select a tiny chat model for CPU smoke tests (for example SmolLM2-135M-Instruct, if it exists and has a chat template).
3. Confirm TRAIT access terms and field names. Confirm the IPIP-NEO-120 item source.
4. Verify the references in Section 9 exist and say what this plan claims. Correct any that do not.
5. Scaffold the repository and `DECISIONS.md`.

Acceptance: `DECISIONS.md` lists model, revision, licenses, and dataset status. The smoke model loads and returns hidden states on CPU.

**HUMAN GATE G0.** Owner confirms the model choice, accepts the TRAIT terms and supplies a Hugging Face token (or approves the fallback in 4.3), and chooses the code license and repository name.

### Phase 1: Core library

Tasks: implement `model.py`, `scoring.py`, and hook-based residual capture and addition. Write tests for chat formatting, option-token discovery, renormalization, reverse keying, option-order averaging, hook correctness (a zero steering vector changes nothing), and determinism (two runs give identical output).

Acceptance: tests pass on CPU with the smoke model.

### Phase 2: Stimuli

Tasks: build everything in Section 4. Write lint tests for adjective-list disjointness, question-set disjointness, banned terms, persona counts, and word limits.

Acceptance: all lint tests pass. A table of intended trait levels by persona is committed.

### Phase 3: Measures and vectors

Tasks: implement SR, BC, IP, vector extraction, and both validations (5.4). Run the full pipeline end to end on the smoke configuration.

Acceptance: the smoke run produces a complete results table and all figures without error. The numbers from the smoke model are meaningless and must not be reported.

### Phase 4: GPU extraction and dev pilot

Requires the real model on a GPU. If you have no GPU, prepare `notebooks/run_kaggle.ipynb` to clone the repository, install dependencies, run a named CLI stage, and save outputs, then hand it to the owner.

Tasks:

1. Extract and validate vectors. Fix L\*.
2. Run the 8 dev personas through SR, BC, and IP.
3. Check option mass (5.1), floor and ceiling effects, and that scores vary across dev personas. Adjust prompt formats only, on dev only, and log every change.

Acceptance: both vector validations pass; option mass at least .80; no trait score is pinned at a scale endpoint for more than half of the dev personas.

Stop and report if a validation fails after two reasonable prompt-format attempts.

### Phase 5: Preregistration

Write `PREREGISTRATION.md` containing: the frozen design, the hypotheses and analyses from Section 7 verbatim, the chosen L\*, the final prompt formats, exclusion rules, and the outcome interpretation table (7.4). Commit it, with all code and stimuli, in a single tagged commit (`prereg-v1`) before any test persona is run.

**HUMAN GATE G1.** Owner reviews and approves the preregistration.

### Phase 6: Main run

Run all 97 conditions through SR, BC, and IP from the tagged commit. Save item-level and condition-level parquet files with full provenance.

Exclusion rule: drop an item for a condition if its option mass is below .50. Drop a condition from a measure if more than 20 percent of its items are dropped. Report all exclusions.

**HUMAN GATE G2** applies if you have no GPU: the owner runs the notebook and returns the results files.

### Phase 7: Analysis

Run Section 7 exactly as preregistered. Produce the figures in 7.5. Any additional analysis is labeled exploratory.

### Phase 8: Write-up

Deliver `README.md` and `REPORT.md` (Section 8). Run a clean-clone reproduction: fresh environment, analysis from committed results, identical numbers.

**HUMAN GATE G3.** Owner reviews everything before any public release.

## 7. Analysis specification

Six variables per condition: E_SR, E_BC, E_IP, A_SR, A_BC, A_IP. Primary sample: all 96 personas. Secondary samples: explicit only (32) and implicit only (64). The baseline is descriptive only.

### 7.1 Preliminary checks

- **Manipulation check:** correlation of each score with the intended trait level of the persona.
- **Reliability:** Cronbach's alpha for SR and BC scales (conditions as rows, items as columns). Split-half reliability with Spearman-Brown correction for IP across probes.

### 7.2 Hypotheses

- **H1 (convergent validity):** for each trait, the three same-trait, different-method correlations are greater than zero.
- **H2 (discriminant validity):** for each trait, each same-trait, different-method correlation exceeds the different-trait, different-method correlations involving the same variables. Also report the comparison against different-trait, same-method correlations.
- **H3 (key contrast, two-sided):** for each trait, r(IP, SR) differs from r(IP, BC).
- **H4 (specificity):** for each trait, r(IP, SR) and r(IP, BC) for the trait vector exceed the 95th percentile of the null distributions in 7.3.
- **H5 (secondary):** r(SR, BC) is lower in the implicit set than in the explicit set.

### 7.3 Methods

- Pearson correlations as primary, Spearman as a robustness check.
- 95 percent confidence intervals from 10,000 bootstrap resamples over conditions (seeded). For H2, H3, and H5, bootstrap the difference between correlations and test whether the interval excludes zero.
- Holm correction across the two traits for H3.
- **Null vectors (H4):** (a) 1,000 random unit vectors at L\*; (b) 1,000 label-shuffled mean-difference vectors computed from the cached extraction activations. Report the trait vector's percentile in each null.
- **Layer sweep:** recompute the IP correlations at every layer and plot them. This is a robustness check; L\* remains the preregistered layer.
- Report correlations corrected for unreliability as a secondary result.

### 7.4 Outcome interpretation (fix in the preregistration)

| Pattern | Interpretation |
|---|---|
| IP tracks BC more than SR | The internal direction relates to choice more than to self-description; questionnaires are the weaker instrument |
| IP tracks SR more than BC | The vector captures self-description more than disposition to act |
| All three converge and discriminate | The trait construct holds for this model under these conditions |
| IP tracks neither, or fails H4 | A mean-difference vector is not a valid trait measure here |
| Convergence in the explicit set only | Agreement is driven by explicit trait instructions, not by a shared construct |

### 7.5 Figures

1. MTMM heatmap with the validity diagonals marked (pooled, explicit, implicit).
2. Scatter plots of IP against SR and IP against BC for each trait, colored by persona set.
3. Layer sweep of the IP correlations.
4. Null distributions with the trait vector marked.
5. Steering curve from 5.4.
6. Held-out d-prime by layer.

## 8. Deliverables

- **README.md:** the question, the design in one diagram or table, the main result in three sentences, how to reproduce, limitations, citation.
- **REPORT.md:** 2,000 to 3,500 words. Sections: motivation and related work, method, results for each hypothesis, exploratory results, limitations, next steps. Report every preregistered analysis, including failures.
- Required limitations: one small model; forced choice measures stated intention, not behavior; personas are designed conditions, not independent individuals; overlap in content between personas and items may inflate convergence despite the banned-terms lint; a single linear direction at one token position is one of many possible internal measures; BC item provenance (and the fallback, if used).
- Required next steps: more models and sizes, open-ended behavior with human ratings, all five traits, confirmatory factor analysis with method factors.

## 9. Related work to verify and cite

Confirm each reference before citing it.

- Serapio-Garcia et al., "Personality Traits in Large Language Models," arXiv:2307.00184. Uses MTMM across two questionnaires.
- Lee et al., "TRAIT," arXiv:2406.14703, Findings of NAACL 2025.
- Han et al., "The Personality Illusion," arXiv:2509.03730. Self-report and behavior dissociate.
- arXiv:2606.09843. Self-report does not predict human-rated behavior.
- "PERSONA," arXiv:2602.15669, ICLR 2026. Trait vectors by contrastive activation analysis.
- arXiv:2602.15847. Big Five steering vectors are geometrically entangled.
- Lu et al., "The Assistant Axis," arXiv:2601.10387.
- "Beyond Static Personas" (IRIS), arXiv:2604.13846, Findings of ACL 2026.
- Zeng et al., persona-vector auditing, arXiv:2607.13162.
- Campbell and Fiske (1959), multitrait-multimethod matrix.
- Johnson (2014), IPIP-NEO-120. Goldberg (1992), Big Five markers.

Also run a fresh literature search in Phase 0 for any study that already combines self-report, behavioral, and activation-based measures in one MTMM analysis. If one exists, log it, tell the owner at Gate G0, and position this project as a replication on a different model.

## 10. Definition of done

- [ ] All human gates passed and recorded in `DECISIONS.md`.
- [ ] `prereg-v1` tag exists and predates all test-set results.
- [ ] Vector validations passed, or the project stopped with a written report explaining the failure.
- [ ] Main run complete for 97 conditions with provenance metadata.
- [ ] Every hypothesis in 7.2 has a reported estimate, interval, and verdict.
- [ ] Six figures generated by committed code.
- [ ] `pytest` passes on CPU. Clean-clone reproduction of the analysis matches.
- [ ] `README.md`, `REPORT.md`, `DEVIATIONS.md` complete. Total spend: $0.

## Project status (added during execution)

The full original plan was restored on 2026-11-20. Earlier repo copies had only a placeholder for §4–10.

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
| 8 Report | Done (2026-11-20): `REPORT.md` released; G3 passed |

**Project status: complete for now (2026-11-20, owner decision).** Deferred against §5.4, §5.5 and §10: off-target steering shift, the secondary IP variant on SR/BC prompts, and a clean-clone reproduction from committed results. See `DECISIONS.md`.
