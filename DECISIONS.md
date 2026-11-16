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
