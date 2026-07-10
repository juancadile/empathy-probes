# V2.1 Stimulus Suite — Design

**Created:** 2026-07-10 · Feeds Stage B5 (#27, confound) and B6 (#28, concept-vs-persona) of `ROADMAP.md`.
Generator: `src/data_generation/generate_v2_1_suite.py` · Cells: `data/eia_scenarios/v2_1_cells.json`.

## Why (the two problems this fixes)

1. **The task-cost confound is in the instructions, not just the scenarios.** The V1/V2 empathic prompt literally says *"even if it means sacrificing some task performance"*; the non-empathic one says *"rather than emotional considerations."* Empathy and task-sacrifice are coupled by construction, so the probe cannot distinguish an empathy direction from a task-focus direction.
2. **The extraction pipeline is persona-shaped.** Contrastive system prompts inducing a disposition is the persona-vectors method (Chen et al. 2025) — we may have extracted "playing an empathetic character," not empathy.

## Design principles

- **Decouple in the instructions**: no template ever mentions the trade-off. Empathic templates say "you genuinely care about the people you interact with" — nothing about cost. Task templates say "you are focused and efficient" — nothing about ignoring feelings. The *scenario structure* carries the cost manipulation.
- **Template paraphrases**: 3 instruction variants per side per cell, rotated — reduces template-leakage (probes learning the instruction's echo rather than the behavior).
- **Small models only**: `claude-haiku-4-5`, `gpt-4o-mini`, `gemini-2.5-flash` (~$5–10 total).
- **Lazar mapping** ([Cosmos essay](https://blog.cosmos-institute.org/p/the-construction-of-moral-character)): cells A–E measure *sensitivity* to morally relevant features under varying cost (local competence); F separates *analytical* from *practical* competence (recognizing vs enacting); R measures *robustness* to morally irrelevant changes; cross-cell generalization of the direction measures *coherence*.

## Cells

| Cell | Name | Contrast (pos vs neg) | Scenario structure | Answers |
|------|------|----------------------|--------------------|---------|
| **A** | empathy_at_cost | empathic vs task-focused response | helping costs task performance (reuses 5 EIA scenarios, **decoupled prompts**) | anchor / replication |
| **B** | empathy_no_cost | empathic vs task-focused | helping is **free or task-synergistic** (5 new scenarios) | task-focus confound |
| **C** | *(= neg side of A)* | — | — | — |
| **D** | no_task_social | warm-empathic vs polite-but-detached | **no task at all** — pure social contexts (5 new) | empathy without opportunity cost |
| **E** | task_focus_only | on-task vs distracted | distraction is **non-emotional** (trivia, formatting, small talk from a content bot) — empathy dimension absent (5 new) | isolates a task-focus direction to regress out |
| **F** | third_person | narrates someone *else* acting empathically vs efficiently | observer/reporter framing over cell-A scenarios | analytical competence; persona shouldn't fire |
| **G** | instrumental_empathy | genuine care vs **calculated care** — *same caring actions*, cold instrumental motive | trust-farming, sales comfort, reputation play (3 new) | content vs character: actions identical |
| **H** | constrained_coldness | caring character forced to act coldly vs cold character acting coldly — *same cold actions* | protocol/triage constraints (3 new) | character without content |
| **R** | robustness | paraphrase / name-swap / register perturbations of A pairs | post-hoc transform of cell A | probe invariance to morally irrelevant changes |
| **S** | severity_gradation | same scenario at mild / moderate / urgent need | graded triples, empathic response each | dose-response of moral salience |

## Prediction matrix (what each hypothesis expects the probe to do)

| Cell test | Empathy concept | Task-focus direction | Empathetic persona |
|-----------|----------------|---------------------|--------------------|
| B (pos vs neg) | separates | **silent** (no cost either way) | separates |
| D (pos vs neg) | separates | silent / undefined | separates |
| E (pos vs neg) | **silent** (no empathy dim) | separates | silent |
| F (pos vs neg) | separates (tracks content) | silent | **silent / weak** (model isn't *being* anyone) |
| G (pos vs neg) | **silent-ish** (same actions) | silent | separates (motive/character differs) |
| H (pos vs neg) | silent-ish (same actions) | silent | separates |
| R (within-set) | invariant | invariant | invariant |
| S (gradation) | monotone in severity | flat | flat-ish (persona constant) |

No single cell decides; the *pattern across cells* does. E.g. fires on B+D+F, silent on E+G → empathy concept. Fires on B+D+G+H, weak on F → persona. Silent on B, fires on E → V1 measured task-distraction.

## Counts (draft)

A: 90 pairs (5 scen × 3 models × 6) · B: 90 · D: 90 · E: 90 · F: 60 · G: 54 · H: 54 · R: 60 transforms · S: 45 triples ≈ **~630 items, ~1,900 API calls** on small models.

## Analysis plan (once generated)

1. Extract activations (existing pipeline) on Gemma-2-9B for all cells.
2. Probe trained on A only (V1 replication) → test on every other cell → fill the prediction matrix.
3. Extract a task-focus direction from E; compute cos(d_empathy, d_task); regress E-direction out of A-direction, re-test.
4. Persona-vector comparison (#28): Chen et al. pipeline for "empathetic assistant" on the same model → cosine + cross-steering.
5. Feed the winning interpretation into Stage B circuit labels and the paper's framing (Fork 2 in `ROADMAP.md`).
