# Integrity Repair A — Implementation Memo

**Date:** 2026-07-13
**Brief:** `notes/FABLE_INTEGRITY_REPAIR_A_2026-07-13.md` (Gate 0A, issue #35)
**Scope executed:** code/schema/test repairs only. No GPU, API, Spark, paper,
roadmap, or raw-result edits. No reruns were started.

## A1. T/T-confirm pseudoreplication repair

**Defect (science audit):** `build_cell_t.py` and `build_cell_t_confirm.py`
iterated eight indices while cycling `OPENERS[index % 2]` and
`CLOSINGS[index % 4]` in lockstep; lcm(2,4)=4, so indices 4–7 repeated 0–3.
Both artifacts had 40 rows but only 20 unique `(pos_text, neg_text)` pairs.

**Repair:**

- Both builders now emit the explicit `itertools.product` of opener x closing
  (2 x 4 = 8 unique variants per family; stimulus TEXT unchanged — only the
  combination structure changed). Rows carry stable `opener_variant`,
  `closing_variant`, and `pair_index = opener_variant*4 + closing_variant`.
- New shared module `src/data_generation/builder_integrity.py`:
  `assert_unique_pairs`, `assert_variant_grid` (full Cartesian grid per
  family), `assert_disjoint_families` (dev/confirm cross-check at write
  time), and `write_jsonl_guarded` (validate → preserve any differing
  existing artifact → write → `*.provenance.json` sidecar). Preservation
  records are append-only: same-content re-preservation verifies, differing
  content at a preserved path raises. A builder can no longer silently
  overwrite a preserved artifact or its provenance record.
- Rows are timestamp-free, so regeneration is byte-deterministic; volatile
  facts live in the sidecars.

**Preserved historical artifacts (committed, byte-identical to prior git
HEAD versions):**

| artifact | preserved copy | sha256 | structure |
|---|---|---|---|
| T_templated.jsonl (lockstep) | `data/contrastive_pairs/v2_1/historical/T_templated.jsonl.511a2bac/` | `511a2bac3359a858b5c7538deca28b72c1fdfe8a845e3025d98f0c5aec95b525` | 40 rows / 20 unique pairs |
| T_confirm_templated.jsonl (lockstep) | `data/contrastive_pairs/v2_1/historical/T_confirm_templated.jsonl.c2740918/` | `c274091806e1c9fbccbf547971692e3f8460293da3686069f1cdef1fe79d050c` | 40 rows / 20 unique pairs |

**Regenerated artifacts:**

| artifact | sha256 | structure |
|---|---|---|
| `data/contrastive_pairs/v2_1/T_templated.jsonl` | `bb7ca413dcaccde65e2a6de22dedece43e9cf80b61c23830710f23201494a12e` | 40 rows / 40 unique pairs, 5 families (t_*), full 2x4 grid |
| `data/contrastive_pairs/v2_1/T_confirm_templated.jsonl` | `33bed6a287d71e6515838c94027ec5a754e5bd2fb86970b29ca4c966225300e4` | 40 rows / 40 unique pairs, 5 families (tc_*), full 2x4 grid |

Development/confirmation family disjointness verified (t_* ∩ tc_* = ∅) and
asserted at every future write.

**No result-equivalence claim is made.** Per `GATE0B_PREREG_2026-07-13.md`,
historical results remain valid for the historical weighted stimulus set;
the repaired controls must be re-scored before any accepted control claim
references them.

### Model runs the changed T artifacts invalidate or require

Every accepted artifact whose T rows came from the 20-unique-pair files can
no longer certify the repaired control. Gate-0B (or later gates) must rerun
the T-dependent portion:

1. **Repaired T/T-confirm scoring (REQUIRED, Gate 0B part 1):** baseline +
   writer + suppressor edits on repaired T-confirm (with simultaneous
   M-confirm), dual A/B order, raw + continuation readouts, chat secondary —
   protocol frozen in `GATE0B_PREREG_2026-07-13.md`.
2. `results/e25_stage3_resid_gemma/stage3.json` — T_confirm selectivity
   ratios (writers 20.2x etc.) used the duplicated T_confirm.
3. `results/e26_format_stress_gemma/e26.json` — all T control rows
   (raw/chat/continuation) used the duplicated T_confirm.
4. `results/norm_matched_resid_gemma_v2/` (corrected R1) — T_confirm feeds
   the joint selectivity statistic `|dM| - 3|dT|`.
5. `results/controlled_directions_gemma2_9b_it/e25_heldout_cert.json` —
   the held-out T-quietness FAIL (AUROC 0.210) was measured on duplicated
   T_confirm rows; re-certify on the repaired file (the fail verdict is not
   assumed to flip).
6. `results/e17b_null_audit_llama31_8b_it/e17b.json`,
   `results/e17b_frac_{gemma,llama}/e17b.json` — T_confirm control rows in
   test1/test2/test3; the Gemma fractional rerun is already a Gate-0B item
   for the `d_resid` reason as well.
7. Historical/method-history only (no rerun required for current claims):
   E17 Llama stage-3, E14c retro-eval, E13-era pilot runs on development
   T_templated, LB1/LB2 lens batteries using T_confirm contrast cells
   (exploratory), and all pre-correction results already superseded by the
   E24 rescue.
8. **Derivation note:** `d_resid` was residualized against activations of
   the historical development T cell (among others). The preserved
   historical file keeps that derivation exactly reproducible; the direction
   of record is NOT retroactively invalidated, but any future re-derivation
   will consume the repaired T.

## A2. Guardrails and result schemas

### Environment and model revisions

- New `src/utils/run_provenance.py`:
  - `collect_run_provenance()` — embeddable provenance block: Python/
    platform, CUDA/device (recorded-absent when torch/CUDA missing), git
    commit/branch/dirty state, exact versions of torch, transformers,
    numpy, scikit-learn, datasets, transformer-lens, sae-lens (+ hub stack);
    absent packages recorded as `null`, never import failures.
  - `resolve_hf_commit` / `resolve_model_and_tokenizer` — model and
    tokenizer commit hashes resolved SEPARATELY from the local HF cache
    (refs/snapshots parse; no network).
  - Deterministic environment-lock schema (`empathy-action-probes/env-lock/1`)
    with pure `build/serialize/parse` functions and CLI:
    `python -m src.utils.run_provenance export|validate` (sorted-key
    deterministic serialization; `--omit-timestamp` for byte determinism).
  - The audit's expected Spark reference values (Gemma cache
    `11c9b309abf7…`, Llama `0e9e39f249a1…`, Python 3.12.13, Torch
    2.13.0+cu130, Transformers 5.13.0, TransformerLens 3.5.1, SAELens
    6.45.3, Datasets 5.0.0) are documented in the module docstring as
    values to VERIFY at capture time, not hard-coded facts.
  - **The actual Spark `empathy` lock capture remains a Gate-0B pre-run
    action** — batch A did not contact the Spark.
- `requirements.txt` now carries an explicit header labeling it broad
  developer constraints, not an environment lock.
- `weight_orthogonalization.py`, `capability_eval.py`, `e17b_null_audit.py`
  gained `--revision` / `--tokenizer-revision` passed through to
  `from_pretrained` and persisted; accepted reruns must pin these after
  resolving cached commits.

### Weight edits (stale component defaults)

- The superseded pre-correction sets (`TARGETED/RANDOM/POSITIVE_WRITERS/
  SUPPRESSORS`) are REMOVED as module constants/CLI defaults from
  `weight_orthogonalization.py`.
- New `src/component_sets.py`: immutable, versioned set-of-record registry —
  `gemma2_9b_it_precorrection_2026-07-11` (superseded, byte-exact historical
  values), `gemma2_9b_it_resid_2026-07-12` (current: writers L19MLP,L20MLP;
  suppressors L18H13,L20H10,L19H12,L17H7; composition-matched random
  L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2), and
  `llama31_8b_it_grouped_2026-07-12` (E17 Llama sets, moved out of
  e17_stage3/e17b module defaults).
- `resolve_component_sets()` enforces explicitness: every run names a
  registry key (`--component-set`) or passes full literal specs; anything
  else is a hard CLI error. Registry/model mismatches abort; superseded
  keys resolve but carry a persisted warning.
- Updated callers (import compatibility preserved; every result now
  persists the full resolution — registry key, per-role origins, resolved
  specs): `weight_orthogonalization.py`, `capability_eval.py`,
  `norm_matched_controls.py`, `e17_stage3.py`, `e17b_null_audit.py`.
  (`e18_interaction.py`/`e26_matched_nulls.py` already required explicit
  sets; `e14d_random_component_sets.py` is a completed old-set experiment
  whose internal constant documents what it tested.)
- `norm_matched_controls.py --reference` no longer defaults to the stale
  pre-correction pilot `analysis.json` (default now empty = in-run deltas).

### Fractional ablation (`e17b_null_audit.py`)

`e17b.json` is now self-contained for independent recomputation:

- direction path + SHA-256; per-cell path/SHA-256/row-count/families;
- exact baseline per-pair scores and A/B option order (flip signs) for
  test-1 raw/chat baselines and the shared test-2/3 raw baseline;
- per-pair edited scores and per-pair deltas for every condition, ablation
  block, and fraction (legacy `mean`/`ci95` keys unchanged);
- random-direction control seeds;
- config block: model, resolved revisions, dtype, attn implementation,
  seed, batch size, max tokens, tokenizer settings, prompt formats;
- full environment provenance block.

Every historically reported percentage (e.g. the 88%/5.5% baseline-margin
ratios) is reconstructible from one file: `fraction_f.cell.mean /
baseline_raw.cell.mean`, and per-pair arrays reproduce both numerator and
denominator. Backward compatibility: existing summary keys are unchanged;
`test1_*.baseline.<cell>` changed from a bare float to
`{mean, per_pair_scores, option_flips}` (audit-required; the only in-repo
consumer, `paper_v2_figures.py`, reads historical files and only the
untouched `fraction_*`/`random_dir_*` `mean`/`ci95` keys).

### Capability evaluation (`capability_eval.py`)

- **Documentation/implementation mismatch resolved in favor of stratified
  sampling:** `sample_mmlu` now performs deterministic subject-stratified
  sampling (largest-remainder allocation over the subject distribution;
  per-subject rng seeded by `(seed, subject)`). The accepted 2026-07-12 run
  (`results/capability_eval_resid_gemma/capability_eval.json`) used SIMPLE
  RANDOM sampling despite the old docstring; that artifact stands as-is and
  its wording anywhere downstream must say simple random sampling.
- Result JSON now persists: MMLU dataset name/config/split/revision/
  fingerprint, sampling method/seed/subject allocation, sampled row ids,
  subjects, question+choices SHA-256 hashes, target letters; per condition,
  per-item generated completion, predicted letter, parser status, and
  correctness (legacy `mmlu_per_item` boolean list kept); WikiText dataset
  info, char/window config, text hash, and per-window NLL sums; resolved
  component sets; model/tokenizer revisions; environment provenance.
- `--mmlu-revision` / `--wikitext-revision` flags for pinned dataset
  revisions (recorded; accepted reruns must pin).
- **No model was run in this batch.**

## A3. Judge and manipulation-check provenance

### E27 scorer (`src/eia_validation/score_e27.py`)

Rewritten around the Gate-0B adjudication protocol; no API call was made:

- `--judge-model` REQUIRED for live judging (floating-alias default
  removed); judge model, versioned exact prompt, temperature, max tokens,
  attempts, and API metadata persisted in the artifact.
- Prompt registry `JUDGE_PROMPTS`: `e27_say_classifier_v1_2026-07-12`
  preserves the historical protocol verbatim (400-char truncation, last
  user message) as the record of how `e27_scores.json` was produced;
  `e27_say_classifier_v2_2026-07-13` uses full prior user transcript + full
  message.
- Deterministic input ids: `sha256(history_sha | seed | step | msg_sha)`
  exactly as the Gate-0B prereg specifies; content-derived by design (test
  documents that byte-identical histories share ids while detail rows stay
  distinguishable by file).
- Per-event persistence: input id, file, condition/variant/seed
  (bookkeeping only), step, say index, FULL message, FULL prior user
  context, rendered judge input, presentation index, and per-attempt
  request metadata, HTTP status, raw response JSON (or raw text on JSON
  failure), response id/usage/stop reason, parse status, error — plus the
  final label.
- Condition blindness preserved and tested: the judge input renders from
  prior user context + player message only (byte-exact template equality).
- Failures are never coerced: exhausted attempts yield explicit
  `UNKNOWN` + `error_or_parse_failure` with full history; the run manifest
  registers zero-say runs; an incomplete condition x variant x seed grid
  aborts; nonzero exit when any UNKNOWN remains.
- Randomized judging order (`--shuffle-seed`, persisted); `--dry-run`
  exports the exact judge inputs offline (validated against the real tree:
  48 runs, 356 events, matching the historical scorer's counts and the 48
  raw-history hashes the audit asked to persist).
- Output-path guards: never overwrites an existing file; writing to the
  historical `e27_scores.json` is hard-refused.
- **The three UNKNOWN rows were NOT adjudicated** (Gate 0B).

### V2.2 manipulation pretests (`src/evaluation/manipulation_pretest.py`)

New reusable runner covering both audit batteries:

- `need_v2_2` (urgent/mild/resolved/excited from the four axis JSONLs;
  cost stem + closing pinned; item = family x ack variant → 20 items/arm,
  matching the historical n=20) and `moral_v2_2` (lower/equal/higher from
  `relative_need`; two questions: `need_now`, `respond_now`).
- Randomized blinded presentation (seeded, persisted order; judge sees
  rated text + versioned question/scale only); exact item/family/arm ids
  and full rated text persisted; source files hashed.
- Versioned exact prompts and rating scales (`PRETEST_PROMPTS`);
  configurable pinned judge model (required for live runs); strict
  whole-response integer parse; per-attempt raw response/retry/error
  persistence; UNKNOWN never coerced; item-level ratings plus family/arm
  summaries.
- `--dry-run` export mode requires no API access.
- Historical `need_pretest.json` / `moral_pretest.json` are untouched and
  hard-protected as output paths.

## Tests

`pytest -q tests` → **118 passed** (19 pre-existing + 99 new). New files:

- `tests/test_builder_integrity.py` — the FORMER lockstep construction
  fails the new uniqueness and grid asserts (acceptance criterion); the
  repaired builders emit 40/40 with the full grid; committed artifacts are
  byte-identical to `build_rows()`; dev/confirm disjointness; guarded write
  creates/verifies/preserves; preservation records are append-only and
  tamper-detected.
- `tests/test_run_provenance.py` — lock build/serialize/parse round-trip,
  deterministic serialization, schema rejection cases, absent-package
  recording, HF cache refs parsing, separate model/tokenizer resolution,
  CLI export/validate, file-hash provenance.
- `tests/test_component_sets.py` — registry well-formedness and exact
  sets-of-record values; resolution requires explicitness; registry/
  override/mismatch/superseded-warning paths.
- `tests/test_capability_sampling.py` — largest-remainder allocation
  (exact n, proportionality, caps, oversampling rejection), deterministic
  stratified sampling, parser status.
- `tests/test_e17b_schema.py` — legacy summary shape unchanged; per-pair
  detail reconstructs every summary number; cell provenance self-contained.
- `tests/test_score_e27.py` — deterministic ids + context replay (player
  say precedes same-step tick), content-derived id property, condition
  blindness (byte-exact), v1 truncation vs v2, strict label parse, judge
  retry/error persistence (HTTP failure, non-JSON, exception, custom
  parser), aggregation (UNKNOWN separation, grid abort, zero-say
  registration, hand-computed interaction), output guards, offline dry-run
  end-to-end, pinned-judge requirement.
- `tests/test_manipulation_pretest.py` — extraction matches the historical
  design (80/60 items, 20/arm), determinism, ambiguous-pinning rejection,
  blind rendering, strict rating parse, no wasted API calls on valid
  ratings, UNKNOWN persistence, summaries (incl. all-UNKNOWN arms),
  historical-artifact protection, offline dry-run end-to-end, pinned-judge
  requirement.

## Compatibility effects

- `weight_orthogonalization.py`, `capability_eval.py`,
  `norm_matched_controls.py`, `e17_stage3.py`, `e17b_null_audit.py` now
  FAIL FAST when invoked without `--component-set` or explicit specs. The
  historical rescue chain scripts (`run_rescue3*.sh`) already pass explicit
  sets and are unaffected; an unqualified rerun of any of these scripts can
  no longer silently edit superseded components (acceptance criterion).
- Removed imports `POSITIVE_WRITERS/SUPPRESSORS/TARGETED` from
  `weight_orthogonalization` were only consumed by `capability_eval.py` and
  `norm_matched_controls.py`, both updated; all other importers use
  functions only. `NEUTRAL_PROMPTS` and all function APIs are unchanged
  (`choice_scores_fmt` gained an optional `return_flips`; `clustered_delta`
  an optional `detail`; `judge_event` an optional `parse_fn`).
- `e17b.json` schema: legacy keys preserved except
  `test1_*.baseline.<cell>` (float → object with `mean`); no in-repo
  consumer reads that key.
- T row schema: all previously consumed fields unchanged; new
  `opener_variant`/`closing_variant` fields added; `pair_index` now indexes
  8 UNIQUE variants (previously 8 lockstep indices with duplicated text).
  No analysis script reads `pair_index`.
- New sidecar files (`*.provenance.json`) and `historical/` directories
  under `data/contrastive_pairs/v2_1/`.

## Required follow-ups (NOT executed in this batch — Gate 0B and later)

1. Spark `empathy` environment-lock capture + Gemma/Llama cached-revision
   verification (expected values above), before any rerun.
2. Repaired T/T-confirm scoring per `GATE0B_PREREG_2026-07-13.md` part 1
   (dual-order, raw + continuation, chat secondary; diagnostic bridge on
   historical T-confirm in the same process).
3. Blinded full E27 rejudge with a pinned independent judge + stratified
   human audit + majority adjudication (prereg part 2); resolves the three
   UNKNOWNs.
4. Gemma fractional activation ablation rerun with `d_resid` under the new
   self-contained e17b schema (raw + chat endpoints, adequate
   random-direction controls).
5. Manipulation-pretest reruns with a pinned judge (+ independent judge
   family and human sample) under the new runner, with monotonicity/
   separation gates predeclared.
6. `results/PROVENANCE.json` regeneration only after the accepted reruns.
