# Fable Implementation Brief: Integrity Repair A

Execute after reading:

- `notes/SCIENCE_AUDIT_2026-07-13.md`
- `notes/EXECUTION_ORDER_2026-07-13.md`
- `notes/EXPERIMENT_LOG.md` entries E24-E28 and later audit entries

## Worktree exclusions

- Do not touch, stage, or revert `notes/ROADMAP.md`.
- Do not stage/delete existing untracked paper figures, activation directories, or raw E27 game directories except when the brief explicitly asks for a later reviewed provenance action.
- Do not edit `paper-v2/paper.tex` in this batch.
- Do not run GPU jobs, API calls, or alter Spark state.
- Preserve accepted historical artifacts; never silently overwrite scientific history.

## A1. Repair T/T-confirm pseudoreplication

1. Replace lockstep `%2`/`%4` variant cycling with the explicit opener x closing Cartesian product.
2. Persist stable `opener_variant`, `closing_variant`, and `pair_index` fields.
3. Assert unique `(pos_text, neg_text)` pairs and expected scenario x variant counts before writing.
4. Add reusable deterministic-builder integrity helpers/tests where practical.
5. Preserve current JSONLs under clearly named historical paths plus hashes/metadata before regenerating.
6. Regenerate both files to 40 rows and 40 unique pairs, preserving development/confirmation family disjointness.
7. Do not claim result equivalence. List every model rerun that the changed T artifacts invalidate or require.

## A2. Guardrails and result schemas

### Environment and model revisions

- Add a small shared provenance helper that records Python/platform, CUDA/device, git commit/dirty state, and exact versions for Torch, Transformers, NumPy, scikit-learn, Datasets, TransformerLens, and SAELens when present.
- Resolve and persist Hugging Face model/tokenizer commit hashes, not only mutable model IDs. Accepted reruns should pass `revision=` explicitly after resolving the cached/current commit.
- Capture an exact Spark `empathy` environment lock/export for experimental reproduction; keep broad `requirements.txt` developer constraints separate and label them accordingly.
- Known audit references to verify rather than hard-code blindly: Gemma cache revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`, Llama cache revision `0e9e39f249a16976918f6564b8830bc894c89659`; Python 3.12.13, Torch 2.13.0+cu130, Transformers 5.13.0, TransformerLens 3.5.1, SAELens 6.45.3, Datasets 5.0.0 on the Spark.

### Weight edits

- Remove silent superseded component defaults from `weight_orthogonalization.py`.
- Require explicit component specs or a clearly versioned set-of-record config.
- Persist the resolved component set in every result.

### Fractional ablation

Upgrade `e17b_null_audit.py` to persist:

- direction path and SHA-256;
- cell paths and hashes;
- exact baseline per-pair scores;
- per-pair edited scores/deltas;
- option order/flip signs when exposed by the scoring API;
- prompt format, model, block, seed, model revision, dtype, and tokenizer settings;
- sufficient same-file data to reconstruct every reported percentage.

Maintain backward compatibility where it does not compromise auditability.

### Capability evaluation

- Resolve the mismatch between “stratified” documentation and simple random sampling.
- Prefer deterministic subject-stratified sampling; otherwise correct all claims and function names.
- Persist dataset name/config/revision, sampled row IDs, subjects, questions or stable hashes, targets, generated/predicted letters, parser status, and per-window perplexity losses.
- Do not rerun the model in this batch.

## A3. Judge and manipulation-check provenance

### E27 scorer

- Make the judge model/version configurable and require a pinned value for accepted runs.
- Version and persist the exact judge prompt.
- Persist deterministic input ID, full player message, full prior context, raw judge response, response/request metadata, attempt number, retry/error history, and final parsed label.
- Preserve condition blindness in judge inputs.
- Do not call an API or adjudicate the three current UNKNOWN rows.

### V2.2 manipulation pretests

Add a reusable runner/schema supporting need and moral stimuli with:

- randomized, blinded arm presentation;
- exact item/family/arm IDs and full rated text;
- versioned exact prompts and rating scales;
- configurable pinned judge model;
- raw response and retry/error persistence;
- item-level ratings and family/arm summaries;
- dry-run/export mode requiring no API access.

Do not overwrite historical `need_pretest.json` or `moral_pretest.json`.

## A4. Verification and delivery

1. Add focused unit tests for all pure logic changed.
2. Run targeted tests, then `pytest -q tests`. Do not use repository-wide pytest because root `test_gpt5_access.py` exits without an API key.
3. Write `notes/INTEGRITY_REPAIR_A_2026-07-13.md` with changes, preserved artifacts/hashes, tests, compatibility effects, and required reruns.
4. Check `git diff --check` and ensure excluded dirty files are untouched.
5. Commit only this batch and push `random-direction-control`.
6. Report commit hash, tests, and unresolved work. Do not start the reruns.

## Acceptance criteria

- T and T-confirm each contain 40 unique pairs with disjoint family sets.
- Historical T artifacts are recoverable and hashed.
- Unit tests demonstrate the former lockstep construction would fail.
- No accepted experiment can silently use old component defaults.
- New result/judge schemas are self-contained enough for independent recomputation.
- Experimental environment and resolved model snapshots are recorded independently of mutable package/model aliases.
- No API/GPU/Spark action occurred.
- Existing user work remains untouched.
