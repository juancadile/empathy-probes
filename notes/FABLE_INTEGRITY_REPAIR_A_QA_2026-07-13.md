# Fable Integrity Repair A: Independent QA Findings

**Date:** 2026-07-13  
**Scope:** Offline repair of commit `96571f017cf3ab5c8f3471a7e0614f5fe6de5d55`.  
**Status:** Repair A is not cleared for Gate 0B. Do not run Spark, GPU jobs, APIs,
or scientific experiments until every blocker below is fixed and independently
reviewed.

Read first:

- `notes/FABLE_INTEGRITY_REPAIR_A_2026-07-13.md`
- `notes/INTEGRITY_REPAIR_A_2026-07-13.md`
- `notes/GATE0B_PREREG_2026-07-13.md`
- the open super.engineering review comments on the files below

Keep the original worktree exclusions. In particular, do not touch or stage
`notes/ROADMAP.md`, paper figures, cached activations, raw E27 run directories,
the paper, or result artifacts. This is an offline code/schema/test repair.

## Q1. Restore EIA driver importability without silent component defaults (P0)

`src/eia_validation/run_eia_local.py` still imports removed constants from
`weight_orthogonalization.py`, so the module fails to import before argparse.

Required repair:

- Resolve components through the versioned registry or complete literal specs.
- Require an explicit component-set selection for evidence-producing runs.
- Do not reintroduce generic `POSITIVE_WRITERS`, `SUPPRESSORS`, or `TARGETED`
  constants.
- Persist the complete resolution in the run artifact.

Acceptance tests:

- `python -c 'import src.eia_validation.run_eia_local'` succeeds.
- CLI help succeeds without model loading.
- A test proves omission of component selection cannot silently choose a set.
- A test proves the current Gemma registry key resolves the sets of record.

## Q2. Make independent judging an actual supported path (P1)

`score_e27.py` and `manipulation_pretest.py` currently hardwire Anthropic
transport. The historical E27 judge was Claude-family, while Gate 0B requires a
version-pinned independent judge family.

Required repair:

- Make provider explicit in the CLI and persisted provenance.
- Support at least Anthropic and the intended independent provider (OpenAI), or
  implement a provider-neutral injected transport with concrete adapters for
  both.
- Keep parsing, retry history, raw response, request metadata, model ID, and
  provider in the result schema.
- Dry-run must perform no API call and should work without provider SDK imports
  where practical.
- Do not call either API in this batch.

Acceptance tests:

- Unit tests exercise both adapters with mocked responses and errors.
- E27 and both manipulation batteries render/export in dry-run mode for an
  explicitly selected independent provider.
- Live mode refuses an omitted provider or model.
- Output provenance distinguishes provider, requested model, returned model or
  version when available, and prompt version.

## Q3. Identify the manipulated beneficiary in moral pretests (P1)

The moral prompt asks about "the OTHER person," but each item contains two
non-narrator beneficiaries. The target must be the newly arriving person/P2,
whose `relative_need` arm is manipulated.

Required repair:

- Give each battery and question a target definition derived from structured
  row fields, not a shared ambiguous phrase.
- For `moral_v2_2`, name or otherwise unambiguously identify P2/new arrival in
  both `need_now` and `respond_now` questions.
- Preserve target identity in the item-level output schema.

Acceptance tests:

- Snapshot/render tests show that moral prompts identify P2/new arrival and do
  not refer merely to "the other person."
- Tests cover at least two different names/families and both moral questions.
- Need-battery wording remains semantically unchanged except where needed for
  explicit targeting.

## Q4. Bind component registries to their direction (P1)

`component_sets.py` records a direction for each registry entry but currently
checks only the model ID. A caller can pair the current components with the old
direction, recreating the mismatch the registry is intended to prevent.

Required repair:

- Normalize and compare the supplied direction path to the registry entry.
- Prefer also verifying SHA-256 when the file exists; persist expected and
  observed path/hash.
- Evidence-eligible runs must fail on a direction mismatch.
- An exploratory override may exist only if explicit, prominently warned, and
  persisted as making the run ineligible for confirmatory evidence.
- Apply the same resolution contract to every caller that can produce a result,
  including `norm_matched_controls.py`.

Acceptance tests:

- Current set + current direction passes.
- Current set + historical direction fails.
- Literal component specs require explicit direction provenance.
- Exploratory override is visible in the returned/persisted resolution.

## Q5. Replace the permissive MMLU answer parser (P1)

`parse_letter` currently searches for any capital A-D character. For example,
`Answer: C` can be scored as A.

Required repair:

- Accept an exact bare option, an answer-marked option, or a single
  unambiguous standalone option.
- Reject/no-score prose with zero candidates.
- Return explicit `ambiguous` status when multiple candidates remain.
- Do not silently choose the first candidate.

Acceptance tests must cover:

- `C`, `(C)`, `Answer: C`, and `The answer is C.` -> C;
- ordinary prose containing letters A-D -> no parse unless a valid standalone
  answer marker exists;
- `A or C`, multiple answer markers, and conflicting options -> ambiguous;
- parser status is persisted and ambiguous/unparsed items are not counted as
  ordinary wrong answers without an explicit analysis policy.

## Q6. Enforce immutable revisions for evidence-eligible runs (P1)

Revision pinning is documented but optional. A run can still silently use
mutable model, tokenizer, MMLU, or WikiText aliases.

Required repair:

- Add an explicit run class such as `--run-mode accepted|exploratory`.
- `accepted` must require immutable model, tokenizer, and dataset revisions and
  verify resolved commits/fingerprints against the requested values.
- `exploratory` may allow omissions, but this must be persisted and visibly
  mark the artifact ineligible for confirmatory evidence.
- Apply the accepted-run contract to all repaired model-producing entry points,
  not only `capability_eval.py`, where feasible in this batch.

Acceptance tests:

- Accepted mode fails before loading data/model when a required revision is
  absent or mutable.
- Accepted mode fails on a requested/resolved commit mismatch.
- Exploratory mode persists its status and missing pins.
- Pure validation tests require no network or model load.

## Q7. Make fractional-ablation null resolution adequate and explicit (P2)

`e17b_null_audit.py` hardcodes three random directions. That is acceptable only
for a smoke test, not the frozen follow-up.

Required repair:

- Make random-direction count configurable and required for evidence-eligible
  runs.
- Freeze and persist count, seeds, generation algorithm, and minimum attainable
  plus-one Monte Carlo p-value.
- Keep three draws only behind an explicit smoke/exploratory mode.
- Deterministic seed generation must not change with batch partitioning.

Acceptance tests:

- Same master seed/count yields the same direction seeds.
- Accepted mode rejects three draws and an omitted count; use a documented
  minimum justified by the preregistered inferential resolution.
- Smoke mode clearly labels the artifact as non-evidential.

## Q8. Validate existing deterministic-builder sidecars (P2)

`write_jsonl_guarded` trusts an existing sidecar when the rebuilt artifact is
byte-identical. A stale or tampered sidecar can survive indefinitely.

Required repair:

- Before returning unchanged, validate sidecar schema, artifact hash, row and
  unique-pair counts, family list, preserved-copy path/hash, and any grid facts
  asserted by the builder.
- Contradictory or missing required provenance must fail closed. Do not silently
  rewrite a contradictory scientific record.
- A separate explicit offline repair command may reconstruct a missing sidecar
  only from verified artifact and preserved-copy bytes, leaving an audit entry.

Acceptance tests:

- Tampering each critical sidecar field causes deterministic rebuild failure.
- Tampering the preserved copy causes failure.
- A valid unchanged artifact+sidecar remains byte-identical after rebuild.

## Delivery Contract

1. Add focused tests for every acceptance criterion above.
2. Run `pytest -q tests` and report the exact count.
3. Run import smoke checks for every changed executable module.
4. Run offline dry-runs for E27 and both manipulation batteries using the
   independent-provider path; no API calls.
5. Run `git diff --check` and compile changed Python files.
6. Update `notes/INTEGRITY_REPAIR_A_2026-07-13.md` with a clearly separated
   independent-QA repair addendum; do not rewrite history.
7. Commit only the QA repair and push `random-direction-control`.
8. Report unresolved scientific or operational work. Do not start Gate 0B,
   Spark, GPU, API, or paper work.

## Clearance Rule

Repair A is cleared only after an independent post-Fable review confirms all
eight items, the complete test suite passes, import smoke succeeds, and dry-run
artifacts demonstrate provider/revision/component provenance. Passing unit tests
alone is not clearance.
