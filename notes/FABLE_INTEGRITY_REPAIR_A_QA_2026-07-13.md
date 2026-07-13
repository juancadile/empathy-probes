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

The existing historical moral JSONL has `scenario_id` and role prose but no
beneficiary-name field. Do not infer identity by brittle text parsing and do not
mutate that historical stimulus artifact in this repair. Use a deterministic,
versioned target-descriptor manifest keyed by `scenario_id` (for example, “the
student who has just messaged for help”), validate complete one-to-one coverage,
and persist the descriptor plus manifest version/hash in every rendered item.

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
- A superseded registry key may support exact historical reproduction but must
  be rejected in `accepted` mode; a warning alone is insufficient.

Inventory every CLI that imports the low-level weight-edit functions. Each must
either adopt the shared accepted/exploratory contract or be explicitly and
persistently `historical_or_exploratory_only`, with no accepted mode. In the
current tree this includes at least `weight_orthogonalization.py`,
`capability_eval.py`, `norm_matched_controls.py`, `e17_stage3.py`,
`e17b_null_audit.py`, `e18_interaction.py`, `e26_format_stress.py`,
`e26_matched_nulls.py`, `e28b_slope_nulls.py`, both EIA game drivers, and the
edit path in `analysis/logit_lens_trajectory.py`. Historical scripts do not all
need full migration in this batch, but none may emit an artifact whose evidence
eligibility is absent or ambiguous.

Acceptance tests:

- Current set + current direction passes.
- Current set + historical direction fails.
- Literal component specs require explicit direction provenance.
- Exploratory override is visible in the returned/persisted resolution.
- A static/inventory test proves every direct weight-edit CLI is either governed
  by the shared accepted contract or explicitly non-evidential.

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

For the capability summary, persist both an all-items analysis (format failure
counts as task failure) and a parsed-only sensitivity analysis with parse rates
by condition. Because sampling is subject-stratified, uncertainty for the
accuracy delta must resample subjects/clusters rather than treating all 400
questions as independent. A confidence interval containing zero supports only
"no detected change"; it is not an equivalence/no-cost result unless a separate
equivalence margin was frozen before the run.

Gate 0C additionally freezes forced A/B/C/D option likelihood as the **primary**
MMLU readout. Implement that estimator in the repaired capability path (with
multi-token-safe label likelihood and mock-logit tests); greedy generation plus
the audited parser is the secondary format sensitivity. Do not let the parser
repair silently redefine the preregistered primary estimator.

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

Evidence-eligible output paths must also fail closed if the result artifact
already exists. Write through a temporary file and atomically rename only after
the complete artifact validates; never silently overwrite a prior accepted run.

An accepted run must also bind the executed scientific source to the recorded
commit. It may not remain eligible merely because provenance records
`dirty: true`. Fail if tracked scientific code/data are modified or relevant
untracked files exist, or persist and validate a complete content-addressed
source/input manifest whose hashes are used as the executable identity. Merely
recording a dirty-path count is insufficient. Generated output paths may be
excluded only by an explicit, persisted rule.

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
- Random-direction controls score the same M and T cells/readouts as the target
  ablation so they can assess effect magnitude and selectivity, not M alone.

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

## Q9. Require the complete E27 design, not only the observed cross-product (P1)

`score_e27.aggregate` currently forms the expected grid from conditions,
variants, and seeds that happen to exist. If an entire condition, variant, or
seed is absent, that dimension disappears and the incomplete dataset can pass.

Required repair:

- The scorer must consume or require an explicit design manifest containing
  expected conditions, variants, seeds, and exactly one run artifact per cell.
- Reject missing cells, duplicate cell artifacts, unexpected cells, failed-run
  summaries, and hash/path disagreements before judging or aggregation.
- Persist the validated design manifest and validation report.
- Dry-run must exercise the same design validation as live judging.

Acceptance tests:

- Removing one cell fails (already partially covered).
- Removing every cell for one seed, variant, or condition also fails.
- Duplicating a run for one cell fails.
- The real 2 x 3 x 8 E27 tree validates as exactly 48 runs before any API call.

## Q10. Make game-generation artifacts evidence-auditable (P1)

`run_eia_local.py` and `e27_game_variants.py` can still generate behavioral
evidence without immutable model/tokenizer revisions, direction hash, registry
resolution, generation configuration, game/message-pool hashes, or protected
output semantics. Existing E27 histories cannot be retroactively upgraded; they
remain historical evidence and require the separately preregistered fresh EIA
confirmation for a final claim.

Required repair:

- Apply the accepted/exploratory run contract and direction-bound component
  resolution to both game drivers.
- Persist model/tokenizer requested and resolved revisions, direction path/hash,
  complete component resolution, edit diagnostics, seed/run identity,
  generation parameters, scenario/message-pool hashes, EIA code commit/hash,
  and environment provenance in a run-level manifest.
- Use a fresh deterministic player RNG/call counter per run so paired seeds do
  not depend on preceding trajectories; persist the seed derivation.
- Refuse existing accepted output directories and avoid partial results being
  mistaken for complete grids. Preserve failure states explicitly.
- Do not alter or claim to repair the existing raw E27 histories in this batch.

The baseline arm in a paired accepted game design must resolve and persist the
same registry entry and direction as the edited arm even though it applies no
edit. Otherwise the baseline artifact is not bound to the intervention it is
supposed to control.

Acceptance tests:

- Both modules import and show CLI help offline.
- Pure/mock tests validate run-level seed reset and manifest construction.
- Accepted mode rejects missing revisions, direction mismatch, or existing
  output; exploratory mode records ineligibility.
- No model, game, API, GPU, or Spark run occurs in this repair.

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

## Resume Addendum (2026-07-13, independently verified before Fable reset)

Architect HEAD is `f1b846a`; commits after `96571f0` are preregistration and QA
documents only. The six pre-existing dirty/generated paths remain excluded.

Offline baseline on the local worktree:

- `pytest -q tests` -> `118 passed in 6.13s`;
- `python3 -c 'import src.eia_validation.run_eia_local'` -> fails because
  `POSITIVE_WRITERS` was removed but is still imported;
- `python3 -m src.eia_validation.run_eia_local --help` -> same import failure;
- `score_e27.py --help` and `manipulation_pretest.py --help` succeed but expose
  no explicit provider argument and still hardwire Anthropic live transport.

Passing tests therefore do not clear the ten gates. Add tests that fail before
each repair and exercise the actual CLI/import surfaces.

### Recommended implementation order

1. **Shared evidence-run contract (Q4/Q6):** extend `component_sets.py` and
   `run_provenance.py` with direction path/hash binding, accepted/exploratory
   eligibility, immutable revision verification, and atomic no-overwrite output.
   Migrate every evidence-producing caller, including
   `norm_matched_controls.py`.
2. **Game drivers (Q1/Q10):** migrate `run_eia_local.py` and
   `e27_game_variants.py` to that shared contract; remove stale constants;
   require an explicit registry key or complete literal specification; add
   deterministic per-run player RNG and complete manifests.
3. **Judge transport and target identity (Q2/Q3):** create concrete mocked
   Anthropic and OpenAI adapters behind one injected interface. Add explicit
   provider/model CLI fields and schemas, then make moral prompts identify P2.
4. **E27 design validation (Q9):** validate an explicit 2x3x8 manifest before
   rendering or transport. Missing whole dimensions, duplicates, unexpected or
   failed cells, and path/hash disagreement must fail before any API call.
5. **Capability evaluation (Q5/Q6):** replace `parse_letter`, persist parse
   status, add all-items and parsed-only analyses, parse rates, and subject-
   cluster uncertainty; apply accepted-run revisions/output semantics.
6. **Fractional nulls (Q7):** accepted mode requires the preregistered random
   count and scores the same M/T/readouts; three draws remain smoke-only.
7. **Builder sidecars (Q8):** validate every existing sidecar and preserved copy
   before unchanged return; contradictory provenance fails closed.

Run focused tests after each shared layer, then the full delivery contract.
Ignore paper/showcase review comments in this batch. Do not resolve scientific
review threads merely because code compiles.

## Clearance Rule

Repair A is cleared only after an independent post-Fable review confirms all
ten items, the complete test suite passes, import smoke succeeds, and dry-run
artifacts demonstrate provider/revision/component provenance. Passing unit tests
alone is not clearance.

## Reset Handoff Update (2026-07-13 04:45 EDT)

- Current architect HEAD is `d7189c3`. Commits after the original Fable repair
  remain preregistration, QA, and roadmap-coverage documents only; no protected
  score was opened.
- `ROADMAP_COVERAGE_AUDIT_2026-07-13.md` closes the planning audit. It does not
  add work to Integrity Repair A.
- `pytest -q tests` remains green at `118 passed in 5.62s` before the QA repair.
- Do not use bare `pytest -q` as the acceptance command: the repository root
  contains `test_gpt5_access.py`, an unrelated live-API smoke script that exits
  during collection when `OPENAI_API_KEY` is absent. This is outside the repair
  batch. The required suite is `pytest -q tests` plus the explicit import, CLI,
  dry-run, compilation, and diff checks in the Delivery Contract.
- Q1 and Q2 remain visibly failing at the CLI surfaces: the EIA driver cannot
  import because of removed component constants, and the judge/pretest CLIs
  still expose no provider selection.
