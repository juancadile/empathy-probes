# Gate 0C Preregistration: Evidence Reconstruction

**Status:** frozen before any corrected model/API scores.  
**Purpose:** replace three scientifically incomplete historical artifacts with
self-contained reruns. This is audit-driven reconstruction, not an untouched
first confirmation: the earlier capability, fractional-ablation, and
manipulation-check outcomes are already known.

Gate 0C may start only after Integrity Repair A independent QA is cleared and a
Spark environment lock is committed. Every run uses `accepted` mode, immutable
model/tokenizer/dataset revisions, direction-bound component resolution, fresh
output paths, atomic finalization, and complete per-example records.

## Part 1: capability sensitivity under the sets of record

### Inputs and conditions

- Model: `google/gemma-2-9b-it`, exact cached model and tokenizer commits
  captured before execution.
- Direction/components: registry key
  `gemma2_9b_it_resid_2026-07-12`; no literal overrides.
- Conditions in one process with snapshot/restore checks: baseline, writers,
  suppressors, and targeted.
- MMLU: `cais/mmlu`, exact dataset revision, test split, 800
  subject-stratified items. Fresh sample seed `946441184`, obtained before the
  run as the first 32 bits of SHA-256 of
  `Gate0C capability fresh sample v1 2026-07-13`.
- WikiText: exact `Salesforce/wikitext` revision. Use a frozen text hash and a
  disjoint segment from the historical first-200k-character pilot where the
  dataset length permits; record any unavoidable overlap before scoring.

### Readouts and inference

1. Primary MMLU readout: forced A/B/C/D option likelihood under one frozen
   chat-template prompt, eliminating free-generation parsing.
2. Secondary format readout: greedy short generation with the audited parser.
   Report all-item accuracy, parsed-only sensitivity, unparsed/ambiguous rates,
   and response examples by condition.
3. Report paired item deltas, per-subject deltas, subject-cluster bootstrap
   intervals, and all subject signs. Item bootstrap is secondary only.
4. WikiText reports per-window NLL and perplexity ratio with all window losses.
5. No equivalence margin is introduced after seeing the historical run.
   Therefore the strongest passing language is **no detected degradation on
   these measured benchmarks**, with the interval shown. A confidence interval
   containing zero is not proof of preserved general capability.

### Failure conditions

- Any revision/fingerprint mismatch, parser imbalance that changes the
  qualitative conclusion, restoration failure, missing per-item record, or
  output overwrite aborts acceptance.
- A detected decrement is reported; do not change sample, parser, or benchmark
  after opening scores.

## Part 2: corrected fractional activation ablation

### Scope and data status

This replaces the non-reconstructible pre-`d_resid` 88% headline. It is a
historical-assay robustness test because `M-confirm` is confirmation-exhausted;
it cannot finalize mediation or construct identity. Repaired `T-confirm` is the
paired task-control assay.

### Protocol

- Model and direction: exact Gemma revision and `d_resid`, block 20.
- Cells: existing M-confirm and repaired T-confirm, with hashes and families.
- Readouts: raw dual-order A/B primary; chat dual-order A/B and scaffold-free
  continuation likelihood as format sensitivities.
- Fractions fixed at `0, .25, .50, .75, 1.0`.
- Run 39 isotropic unit random directions at fraction 1.0 on the same M and T
  cells/readouts. Master seed `2490282906`, derived before execution as the
  first 32 bits of SHA-256 of
  `Gate0C fractional random directions v1 2026-07-13`; child-seed derivation is
  frozen in the accepted artifact.
- Primary statistic: absolute target-direction change in M margin. Empirical
  one-sided plus-one p-value compares the signed target effect to the 39 null
  effects (minimum resolution 1/40 = .025).
- Selectivity is descriptive `|delta T|/|delta M|` under every readout. Fraction
  of baseline margin is secondary because ratios are unstable near zero.
- Report all family effects, LOFO, direction seeds, per-pair baseline/edited
  scores, and option orders.

### Claim ceiling

Passing supports causal dependence of the tested forced-choice assay on the
block-20 direction relative to isotropic directions. It does not show that the
direction is welfare-pure, that the component circuit is complete, or that the
same dependence transfers to enacted behavior.

## Part 3: manipulation-check reconstruction

### Batteries

- Need battery: urgent, mild, resolved, and excited-positive-social arms using
  the frozen nuisance pins in the audited runner.
- Moral battery: lower/equal/higher relative claim of the **newly arriving
  person/P2**, named unambiguously in each judge question.
- Full exact texts, structured target identity, source hashes, item/family/arm
  IDs, and blinded randomized order are persisted.
- Presentation seed `3987230607`, derived before execution as the first 32 bits
  of SHA-256 of `Gate0C manipulation presentation v1 2026-07-13`.

### Adjudication

1. Primary automated judge must be an exact snapshot from a model family
   independent of the historical Claude-family judge. Provider, request model,
   returned model, prompt version, raw responses, retries, and errors persist.
2. A second model family or blinded human audit rates a stratified minimum of
   five items per arm per question. Disagreement is reported, not silently
   reconciled.
3. The inferential unit is scenario family. Report every family profile,
   arm-order monotonicity, LOFO, and descriptive cluster uncertainty.
4. Any UNKNOWN remains missing; no aggregate is accepted until resolved by the
   predeclared independent adjudication path.

### Claim ceiling

These checks establish that the authored prompts are perceived as ordered on
the named composite manipulation by the tested adjudicators. They do not prove
that model activations isolate need, welfare, morality, or task cost.

## Deliverables and exit

Each part produces a new immutable result directory, frozen manifest, complete
artifact, engineer interpretation, and independent adversarial review. Gate 0C
closes only when all three are reconstructible or their failures are logged.
Paper/showcase claims remain unchanged until that review.

