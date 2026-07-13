# Gate 1 Preregistration: Parameter-Intervention Localization

**Status:** frozen before new Gate-1 target-model scores.  
**Model:** Gemma-2-9B-it, exact revision captured by Gate 0.  
**Intervention axis:** `d_resid`, treated only as the current costly-helping
assay axis.  
**Sets of record:** writers `L19MLP,L20MLP`; suppressors
`L18H13,L20H10,L19H12,L17H7`.

Gate 1 asks two different questions and keeps their data separate:

1. Does the frozen writer intervention replicate on genuinely new costly-helping
   families while remaining smaller on matched task controls?
2. Is the suppressor need-by-cost slope unusually associated with the selected
   four heads after realized edit norm is paired?

Neither test identifies a welfare representation. Scenario family is the
inferential unit; lexical variants and A/B orders are repeated measurements.

## Shared integrity requirements

- Integrity Repair A, Gate 0B, and the Spark environment lock must pass first.
- Every artifact runs in `accepted` mode with immutable model/tokenizer
  revisions, direction-bound registry resolution, input hashes, per-example
  scores, exact edit deltas, restoration checks, and fresh atomic output paths.
- New families cannot overlap M, M-confirm, T/T-confirm, V2.2, WP3, B6,
  circuit-confirm, E22b, or EIA confirmation families.
- Authors/providers are stimulus sources, not independent replications. Preserve
  source labels and report source-stratified sensitivity without inflating n.
- No protected target-model scores are inspected until structural audits,
  manipulation checks, split manifests, code commit, and analysis manifests are
  frozen.
- Failure remains the result. Prompt, component, direction, dose, endpoint, and
  threshold changes require a new experiment and new families.

## Part A: writer intervention replication

### Stimulus generation and sealed split

Create at least 48 new costly-helping scenario families plus one matched
task-control family for each. Cover at least four domains and at least four
stimulus-source strata. Normalize every family into the same structural schema:

- shared prefix ending immediately before the decision;
- help/interruption versus task-persistence decision tails matched for syntax,
  tense, register, and approximate token count;
- active objective with a real stated loss from interruption;
- matched task control in the same domain with no person/current welfare need;
- at least four deterministic surface variants per family, all unique.

Before any model score, split the 48 family pairs into:

- 16 `M-confirm2` + matched task-control families for this Gate-1 test;
- 16 `M-circuit-validation` + matched task controls for Gate-3 graph/feature
  model selection after historical discovery screening;
- 16 `M-circuit-confirm` + matched task controls reserved for Gate 3.

The split seed is `85758990`, the first 32 bits of SHA-256 of
`Gate1 M-confirm2 circuit split v1 2026-07-13`. Stratify by domain and stimulus
source. Persist the complete assignment and source hashes. If fewer than 16
valid family pairs remain in any partition after structural audit, generate
additional families before scoring; never move a scored family between roles.

### Pre-model validity gates

- Byte-identical shared prefix within each positive/negative pair.
- Decision tails contain the only branch difference and remain within a frozen
  token-length tolerance reported per tokenizer.
- No duplicate normalized text, family, objective, or beneficiary across old or
  new pools under exact and near-duplicate checks.
- Blinded manipulation audit identifies the helping branch and task branch with
  at least 15/16 family-majority correctness; any failed family invalidates the
  sealed batch rather than being silently rewritten.
- Matched task controls contain no person in present need. This is checked by an
  independent model family plus a blinded human sample before model scoring.

### Conditions and readouts

Evaluate from the original snapshot each time:

1. baseline;
2. full writer-set `d_resid` edit;
3. each writer alone (secondary decomposition only);
4. 39 norm-matched random directions on the same two writer components;
5. the frozen composition-matched random-component set as descriptive context,
   not an empirical null.

Random-direction master seed is `1113820829`, the first 32 bits of SHA-256 of
`Gate1 writer random directions v1 2026-07-13`. Child seeds and the generator
algorithm are persisted. Random controls use the same M/task families and all
primary readouts. One isotropic unit direction is shared across both components
within a draw. For draw `j`, measure the target and random natural full joint
post-bf16 delta norms, set `N_j` to the smaller, and dose-reduce the larger from
a fresh checkpoint until both are within 3%; never extrapolate beyond full
orthogonalization. Direction specificity compares the paired target and random
effects at `N_j`, not unmatched full edits.

Primary readouts, averaged over both A/B orders:

- raw forced-choice logit difference;
- scaffold-free continuation log-likelihood difference.

Chat-templated forced choice is a required format sensitivity. Report per-pair,
per-family, and per-source effects, all family signs, family-cluster bootstrap,
and LOFO. Intervals at 16 authored families remain approximate.

### Fixed decisions

Writer replication passes only if all are true:

1. joint writer edit reduces helping preference under both primary readouts;
2. the family-cluster interval excludes zero under both primary readouts;
3. at least 13/16 family effects have the predicted sign under each primary
   readout and every LOFO aggregate retains the predicted sign;
4. repaired matched-task effect satisfies `|delta T| < |delta M|/3` under both
   primary readouts;
5. at every paired norm `N_j`, the target M effect is more negative than its
   same-component random-direction effect under the frozen signed statistic
   (39/39; plus-one resolution 1/40);
6. chat sensitivity has the same sign and no parser/tokenization/restoration
   gate fails.

If only some readouts pass, report format dependence. Single-writer results do
not rescue a failed joint replication and cannot establish unique-component
localization. A pass supports a frozen two-MLP-set intervention on this assay,
not a complete circuit or welfare-value writer label.

## Part B: suppressor component specificity (R2b)

### Fresh factorial families

Create 32 new families with an expressed-current-need x active-objective-cost
factorial, split before any target-model score into 16 `R2b-dev` and 16 sealed
`R2b-confirm` families. Split seed is `1448514577` (first 32 bits of SHA-256 of
`Gate1 R2b dev confirm split v1 2026-07-13`), stratified by domain and stimulus
source. Primary factors:

- need: resolved/no-current-need versus urgent/current-need;
- stipulated/anticipated interruption loss: zero, low, medium, high, while an
  active objective remains present at every primary cost level.

`No active objective` is a separate diagnostic arm, never the zero-cost primary
level. Within family, decision tails are byte-identical across need/cost cells;
only the structured prefix manipulation changes. Surface variants are nested
within family. These are textual consequence manipulations, not mechanically
incurred costs; mechanically realized cost is reserved for Gate 4.

Before target-model scoring, blinded manipulation checks must show in each
split:

- urgent rated above resolved in at least 13/16 families at every cost level;
- aggregate cost ratings strictly ordered zero < low < medium < high;
- no need-by-cost manipulation interaction larger than 0.5 points on the
  five-point rating scale in either direction;
- no arm-label leakage in decision tails;
- independent-model/human disagreement is reported and any systematic target
  ambiguity invalidates the batch.

Only `R2b-dev` wording may be revised after a failed manipulation check.
Structurally invalid sealed families are replaced before any sealed target
score; after that point confirmation is immutable.

Presentation seed is `216633504`, the first 32 bits of SHA-256 of
`Gate1 R2b manipulation order v1 2026-07-13`.

### Frozen target and null universe

- Target: exactly one selected attention head at each of L17-L20.
- Primary null universe: all four-head sets with exactly one non-target head at
  each of L17-L20 and no overlap with the target.
- Draw 32 null sets uniformly without replacement before behavioral scores.
  Master seed `326118207`, the first 32 bits of SHA-256 of
  `Gate1 R2b null-set sample v1 2026-07-13`.
- Persist the complete universe definition, ordered sampled sets, and inclusion
  probability. A secondary all-nontarget universe is descriptive only.

For every target/null comparison `j`:

1. restore the original checkpoint;
2. measure natural full post-bf16 joint delta norm as
   `sqrt(sum(component_delta_norm_i^2))`;
3. set `N_j` to the smaller target/null natural full norm;
4. solve doses in `[0,1]` from fresh snapshots until both realized norms are
   within 3% of `N_j`; never over-edit or dose cumulatively;
5. score target and null in the same process on identical ordered inputs.

### Estimand and inference

For each family, compute the intervention-induced urgent-minus-resolved change
in the cost slope over the four active-objective levels. For null set `j`:

`D[f,j] = slope_target[f,j] - slope_null[f,j]`.

Primary effect is the mean over null sets within each family, then the mean over
16 families. Report all `D[f,j]`, family means, null-set means, family-cluster
bootstrap interval, family signs, and LOFO. Resampling null sets is a sensitivity
analysis because all comparisons share the target.

Target dose monotonicity and any numerical solver tolerance beyond the fixed 3%
norm gate are checked only on `R2b-dev` before sealed scores. On confirmation
families, a
sign reversal or gross nonmonotonicity invalidates the component-specific
interpretation rather than triggering dose selection.

R2b passes on `R2b-confirm` only if:

1. every retained pair clears the 3% realized-norm gate;
2. target dose response is monotone in the predicted direction;
3. the family-cluster interval for mean `D` excludes zero on the positive side;
4. at least 13/16 family-average `D` values are positive;
5. every LOFO aggregate remains positive.

Natural full-edit rank is descriptive unless the complete universe is
enumerated. A pass localizes the slope profile to the selected heads conditional
on this layer/type/null universe. A failure retains only the existing joint-set,
direction-specific profile; do not search new null pools on these families.

## Gate-1 deliverables

For each part: frozen manifest, structural/manipulation audit, complete result,
Fable engineer interpretation, independent adversarial review, and an
experiment-log entry preserving failures and claim ceilings. Gate 1 closes only
after both parts are resolved; Gate 3 may use individual suppressor heads only
if Part B passes.

## Frozen R2b split amendment (2026-07-13, before generation)

The earlier text referred to development families for dose monotonicity but
defined only one 16-family R2b pool. R2b now has 16 development and 16 sealed
confirmation families. Development may calibrate implementation behavior;
component-specific inference uses confirmation once.
