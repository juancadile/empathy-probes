# C3 Preregistration: Base-versus-Instruct Weight Differences

**Status:** frozen before loading/comparing the base checkpoint.

## Question

Did instruction tuning change the selected costly-helping mechanism, and are those parameter differences causally responsible for a corresponding behavioral difference?

A static base-versus-instruct weight difference is descriptive. It becomes causal evidence only if a frozen, localized delta transplant/reversal changes held-out behavior in the predicted direction against matched controls.

## Prerequisites

1. Verify from authoritative checkpoint metadata that the instruct checkpoint descends from the named base checkpoint without an intervening incompatible pretraining branch. Persist model revisions and lineage evidence.
2. Verify identical architecture, tensor names/shapes, tokenizer/vocabulary, tied-weight configuration, and layer/norm conventions.
3. Use a common raw prompt protocol for primary base/IT comparison. Chat templates are model-specific and therefore secondary, not mixed into the primary contrast.
4. Freeze the current component sets and behavioral assays before weight-difference inspection.
5. Generate 16 `C3-dev` and 16 sealed `C3-confirm` costly-helping families
   with matched task controls. They may use the Gate-1 schema but cannot reuse
   Gate-1/circuit/WP3/B6/EIA families or close paraphrases.
   Split seed is `2081466070` (first 32 bits of SHA-256 of
   `C3 assay split v1 2026-07-13`), stratified by domain/source.
6. If a Gate-2 construct-transfer claim is attempted, generate a separate 16
   `C3-factorial-dev` and 16 `C3-factorial-confirm` pool before checkpoint
   differences are inspected. Without it, C3 remains a costly-helping-assay
   result and WP3/B6 transfers are diagnostic only.

If lineage or tensor correspondence cannot be established, restrict the analysis to cross-checkpoint descriptive geometry and do not call the difference an instruction-tuning delta.

## Checkpoints and sets

- Base: exact revision of `google/gemma-2-9b`.
- Instruct: exact accepted revision of `google/gemma-2-9b-it`.
- Current writers: L19MLP, L20MLP.
- Current suppressors: L18H13, L20H10, L19H12, L17H7.
- Matched component controls are fixed before behavior evaluation and preserve type/layer/norm structure.

The primary intervention unit is the joint six-component selected delta (two
writer MLPs plus four suppressor heads). Writer-only, suppressor-only, and
single-component deltas are mandatory decompositions but cannot rescue a failed
joint primary test.

## Descriptive decomposition

For each corresponding tensor `W`:

`Delta_W = W_IT - W_base`

Record:

- absolute and relative Frobenius norm;
- cosine/alignment with the rank-1 intervention delta and effective `d_resid` preimage;
- singular spectrum/top-subspace overlap;
- layer/type-matched empirical percentile;
- changes to relevant RMSNorm gain vectors separately from output weights.

Do not select components or ranks from held-out behavioral effects. Descriptive whole-model scans are multiple-comparison maps, not localized findings.

## Causal interventions

### Local delta transplant

Starting from the base model, add fractions `alpha in {0,.25,.5,.75,1}` of the exact IT-minus-base delta for the frozen selected tensors only.

### Local delta reversal

Starting from the IT model, subtract the same fractions from the corresponding tensors.

### Direction-aligned versus orthogonal decomposition

Decompose each selected `Delta_W` into:

- component parallel to the frozen rank-1 costly-helping edit geometry;
- Frobenius-orthogonal remainder.

Test parallel and remainder transplants separately at matched realized norm. This distinguishes a specific direction-aligned training change from generic component drift.

### Controls

- exactly 39 layer/type/norm-matched six-component IT-minus-base delta sets;
- exactly 39 random rank-matched projections of the selected deltas within the
  same six tensors;
- parameter-count and realized-joint-norm matched edits;
- sham snapshot/restore condition;
- tokenizer/prompt parity and parser gates.

All fractional edits are constructed from the original checkpoint snapshot, never applied cumulatively.
Control master seed is `4240959829` (first 32 bits of SHA-256 of
`C3 matched delta nulls v1 2026-07-13`); separate child streams are persisted
for component-set and within-tensor projection nulls. Every target/null pair is
matched on realized post-cast joint norm within 3% without over-editing.

## Behavioral evaluation

Primary confirmation sets:

- sealed `C3-confirm` and matched task controls;
- the separately generated `C3-factorial-confirm` pool only if a Gate-2
  construct-transfer claim is attempted;
- a fresh 800-item subject-stratified MMLU sample and disjoint WikiText segment
  for transplant/reversal conditions, following the Gate-0C forced-option,
  subject-cluster, and no-post-hoc-equivalence rules. MMLU sample seed is
  `1581550924` (first 32 bits of SHA-256 of
  `C3 capability sample v1 2026-07-13`) and must exclude Gate-0C item IDs.

M-confirm2, WP3-confirm, and R2b families are development/diagnostic transfers
because their outcomes are known before C3. They cannot satisfy C3 confirmation.

Report base and IT baselines before interventions. A transplant cannot be interpreted when the receiving checkpoint lacks assay headroom or prompt validity.

## Primary contrasts

On C3-dev, freeze polarity `s = sign(mean(Y_IT - Y_base))`, where Y is the
dual-order costly-helping action log odds. For each confirmation family, fit the
predeclared OLS slope over `alpha={0,.25,.5,.75,1}` separately for:

1. base transplant, `S_base = s * dY_base/dalpha`;
2. IT reversal, `S_reverse = -s * dY_IT/dalpha`;
3. Bidirectional consistency: transplant and reversal effects have opposite signs under a common metric.
4. Specificity: selected delta effect exceeds matched component/rank controls and remains bounded on task/capability controls.
5. Geometry: direction-aligned delta explains a predeclared fraction of the full selected-delta effect; otherwise the mechanism change is not reducible to the current rank-1 axis.

The joint C3 causal result passes only if:

1. the unedited base-versus-IT gap on C3-confirm has polarity `s`, a
   family-clustered 95% interval excluding zero, and at least 13/16 family signs;
2. family-clustered 95% intervals for both `S_base` and `S_reverse` are above
   zero, at least 13/16 family slopes are positive for each, every LOFO aggregate
   is positive, and aggregate dose means are monotone;
3. the bidirectional mean `(S_base + S_reverse)/2` exceeds both 39-draw null
   families at plus-one rank at most `2/40`;
4. under both directions, the matched task slope has absolute magnitude below
   one third of the costly-helping slope; and
5. the fresh capability results support only the separately calibrated language
   allowed by their own preregistration.

For the stronger rank-1 geometry claim, the parallel delta must pass both
checkpoint slopes, reach at least 50% of the full joint-delta point estimate in
both directions, and exceed the realized-norm-matched orthogonal remainder with
a family-paired 95% interval above zero. Otherwise report selected-block tuning
causality without reduction to the current rank-1 axis.

Family-level effects, all doses, all nulls, and LOFO are mandatory. Finite
resolution is reported; no Gaussian z-score extrapolation is allowed.

## Interpretation

| Outcome | Supported claim |
|---|---|
| Bidirectional selected-delta transfer, matched controls quiet | instruction tuning causally altered behavior through these parameter blocks |
| Parallel delta alone transfers | training change partly aligns with the identified rank-1 mechanism |
| Only orthogonal/full delta transfers | training changed the component, but not through the current direction |
| Static alignment without behavioral transfer | descriptive weight correlation only |
| Base assay invalid/no headroom | no causal base/IT conclusion on this assay |
| Matched deltas transfer similarly | broad tuning drift, not selected-mechanism specificity |

Never infer moral improvement, empathy installation, or training intent from checkpoint differences alone.

## Frozen amendment (2026-07-13, before checkpoint comparison)

M-confirm2 and R2b are opened in earlier gates and cannot remain C3 primary
confirmation. C3 now has dedicated development/confirmation families and a
fresh capability sample; earlier assays are diagnostic transfer only.

The joint intervention unit, dedicated optional factorial pool, bidirectional
slope gates, exact matched-null families, and rank-1 geometry threshold were
subsequently frozen before checkpoint comparison. Writer/suppressor
decompositions cannot replace a failed joint test.

## Frozen parameter-block amendment (2026-07-13, before checkpoint loading)

### Exact selected deltas and endpoint identity

The primary six blocks are exactly the parameter slices targeted by the
accepted Gate-1 rank-one edit, not every parameter in the surrounding module:

- `L19MLP` and `L20MLP`: the complete `mlp.down_proj.weight` matrix at each
  layer;
- each selected attention head: only the columns
  `[head_index * head_dim : (head_index + 1) * head_dim]` of that layer's
  `self_attn.o_proj.weight`.

No Q/K/V matrix, RMSNorm gain, embedding, bias, or other MLP matrix enters the
primary transplant. Those tensors remain descriptive maps unless separately
preregistered. Persist exact state-dict keys, slice bounds, shapes, dtypes, and
source-file hashes.

Read base and IT checkpoint values, cast each selected block once to float32,
and form `Delta_i = W_IT_i - W_base_i` in float32. For each alpha and direction,
construct the edited block from the untouched receiver snapshot in float32 and
cast once to the model dtype. Before behavioral scoring, alpha `1.0` base
transplant must be exactly equal after casting to the corresponding IT block,
and alpha `1.0` IT reversal must equal the base block. Persist elementwise
max error and SHA-256 for donor, receiver, and realized endpoints; any mismatch
aborts causal interpretation.

### Fixed matched-component universe

Each alternate six-component set contains:

- two MLPs at distinct layers sampled from layers 16-23 excluding 19 and 20;
- one non-target attention head at each exact layer 17, 18, 19, and 20.

The complete finite universe is all such six-sets, excluding any set containing
a selected component. Uniformly sample 39 unique sets without replacement
before behavioral evaluation. For every sampled set, use its own exact
IT-minus-base deltas on the same down-projection/head-output slices. Match each
target/null joint realized post-cast norm to the smaller natural full norm by
dose reduction from fresh snapshots; do not amplify either delta.

### Reproducible within-block geometry nulls

A within-block null preserves each selected `Delta_i`'s shape, Frobenius norm,
rank, and singular spectrum by independently permuting rows and columns and
applying independent Rademacher sign flips to rows and columns. It does not use
a newly sampled Gaussian matrix or a low-rank approximation. Apply one such
transformation to every block in each of 39 joint null replicates, then match
the target/null realized joint norm at the smaller natural norm without
amplification.

Derive both null streams with NumPy `SeedSequence`: let
`root = SeedSequence(4240959829)`, assign
`component_root, geometry_root = root.spawn(2)`, and use the 39 children from
`component_root.spawn(39)` and `geometry_root.spawn(39)` in order. Component
sampling and row/column permutation/sign draws use a fresh `default_rng(child)`
per replicate. Persist the complete universe, every child seed state, selected
set, permutation, and sign vector. Batch partitioning cannot change selection.

### Parallel/remainder estimands

For each selected block, define the frozen Gate-1 geometry `E_i` as the
theoretical float32 rank-one delta computed before model-dtype casting on the
accepted IT checkpoint by its set-of-record direction and component. Project
with the Frobenius inner
product:

`Delta_parallel_i = <Delta_i,E_i> / <E_i,E_i> * E_i`, and
`Delta_orth_i = Delta_i - Delta_parallel_i`.

Persist reconstruction error and require numerical orthogonality before any
behavior. Natural-dose parallel and orthogonal alpha curves are both reported;
the requirement that parallel reaches at least 50% of the full joint-delta
effect uses each decomposition at its own natural alpha `1.0`. The
parallel-versus-orthogonal specificity comparison separately dose-reduces the
larger joint delta to the smaller realized norm. Do not use a norm-matched dose
for the 50% decomposition fraction or a natural unmatched dose for the
specificity contrast.
