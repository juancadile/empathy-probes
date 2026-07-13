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

If lineage or tensor correspondence cannot be established, restrict the analysis to cross-checkpoint descriptive geometry and do not call the difference an instruction-tuning delta.

## Checkpoints and sets

- Base: exact revision of `google/gemma-2-9b`.
- Instruct: exact accepted revision of `google/gemma-2-9b-it`.
- Current writers: L19MLP, L20MLP.
- Current suppressors: L18H13, L20H10, L19H12, L17H7.
- Matched component controls are fixed before behavior evaluation and preserve type/layer/norm structure.

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

- layer/type/norm-matched component deltas;
- random rank-matched projections of each selected delta;
- parameter-count and realized-joint-norm matched edits;
- sham snapshot/restore condition;
- tokenizer/prompt parity and parser gates.

All fractional edits are constructed from the original checkpoint snapshot, never applied cumulatively.

## Behavioral evaluation

Primary confirmation sets:

- sealed `C3-confirm` and matched task controls;
- a separately reserved C3 factorial confirmation subset if a Gate-2 construct
  transfer claim is attempted;
- a fresh, frozen capability sample for transplant/reversal conditions.

M-confirm2, WP3-confirm, and R2b families are development/diagnostic transfers
because their outcomes are known before C3. They cannot satisfy C3 confirmation.

Report base and IT baselines before interventions. A transplant cannot be interpreted when the receiving checkpoint lacks assay headroom or prompt validity.

## Primary contrasts

1. Base transplant slope: does adding selected IT delta move base behavior toward the IT baseline monotonically?
2. IT reversal slope: does removing it move IT toward base monotonically?
3. Bidirectional consistency: transplant and reversal effects have opposite signs under a common metric.
4. Specificity: selected delta effect exceeds matched component/rank controls and remains bounded on task/capability controls.
5. Geometry: direction-aligned delta explains a predeclared fraction of the full selected-delta effect; otherwise the mechanism change is not reducible to the current rank-1 axis.

Family-level effects and LOFO are mandatory. Thresholds for “predeclared
fraction” are fixed on `C3-dev` before `C3-confirm` is opened. At least 39
realized-norm-matched random rank/component controls are frozen where an
empirical rank is used; finite resolution is reported.

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
