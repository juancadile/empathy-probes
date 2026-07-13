# Stage D Preregistration: Scale and Family Generality

**Status:** frozen before new scale-model activation/component runs.

## Questions

1. Does a confirmed costly-helping/current-welfare mechanism recur within a model family as size changes?
2. Does its attribution and causal effect become more or less concentrated?
3. Which claims transfer across architectures, and which are Gemma-specific?

Decodability saturation is not the scaling target. A model enters the mechanistic scaling analysis only after passing the same behavioral representation and assay-headroom gates.

## Model strata

### Same-family scale stratum

Use instruction-tuned checkpoints from one documented family with at least four sizes if available and architecturally comparable. Gemma checkpoints are primary; exact revisions and training lineage are persisted.

With fewer than four comparable sizes, report a size comparison, not a scaling law or fitted exponent.

### Cross-family stratum

Llama/Qwen comparisons are architecture/training-family replications, never extra points on the same scaling curve.

### Base/instruct stratum

Base-versus-instruct is a tuning comparison governed by C3, not a size point.

## Common protocol

- Generate a common 16-family `D-scale-dev` and 16-family sealed
  `D-scale-confirm` assay with matched task controls and Gate-2-compatible
  currentness/persona factors. M-confirm2 and earlier Gate-2 confirmation sets
  are diagnostic anchors only.
- Use each model's documented interface; raw and chat protocols remain separate.
- Select direction/block on development families with grouped CV and a relative-depth tie rule; freeze before confirmation.
- Require confirmation headroom, behavior transfer, and nuisance-control gates per model.
- Exact hook equivalence/reconstruction checks are mandatory for each architecture.
- Quantization is disallowed for primary activation/component comparisons. Any 70B 8-bit result is coarse sensitivity only.
- Open each model's `D-scale-confirm` outputs once. A model that fails headroom
  or construct gates on development is excluded before confirmation, not tuned
  against confirmation.

## Comparable concentration metrics

Compute writer and suppressor signs separately from family-aggregated component effects.

### Attribution concentration

For absolute component contributions `a_i` among the preregistered component universe:

- normalized top-50% and top-80% counts: `k50/N`, `k80/N`;
- participation ratio: `N_eff = (sum |a_i|)^2 / sum a_i^2`;
- normalized participation: `N_eff/N`;
- Gini/entropy of normalized absolute contribution mass;
- layer-relative distribution of mass.

Raw `k50`/`k80` is reported but never compared alone across models.

### Causal concentration

- Single-component or preregistered grouped ablation effects on held-out behavior.
- Fraction of total frozen circuit-edit effect recovered by top `q%` of available components.
- Minimal normalized component fraction needed to recover 50%/80% of the effect, with family uncertainty.
- Attribution-versus-causal rank agreement.

No “sparse circuit” conclusion comes from attribution mass alone.

The primary scale metric is signed causal effect coverage from the top 10% of
the preregistered component universe, `coverage10_causal`. Component order and
the universe are frozen on `D-scale-dev`; the restored/ablated effect and source
intervention denominator are recomputed within every `D-scale-confirm` family.
Do not clamp values below 0 or above 1. A model whose source intervention fails
its own confirmation/headroom gate is a failed mechanism replication and has no
comparable concentration value; this exclusion is reported rather than
silently dropped from a trend.

All other attribution/causal concentration metrics are mandatory secondary
descriptions and cannot replace `coverage10_causal` after confirmation.

### Feature concentration

SAE feature counts are comparable only within matched SAE release/width/sparsity regimes or after explicit normalization for dictionary width, activation density, and reconstruction fidelity. Different SAE suites are not raw-count scaling points.

## Generality tests

Freeze the Gemma-derived hypothesis before each family is opened:

- relative-depth band of upstream modulation;
- late-consumer amplification pattern;
- writer/suppressor signed intervention profile;
- need-by-cost interaction;
- direction dependence under full/fractional ablation.

Test the hypothesis without selecting homologous components from confirmation behavior. Discovery of a different mechanism in another family is a new exploratory phase requiring its own confirmation.

## Statistical analysis

- Scenario family is the unit; all models evaluate the same sealed family IDs where tokenizer/context permits.
- Report within-family paired model differences and LOFO.
- Fit a size trend only after the minimum same-family model count and protocol gates pass.
- With few sizes, avoid asymptotic/exponent language; show points and uncertainty.
- Do not use parameter count as the only predictor when layer/head/width architecture changes; report these covariates descriptively.

On `D-scale-dev`, freeze the predicted sign of the `coverage10_causal`
association with log parameter count. On the common confirmation families, fit
one paired family-level linear slope of `coverage10_causal` on log parameter count and bootstrap
families jointly across models with master seed `291112035` (first 32 bits of
SHA-256 of `D scale family bootstrap v1 2026-07-13`). A bounded monotonic scale
trend requires all of:

1. at least four comparable same-family sizes pass their individual gates;
2. the family-bootstrap 95% interval for the slope excludes zero in the frozen
   development direction;
3. aggregate point estimates are monotone in that direction; and
4. every leave-one-model-out slope retains the direction.

Failure of this primary metric cannot be rescued by selecting Gini, entropy,
attribution `k80`, or a cross-family subset. Those remain descriptive.

## Decision table

| Outcome | Supported claim |
|---|---|
| Same mechanism passes in >=4 same-family sizes and primary `coverage10_causal` slope gate passes | bounded within-family monotonic scale trend |
| Mechanism passes but concentration is non-monotone | replication without scaling law |
| Representation passes, component intervention fails | decodability conserved; causal implementation differs |
| Another family requires new direction/components | family-specific mechanism, not transfer |
| Quantized large model only | coarse validation, excluded from primary trend |
| SAE fidelity differs materially | no feature-count comparison |

## Claim ceiling

Report exactly which representation, intervention, and construct gates transferred. “Universal empathy circuit,” “scaling law,” and raw component-count comparisons are prohibited without the corresponding evidence above.

## Frozen amendment (2026-07-13, before scale-model runs)

The original common protocol reused M-confirm2 and Gate-2 confirmations after
their outcomes were known. Dedicated common scale development/confirmation
families now support model-wise selection and one-time replication without
recycling earlier confirmation labels.

The primary scale metric is now `coverage10_causal`, with its sign frozen on
development and a paired family-bootstrap slope test on confirmation. Secondary
concentration metrics cannot be selected post hoc as the scaling headline.

## Frozen comparability amendment (2026-07-13, before scale-model runs)

The earlier `coverage10_causal` text did not define a common source intervention
whose effect could be recovered across architectures. The primary metric is now
fixed as **marginal component-ablation concentration** under the following
architecture-neutral assay. It is not the joint mediation/sufficiency coverage
defined for B3.

### Component universe and intervention

For model `m`, the universe `U_m` contains every attention head and one MLP
residual-output component at every transformer block. It excludes embeddings,
norm gains, unembedding, and individual MLP neurons. Persist the exact mapping
and `N_m = sum_l (n_heads_l + 1)`.

At the final non-padding prompt token of the common dual-order forced-choice
prompt, replace one component's residual contribution with its
`D-scale-dev` mean for the same model, block, component, prompt format, and
token role. For attention, the component is the individual head contribution
after its output projection into residual space; for the MLP it is the complete
MLP residual add. No confirmation activation enters an ablation mean. Numeric
head-sum and residual-reconstruction checks must pass per architecture before
behavioral scores are opened.

For confirmation family `f` and component `i`, define signed marginal effects
`eM[f,i] = margin_clean[f] - margin_ablate_i[f]` on the costly-helping assay and
`eT[f,i]` analogously on its matched task control, averaging option orders and
nested surface variants within family first. Every component is evaluated on
all 16 confirmation families; missing components or favorable-family subsets
invalidate that model's concentration point.

### Development selection and primary metric

On development only, rank all components by
`selectivity_i = abs(mean_f eM[f,i]) - 3*abs(mean_f eT[f,i])`, descending;
ties use larger absolute M effect, earlier relative depth, attention before MLP,
then numeric head ID. Freeze
`K_m = ceil(0.10*N_m)` components. If the K-th selectivity is not positive, the
model fails the selective-component headroom gate and has no primary scale
point.

On confirmation, compute from the frozen set:

`coverage10_causal[m] =
    sum_(i in topK_m) abs(mean_f eM[f,i]) /
    sum_(i in U_m) abs(mean_f eM[f,i])`.

Do not clip or replace absolute-of-family-mean with mean-absolute-family effect.
The denominator must be positive with a family-bootstrap 95% lower bound above
`1e-8`. Report family-bootstrap uncertainty by recomputing every component mean
and the ratio inside each shared family resample.

The frozen top set must also satisfy both selectivity checks on confirmation:

1. its summed marginal task mass is below one third of its summed marginal
   costly-helping mass; and
2. joint mean-ablation of the full top set has the development-predicted sign,
   a family-clustered 95% interval excluding zero, and task-effect magnitude
   below one third of its costly-helping effect.

The joint effect is a nonadditivity sensitivity and is reported beside the sum
of marginals; it never replaces `coverage10_causal`.

### Finite random-set check

For each model, sample 255 uniform size-`K_m` component sets from `U_m` without
replacement within set. The master phrase is
`D scale component set nulls v1 2026-07-13` with integer seed `2158476156`;
derive a model-specific PCG64 seed as the first 64 bits of SHA-256 over the
master phrase, exact model ID, and resolved revision. Persist all sets and
seeds. The frozen top set's confirmation marginal mass must exceed the random
set envelope at plus-one rank at most `12/256`. Otherwise report descriptive
concentration without a selectively localized mechanism point, excluding it
from the primary size slope.

### Scaling-language ceiling

`coverage10_causal` now means the concentration of held-out **marginal
single-component ablation sensitivity** in a development-selected 10% set. It
does not measure Shapley value, account fully for component interactions, or
establish B3-style circuit completeness. The same-family monotonic-slope gate
remains unchanged, but any trend must use this full label and all model
exclusions must remain visible.
