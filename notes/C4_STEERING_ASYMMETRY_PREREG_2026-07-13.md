# C4 Preregistration: Steering Asymmetry and Collapse

**Status:** frozen before rerunning steering on the audited direction/components.

## Question

Why can positive steering along a costly-helping direction alter behavior while negative steering causes instability, collapse, or weak control?

Competing explanations:

1. true asymmetric policy geometry;
2. off-manifold residual displacement or norm distortion;
3. the direction mixes welfare, task interruption, and lexical/action content;
4. nonlinear decoder/logit-softcap saturation;
5. distributed downstream consumption makes subtraction destructive;
6. prompt-format or sampling thresholds create an apparent asymmetry.

No moral-character or workspace interpretation is attached before these alternatives are tested.

## Prerequisites

- Gate 0 provenance/schema repairs complete.
- Use current `d_resid` only as an impure costly-helping intervention axis.
- Repeat with Gate-2 `d_N`/subspace only if it passes held-out certification.
- Generate 16 `C4-dev` and 16 sealed `C4-confirm` costly-helping families with
  matched task controls. M-confirm2 is historical robustness only.
- Model revision, block, hook semantics, prompt format, and generation settings are frozen.

## Intervention families

At the frozen block and token positions, compare:

### Additive steering

`h' = h + alpha * d`

with symmetric signed doses calibrated in units of the development activation-projection SD, not arbitrary raw coefficients.

### Projection scaling

`h' = h + alpha * (h dot d) * d`

which amplifies/reduces the existing coordinate rather than injecting a fixed vector.

### Orthogonal removal

`h' = h - beta * (h dot d) * d`, `beta in [0,1]`

separated from extrapolating beyond removal (`beta>1`), which is explicitly off-manifold-risky and secondary.

### Norm-preserving rotation

Move the residual toward/away from `d` while preserving per-token residual norm. Match angular displacement between signs.

### Distribution-matched patching

Patch projection values from naturally high/low development examples matched on scenario family, token position, and residual norm.

## Controls

- norm- and variance-matched random directions;
- task-persistence and lexical/action directions;
- sham hook;
- opposite token positions and nondecision layers;
- current component weight edits, which alter the source rather than injecting activations;
- raw versus chat prompt formats analyzed separately.

Use at least 64 frozen random directions for the primary geometry/control
comparison. Direction seeds, dose matching, and finite resolution are fixed
before `C4-confirm`.

## Dose calibration

1. On development families, measure natural projection quantiles and residual norms at intervention tokens.
2. Freeze doses corresponding to natural quantile shifts plus one explicitly extrapolative dose.
3. For each signed pair, match realized residual-norm change and angular displacement where the intervention family permits.
4. Do not choose doses from behavioral success. A parser/finite-logit safety gate may remove a dose symmetrically for both signs before confirmation is opened.

## Diagnostics at every layer

- projection onto target, task, lexical, persona, and matched random directions;
- residual norm and angle from baseline;
- action-token/logit trajectory before and after softcap;
- KL divergence and entropy at the next-token distribution;
- invalid-token/parser rate, repetition, response length, perplexity;
- LB1/LB2 late-consumer trajectory, especially hidden indices 29-41;
- transported-norm amplification only if the corrected Jacobian analysis runs.

## Behavioral outcomes

- both-order `C4-confirm` forced choice;
- scaffold-free continuation likelihood;
- fresh task controls and WP1 nuisance cells;
- WP3 agency/current-need interaction if certified;
- prospective EIA action logits and sealed trajectories under the action-space
  preregistration. C4 doses/methods must be frozen on `C4-dev` before the EIA
  confirmation partition is opened; otherwise EIA results are diagnostic only.

All outcomes are reported for every dose; no “best alpha” headline is selected on confirmation data.

## Primary asymmetry estimands

For matched positive/negative doses:

1. behavioral antisymmetry error: `effect(+a) + effect(-a)`;
2. geometric asymmetry after matching norm/angle;
3. excess KL/perplexity/invalid-rate under negative versus positive steering;
4. task-control co-movement relative to M effect;
5. difference between additive steering and natural-distribution patching.

Use family-level paired effects, LOFO, and full dose curves. Small family counts make intervals descriptive.

## Decision table

| Pattern | Interpretation |
|---|---|
| Asymmetry disappears under norm-preserving/natural patching | additive off-manifold artifact |
| Negative instability occurs for random/task directions too | generic residual disruption |
| d_resid asymmetric but certified d_N is not | impurity/task-mixture explanation |
| Geometry matched, controls quiet, behavior remains asymmetric | genuine local policy nonlinearity candidate |
| Late layers amplify only negative perturbations with broad KL | distributed downstream-consumer instability |
| Forced choice shifts but trajectories do not | sampling/action-threshold dissociation |
| Weight edits replicate asymmetry without activation norm drift | source-level mechanism more plausible |

## Claim ceiling

“Steering resistance,” “entrenchment,” “breaking character,” or “broadcast workspace” language requires a surviving geometry-matched, control-specific mechanism. Otherwise report signed intervention asymmetry under the tested protocol and its measured failure mode.

## Frozen amendment (2026-07-13, before audited steering runs)

The original design reused M-confirm2 and left random-control resolution open.
Dedicated C4 families and 64 frozen random directions now separate dose/method
selection from confirmation.
