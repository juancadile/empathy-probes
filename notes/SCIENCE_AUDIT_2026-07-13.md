# Science Audit: Rescue3c Claim Ceiling and Next Tests

**Date:** 2026-07-13  
**Scope:** experiments, stimuli, prompts, environments, causal interpretation, and result provenance. Bibliography and presentation are out of scope.

## Findings that currently survive

1. **Writer-set intervention:** editing L19MLP+L20MLP along the residualized block-20 direction robustly lowers the tested costly-helping choice. The effect transfers across raw A/B, chat-templated A/B, clause-opener variants, and continuation likelihood. On the corrected realized-norm random-direction control, the helping effect and the joint helping-vs-task statistic both clear the 20-draw control at its minimum attainable plus-one p-value (`1/21 = .0476`).
2. **Suppressor-set intervention:** editing the four selected heads raises the costly-helping readout, while task persistence moves in the opposite direction at nearly the same magnitude. The helping change is direction-specific at the minimum attainable 20-draw p-value, but it is not selective under the joint helping-vs-task statistic (`p = 1.0`). This is an arbitration effect, not a welfare-only edit.
3. **Set-level need-by-cost profile:** the suppressor edit's cost slope increases from resolved to urgent need; the paired urgent-minus-resolved slope is `+0.104 [0.064, 0.146]` with 9/10 families positive. Same-head random directions do not reproduce the slope (`0/20`, plus-one `p=.0476`).
4. **Lower-level location:** LB1 shows the edited L17-L20 components are not direct final-logit writers. The dominant direct contributions occur around L38-L41. LB2 shows the edit effect is small near L20 and amplifies after hidden index 29. The current mechanistic claim is therefore "upstream modulators feeding unresolved late paths," not a complete circuit.
5. **Structural corroboration:** the selected MLPs and three of four suppressor heads contain more of the residualized direction in their leading singular subspaces than one seeded random-direction reference. This is exploratory corroboration, not independent localization.

## Claims that do not currently survive

1. **No welfare-pure direction is certified.** The residualized direction fails held-out task certification and an in-sample motive gate. It can be used as an intervention axis, but not named a welfare representation.
2. **No need-gated writer label.** The writer resolved-need effect is nonzero and fails the post-audit exploratory band; the fixed random comparator displays the same-sign need contrast. The supported result is a strong, task-preferential costly-helping intervention effect.
3. **No four-head localization of the suppressor slope profile.** The matched-component null is inconclusive (`p=.080`) and fails its norm-balance gate: null-set realized norms are 11.5%-31.6% smaller than target, with slope-vs-norm correlation about `-0.72`.
4. **No welfare-specific game-edit effect for the sets of record.** Baseline engagement is strongly distress-selective, but the edit-by-variant interaction is unresolved at eight seeds.
5. **No workspace/J-space result.** LB4 failed before fitting. Even a successful Jacobian-lens run would measure transported-norm amplification, not workspace membership.

## R2b: realized-norm-paired component specificity

**Question:** Is the positive need-by-cost slope unusually tied to the selected four heads, rather than to applying a larger direction-removal edit?

### Fixed design

- Preserve the target layer multiset exactly: one attention head from each of layers 17, 18, 19, and 20.
- Exclude the targeted four heads and any set overlapping them from the primary null universe. Report a secondary all-nontarget universe if feasible.
- For each null set, apply the natural full (`alpha=1`) target-direction orthogonalization and measure its realized post-bf16 total Frobenius norm `N_j`.
- On the targeted four heads, solve for a fractional dose `alpha_j in [0,1]` whose realized post-bf16 total norm matches `N_j` within 3%. Do not scale a null beyond full orthogonalization; this avoids over-removing and reversing its native direction projection.
- Evaluate the same family-paired urgent-minus-resolved cost-slope statistic for the null set and its norm-paired targeted dose in the same process and prompt format.
- Primary statistic per null set: `D_j = slope(target at matched norm N_j) - slope(null at N_j)`.
- Primary inference: family-clustered bootstrap CI for the mean paired difference across the fixed null-set sample, with null-set resampling reported only as a sensitivity analysis because all comparisons share the same target set.
- Report the target dose-response curve and verify monotonicity over the observed null-norm range. If monotonicity fails, the norm-paired comparison is not interpretable and component localization remains unresolved.
- Report natural full-edit target rank among all evaluated sets descriptively, but do not call it an exact permutation test unless the complete predeclared universe is enumerated.

### Decision rule

Head-set localization is supported only if the paired target-minus-null CI excludes zero, the target dose response is monotone over the matched range, and no result depends on excluding a single scenario family. Otherwise retain the set-level/direction-specific claim only.

## LB4 retry: local-corpus Jacobian lens

- Build a network-independent corpus from deduplicated shared prefixes across V2.1 and V2.2; do not duplicate a 48-item cell to fake 300 examples.
- Balance by cell, scenario family, and provider where those labels exist. Persist the sampled row IDs and corpus hash.
- Keep all decision branches out of the fit corpus; use prompt/context prefixes only.
- Run a small corpus-size sensitivity (`n=64, 128, all available`) before the final fit.
- Report transported-norm amplification for the residualized direction and matched random directions. Do not use "workspace," "J-space membership," or consciousness-adjacent language.
- This remains exploratory and cannot rescue welfare specificity or component localization.

## Fresh construct-validity work

1. Generate genuinely held-out scenario families for task persistence, motive/persona, warmth, and distress. None may be used for residualization, layer selection, component selection, or prompt tuning.
2. Re-test direction transfer two-sided and behaviorally. Direction purity requires both activation-level quietness and edit-level selectivity on fresh families.
3. Redesign the moral-moral axis before revalidation: two simultaneous welfare claims, controlled incumbency, independently varied urgency and action cost, byte-matched decision syntax where possible. The stale E22 axis is descriptive method history only.
4. Treat taskless and no-cost helping as separate cells. "Do whatever you want" still induces an action-selection objective; the scientifically useful distinction is not task/no-task wording but whether helping competes with an externally specified objective, another welfare claim, or nothing measurable.

## Paper and provenance blockers

- Replace all welfare-value/welfare-pure writer naming with assay-grounded costly-helping language.
- Replace every "matched-component null pending" statement with the completed inconclusive R2 result and its norm confound.
- Replace provisional R1 text with the corrected plus-one Monte Carlo results; z-scores are descriptive only.
- Add LB1-LB3 only with the claim ceilings above.
- Regenerate `results/PROVENANCE.json`; it currently omits R1, R2, LB1, LB2, and LB3 artifacts.

