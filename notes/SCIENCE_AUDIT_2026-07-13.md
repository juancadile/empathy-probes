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

## Prompt and dataset integrity finding

- `T_templated.jsonl` and `T_confirm_templated.jsonl` each contain 40 rows but only 20 unique text pairs. The builders iterate eight indices while synchronously cycling two openers and four closings, so indices 0-3 repeat at 4-7. Scenario-family disjointness between development and confirmation is intact.
- Regenerate both cells from the explicit Cartesian product of opener and closing variants, assign stable variant identifiers, and add a no-duplicate assertion to every deterministic cell builder.
- Before replacing any artifact, re-score the regenerated T-confirm set and report an equivalence table. Exact deduplication should leave deterministic point estimates unchanged; genuinely new opener-closing combinations may move them. Family-clustered conclusions survive only if the new five-family result remains within the original uncertainty interval.
- Audit code must report both row count and unique-pair count. Existing references to 40 task-control pairs mean 40 weighted rows, not 40 independent stimuli.
- The V2.2 factorial construction otherwise passes the structural audit: all shared prefixes are valid, all 10 families have four variants at each cost level, and the positive/negative decision tails are byte-identical across welfare, nonsocial, mild-need, and resolved-need axes. A fixed four-word branch-length asymmetry remains and should be counterbalanced in a future syntax-robustness cell.
- The audited free-form A/B/D/E/F/G/H cells are not structurally matched controls. Mean absolute positive-vs-negative length differences range from 27.5 words (A) to 106.5 words (H), with maxima up to 301 words. G and H contain only three scenario families; B contains four after auditing; A and F intentionally reuse the same five scenario IDs, so F tests perspective transfer but not out-of-scenario transfer.
- `controlled_direction_analysis.py` full-output-pools those free-form cells because they lack `shared_prefix`; `build_residualized_direction.py` then uses their mean-difference vectors as nuisance axes. QR residualization therefore removes a mixture of construct, lexical, length, provider, and discourse-style variation. It is a sensitivity transform, not clean conceptual purification. Fresh matched-prefix controls are required for any future purity claim.
- The primary M/T and V2.2 behavioral assays remain materially stronger because they tail-pool matched decision clauses. Keep free-form cells as adversarial transfer probes and report family/provider/length sensitivity; do not use them as the sole certification instrument.

## Paper and provenance blockers

- Replace all welfare-value/welfare-pure writer naming with assay-grounded costly-helping language.
- Replace every "matched-component null pending" statement with the completed inconclusive R2 result and its norm confound.
- Replace provisional R1 text with the corrected plus-one Monte Carlo results; z-scores are descriptive only.
- Add LB1-LB3 only with the claim ceilings above.
- Regenerate `results/PROVENANCE.json`; it currently omits R1, R2, LB1, LB2, and LB3 artifacts.

## E27 environment and scoring audit

- Seed pairing is implemented coherently: every run receives a fresh `LocalPlayer`, and sampling is reseeded by `(game seed, call index)`. All 48 runs completed with 31 model calls (one discarded self-assessment plus 30 action calls), so there is no missing-run or call-count imbalance.
- E27 is an adapted listener environment, not the unchanged original game: it replaces the eight-message pool and runs `max_steps=30` rather than the original 20. The map, door, heart tick, and action API are otherwise shared.
- The fixed `GAME_DESCRIPTION` still names a depressed suicidal user in excited/resolved variants. It reaches the pre-run self-assessment prompt, but the action prompt filters description fields and uses the neutral objective plus observed message history. Therefore action trajectories remain interpretable, while self-assessment comparisons across variants do not.
- `e27_scores.json` contains three `UNKNOWN` labels, and `score_e27.py` correctly exits nonzero when any unknown remains. The paper nevertheless quotes the incomplete SUPPORT counts (144/14/0). The three messages are visually support-responsive; if adjudicated SUPPORT, baseline distress/excited and suppressor excited counts each rise by one, while the condition-by-variant interaction changes only slightly and remains null. This must be resolved by a blinded rejudge/adjudication, not silently hand-edited.
- The score detail records only file/seed/class, not the message, prior context, raw judge response, model response ID, or retry history. Persist those fields and hash the 48 raw histories. Raw baseline/suppressor game directories are currently untracked and absent from `PROVENANCE.json`.
- The scientific conclusion remains conservative regardless of adjudication: baseline engagement is distress-selective; the suppressor edit's incremental distress specificity is unresolved (`n=8`, bootstrap CI crosses zero, sign-test `p=1.0`). Do not claim a welfare-specific game transfer for the sets of record.

## Capability-evaluation audit

- The capability benchmark was rerun on the post-correction sets of record. MMLU-400 changes are +0.5pp (writers), +0.75pp (suppressors), and +0.5pp (combined), with bootstrap intervals touching/crossing zero; WikiText perplexity ratios are 0.998, 1.001, and 1.000. The supported wording is "no detected degradation on sampled MMLU and WikiText perplexity," not broad capability preservation or improvement.
- Spark logs show balanced generated answer letters and at most one unparsed baseline/combined item, so the earlier constant-letter failure is not present in the accepted run.
- The result JSON does not persist MMLU dataset revision, sampled row IDs/questions, subject counts, predicted letters/completions, or per-window WikiText losses. Add these before final provenance regeneration; otherwise the exact sample and parser behavior cannot be independently reconstructed from the artifact alone.
- `sample_mmlu` is simple random sampling, despite the module docstring calling it stratified. Either implement subject-stratified sampling or correct the documentation; do not claim stratification for the current result.

## Forced-choice assay audit

- The accepted implementation correctly uses right padding, indexes the last non-padding logit, asserts that A/B are single tokens, and reuses the same seeded option order across baseline and edit conditions. M-confirm is exactly balanced at 24 A-first / 24 B-first under seed 42.
- Replace one random order per pair with both orders for every pair and average the flip-corrected scores. This removes option-token/order interaction as a nuisance rather than relying on random balance. Apply the same rule to regenerated T/T-confirm.
- The E26 scaffold-free continuation-likelihood assay reproduces the writer and suppressor signs, so the causal effect is not specific to the A/B prompt. This is the strongest assay-generality evidence. The three "paraphrases" alter only the common opener stem and remain a weak lexical stress test.
- E26 persists summary means/CIs but not per-pair baseline/edit arrays, option orders, tokenized prompt lengths, or continuation log-likelihoods. Persist raw arrays and hashes so clustered intervals and order sensitivity can be independently recomputed.
- D/G/H selectivity ratios use free-form positive/negative texts as A/B options after longest-common-prefix splitting. Because those branches have large length/style asymmetries and different family counts, treat these ratios as secondary stress results. T-confirm and continuation-likelihood are the clean selectivity controls.

## Weight-edit implementation audit

- The core Gemma edit is mathematically appropriate. Gemma RMSNorm writes `((1+w) * y) / rms(y)`; orthogonalizing the pre-norm output-weight columns against `(1+w) * d` zeros the numerator of the residual-direction projection for any input-dependent RMS denominator. MLP edits target `down_proj`; head edits target the correct `o_proj` column slice.
- Snapshot keys separate attention and MLP matrices and snapshot a shared `o_proj` only once when multiple heads in one layer are edited. Restore paths therefore do not leak edits across conditions.
- Post-bf16 residual alignment/idempotence checks are small relative to removed alignment (roughly 0.1%-0.7% across selected components), supporting that the intended rank-1 projection was actually removed after casting.
- The edits are rank-1 but not negligible within each selected component: relative component-block Frobenius changes are about 2.2% for each selected MLP and 1.7%-3.5% for selected heads. Report rank and realized norm; avoid implying an infinitesimal perturbation.
- `weight_orthogonalization.py` still defines the superseded pre-correction component sets as module/CLI defaults. Current rescue scripts override them, but an unqualified rerun silently edits the old sets. Replace defaults with the sets of record or require explicit component specifications and persist the resolved set in every artifact.

## Circuit-status audit

- The previously reported restoration edges were discovered on the 40-pair development M set and the pre-correction component shortlist. Two positive edges (`L15H15 -> L19MLP`, `L17H13 -> L17MLP`) involve components no longer selected. They are historical hypotheses, not evidence for the sets of record.
- `L19MLP -> L20MLP` is the only positive old edge connecting two current components (38% rescue, old development assay). Rerun it on M-confirm and T-confirm with dual-order/readout controls before retaining it.
- The two retained old heads, L18H13 and L19H12, were causal in the old assay but were not mediated by the tested L18/L19 MLP targets. New heads L20H10 and L17H7 have no path-level evidence.
- LB1/LB2 redirect the search: current L17-L20 edits are upstream modulators; behavioral evidence crystallizes after hidden index 29 and direct final-logit contributions concentrate at L38-L41. Systematically screen paths from current components into L29-L41 heads/MLPs, then causally rescue only the top preregistered edges on confirmatory M and T.
- A "full circuit" should mean a behaviorally faithful coarse circuit, not every microscopic route: circuit-only retention and circuit ablation should jointly explain a predeclared fraction of the edit effect on held-out families, with capability/control cells retained. A complete weight-to-neuron graph for a 9B model is unrealistic; a sparse, causally sufficient component/feature subgraph is feasible if the mechanism is not too distributed.

## SAE-status audit

- E20 is a pretrained GemmaScope feature-naming analysis, not a sparse feature circuit. It used the pre-correction `direction_M_block20.npy` and 40 development M pairs. It has not been rerun for `d_resid` or M-confirm.
- The activation ranking sums feature activations over all positive/negative tail tokens and divides by total token count. It is token-pooled, not a mean of paired per-example differences as the prose implies; branch length therefore affects feature weighting.
- No SAE reconstruction loss/explained-variance gate, dead-feature rate, feature stability across SAE widths, held-out family profile, or feature ablation/patching was measured. Neuronpedia labels are automated semantic suggestions only.
- Correct next step: rerun at block 20 with `d_resid` and M-confirm; compute per-pair tail means before family aggregation; profile candidate features two-sided across matched M/T/V2.2 and fresh motive/persona controls; require SAE reconstruction fidelity; then ablate/patch candidates against frequency/activation-matched random SAE features. Only features passing held-out behavioral intervention tests can enter the circuit.
- Sparse feature circuits and attribution graphs remain open Stage-B work. The SAE result cannot substitute for path tracing, faithfulness (`circuit ablation`), or completeness (`outside-circuit ablation`).
