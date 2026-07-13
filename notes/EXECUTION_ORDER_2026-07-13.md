# Scientific Execution Order (post-audit)

**Role:** operational companion to `ROADMAP.md` and `SCIENCE_AUDIT_2026-07-13.md`.
The roadmap preserves scope; this file fixes dependency order, decision gates, and claim ceilings.

## Rules of execution

1. Separate **discovery**, **selection**, and **confirmation** artifacts. Never choose a direction, layer, component, prompt, threshold, or statistic on a confirmation set.
2. Every run persists exact inputs, resolved configuration, model revision, prompt format, per-example outputs, grouping IDs, hashes, and failures. Summary-only JSON is not an accepted artifact.
   The experimental environment is exact-version locked; broad developer dependency ranges are not evidence of reproducibility.
3. A failed gate is a result. Do not alter the gate and rerun under the same experiment ID; log an amendment and use a fresh confirmation set when selection changes.
4. Use scenario family as the inferential unit. Template variants measure robustness within a family, not sample size.
5. Report effect sizes, all family effects, LOFO, and uncertainty. Monte Carlo controls must state their finite resolution and generator.
6. Name constructs at the level isolated by the stimuli. `d_resid` is an intervention axis, not a certified welfare direction.
7. Paper and showcase updates happen only after the corresponding result and adversarial audit are closed.

## Gate 0: Integrity before new claims

### 0A. Data and schema repairs

- Repair duplicated T/T-confirm variants using an explicit Cartesian product; preserve historical files and rerun T-confirm before replacing any accepted control result.
- Add duplicate assertions and row/unique-pair reporting to every deterministic builder.
- Require explicit/versioned component sets for every weight-edit script.
- Upgrade result schemas for capability, fractional ablation, E27 judging, and manipulation checks as specified in the science audit.

### 0B. Existing-result closure

- Blindly rejudge the three E27 UNKNOWN rows with full judge provenance; hash or commit all 48 raw histories.
- Regenerate `PROVENANCE.json` only after result schemas and accepted reruns are complete.
- Rerun Gemma fractional activation ablation with `d_resid`, raw and chat endpoints, sufficient random-direction controls, and same-artifact baselines.
- Rebuild manipulation checks with pinned prompts/models, raw responses, item IDs, randomized arm presentation, and an independent judge/human sample before calling axes validated.

**Gate 0 exit:** all accepted headline artifacts independently reconstructible; no stale/default component path; T and E27 controls finalized.

## Gate 1: Confirm what the parameter interventions localize

### 1A. Writer set

- Retain the current claim ceiling unless a fresh test changes it: a mid-late MLP band intervention, with L19 dominant and L20 the strongest complement, reduces the tested costly-helping choice.
- Do not claim a unique two-component circuit. The conditional 1/28 ranking becomes 2/28 after norm adjustment.
- Rerun the clean T-confirm control after deduplication and dual-order scoring.

### 1B. Suppressor set (R2b)

- Execute the fixed realized-norm-paired design from `SCIENCE_AUDIT_2026-07-13.md`.
- Preserve exactly one head from each of layers 17, 18, 19, and 20.
- Compute joint edit norm as `sqrt(sum(component Frobenius norm^2))`. Pair at the smaller natural full-edit norm and dose-reduce whichever set is larger; never over-edit target or null.
- Require monotone target dose response, paired target-minus-null family CI excluding zero, and LOFO stability.

**Gate 1 outcomes:**

- Pass: selected-head localization is supported conditional on the tested layer/type universe.
- Fail/inconclusive: retain only set-level direction specificity and high-cost trend; do not keep searching null pools on the same families.

## Gate 2: Identify the construct without task leakage

### 2A. Held-out controls (WP1)

- Create genuinely new families for task persistence, warmth, persona, motive, third-person recognition, and non-distress social engagement.
- Prefer matched-prefix, matched-length constructions. Free-form generations remain secondary stress probes.
- Freeze the cell interpretation and two-sided quietness gates before activation extraction.

### 2B. Decision-free welfare observation (WP3)

- Build a 2x2 welfare-salience x decision-opportunity battery.
- The observation contrast must use byte-identical task continuation and never mention interruption or helping in the decision text.
- Manipulation checks separately certify welfare salience and flat task pressure.
- Fit on development families; test transfer to M-confirm and quietness on T-confirm plus WP1 controls.

### 2C. Representation models (WP2/WP4c)

- Compare, under nested family CV: one mean-difference direction, nuisance-subspace residualization, and a jointly fitted task/welfare plane.
- Predeclare model selection and G/motive treatment. Use untouched confirmation families exactly once.

**Gate 2 outcomes:**

- Certified object found: it transfers behaviorally while remaining quiet on all held-out nuisance controls.
- Not found: publish linear welfare/task non-separability at block 20; do not iterate prompts against confirmation failures.

## Gate 3: Trace the current mechanism, not the historical shortlist

### 3A. Current component paths

- Treat historical restoration edges as hypotheses only. Rerun `L19MLP -> L20MLP` on M-confirm and regenerated T-confirm with dual-order and continuation readouts.
- Screen current L17-L20 components into L29-L41 consumers, guided by LB1/LB2. Freeze top edges before confirmatory restoration.
- Require both behavioral rescue and control quietness. Negative rescue is no edge evidence, not an inverse edge.

### 3B. SAE feature circuit

- Use pretrained Gemma Scope; do not train an SAE unless a documented coverage gap makes it necessary.
- First gate SAE fidelity: reconstruction error/explained variance, dead features, and stability across widths.
- Rank features using per-pair tail means on `d_resid` and M-confirm, not token-pooled development data.
- Profile candidates across M/T, V2.2, WP1 controls, and WP3 observation cells.
- Causally ablate/patch candidates against activation/frequency-matched random SAE features.

### 3C. Faithfulness and completeness

- Define a coarse circuit before evaluation.
- Circuit ablation must remove a predeclared fraction of the behavior effect.
- Circuit-only retention/outside-circuit ablation must preserve a predeclared fraction while retaining T/capability controls.
- If the graph is distributed, report bounded partial coverage rather than a full circuit.

**Gate 3 exit:** a current, held-out, behaviorally tested component/feature subgraph with explicit explained-effect coverage, or a documented distributed-mechanism negative result.

## Gate 4: Enacted behavior

### 4A. Prospective probing

- In the EIA harness, score prompt-final/action-selection activations before generated narrative or action tokens.
- Predict upcoming actions across seeds and scenarios with family/game-held-out evaluation.
- Compare against lexical, state-variable, and task-progress baselines.

### 4B. Realized cost environment

- Cross stated cost and mechanically incurred cost. Include active-objective/zero-loss and no-active-objective as separate conditions.
- Use distress, excited-positive, resolved-history, and neutral-information social states with matched requests.
- Analyze policy thresholds and trajectories, not raw `say` counts alone.

### 4C. Moral allocation

- Independently vary need, immediacy, incumbency, and switching cost.
- Include role-swap and simultaneous-arrival arms.
- Power and null-match for the expected small effect before interpreting absence.

**Gate 4 exit:** intervention effects transfer to action-space behavior with construct-appropriate controls, or the forced-choice/action dissociation is retained as the result.

## Gate 5: Generality and advanced methods

- Replicate only the surviving Gate 1-4 claims on Llama, then 2B/27B/32B as compute permits. Do not scale a failed construct label.
- Activation Oracles are triangulating readouts, never causal evidence.
- DAS/causal abstraction follows a fixed high-level variable model and untouched interchange tests.
- Jacobian-lens work is called transported-norm amplification unless the full workspace method is reproduced. It cannot rescue construct purity.
- Persona-vector comparison, tonic/phasic analysis, and steering asymmetry tests use Gate-2-certified stimuli or are labeled exploratory.

## Deliverable order

1. Integrity repair memo and tests.
2. Gate-specific preregistration committed before each run.
3. Raw result plus engineer interpretation.
4. Independent adversarial audit and, if needed, corrective rerun.
5. Experiment-log entry with the claim ceiling and failed alternatives.
6. Paper/showcase update only after steps 1-5.

The next executable sequence is: **Gate 0A -> T-confirm/E27 closure -> R2b -> Gate 2 stimuli -> current circuit/SAE work**. Paper rewriting follows the accepted results; it does not lead them.

## Current-state overlay (authoritative as of this audit)

This table supersedes stale checkboxes for execution purposes; it does not erase historical roadmap entries.

| Work item | State | Evidence / next action |
|---|---|---|
| A5 lexical battery | Complete | Shuffling exposes lexical saturation; matched cell M retains a deeper decision signal. |
| Gemma current direction/set derivation | Complete, claim-limited | `d_resid` plus current writer/suppressor sets; no welfare-purity certification. |
| R1 random-direction controls | Complete | Direction specificity passes only at 20-draw resolution; writer joint selectivity passes, suppressor joint selectivity fails. |
| R2 matched component slope null | Complete, inconclusive | `p=.080` and failed realized-norm balance; does not localize the slope to four heads. Run R2b. |
| R3 current-set need-by-cost profile | Complete at set level | Urgent-minus-resolved slope `+0.104 [0.064,0.146]`, 9/10 families; construct remains composite expressed-need status. |
| E27 current-set game variants | Run complete, scoring open | Three UNKNOWN labels and incomplete judge/raw-history provenance. Close under Gate 0. |
| Capability check | Run complete, provenance incomplete | No detected degradation on MMLU-400/WikiText; exact sample/parser provenance missing. |
| LB1 behavioral DLA | Complete, exploratory | Direct final-logit writers concentrate around L38-L41, not selected L17-L20 components. |
| LB2 trajectory | Complete, exploratory | Current edit effects amplify after hidden index 29. |
| LB3 SVD alignment | Complete, exploratory | Structural corroboration against one random-direction reference; not localization. |
| LB4/Jacobian lens | Failed before fit | Corpus duplication bug; method would estimate transported-norm amplification, not workspace membership. |
| Historical path restoration | Hypothesis-generating only | Old direction/component shortlist and development M. Only current-current edge is L19MLP to L20MLP; rerun required. |
| E20 SAE naming | Historical exploratory only | Old direction, development data, token pooling, no fidelity or causal feature test. |
| Sparse feature circuit | Not started | Gate 3B after held-out construct controls are available. |
| Faithfulness/completeness | Not started | Gate 3C. |
| WP1 held-out gate families | Not started | Gate 2A; blocks all future purity claims. |
| WP2/WP3/WP4 | Not started | Follow Gate 2 order; freeze stopping rule before confirmation. |
| E22b moral redesign | Not started | Existing/current-set rerun on confounded v1 axis does not clear the redesign gate. |
| Fractional cross-model ablation | Historical result, rerun required for Gemma | Existing exact 88% predates `d_resid` and lacks self-contained provenance. |
| Paper/showcase | Draft exists, scientifically stale | Rewrite only after Gate 0, R2b, and accepted claim ceilings; resolve open review comments. |
| Provenance manifest | Stale | Regenerate after accepted repairs/reruns, including raw E27 histories or external hashes. |
