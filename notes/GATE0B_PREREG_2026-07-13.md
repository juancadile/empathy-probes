# Gate 0B Preregistration: T-Control Replacement and E27 Adjudication

**Status:** frozen before regenerated T/T-confirm model scores or E27 rejudging.

## Part 1: regenerated T/T-confirm controls

### Why

Historical T files contain 40 rows but 20 unique text pairs. The repaired Cartesian product adds the four missing opener/closing combinations per family. Historical results remain valid for the historical weighted stimulus set, but cannot certify the repaired control.

### Inputs

- Historical T and T-confirm files preserved and hashed before replacement.
- Repaired T: five development families x two openers x four closings.
- Repaired T-confirm: five disjoint confirmation families x the same 2x4 format crossing.
- Current Gemma model/direction/components only: `d_resid`; writers L19MLP,L20MLP; suppressors L18H13,L20H10,L19H12,L17H7.

### Protocol

1. Score both A/B orders for every pair and average the flip-corrected logit differences.
2. Evaluate baseline, writer edit, and suppressor edit on repaired T-confirm in one process with snapshots/restoration.
3. Use the same model revision, bf16/eager settings, prompt builder, and continuation-likelihood implementation as accepted E26/R1 where applicable.
4. Primary output per condition: family-mean edit effect on task persistence, five family effects, LOFO range, and family-bootstrap interval.
5. Run raw A/B and scaffold-free continuation likelihood. Chat A/B is a secondary format sensitivity.
6. Score historical T-confirm in the same process only as a diagnostic bridge; do not combine historical and repaired rows.

### Interpretation fixed in advance

- **Writer selectivity survives** if the repaired T-confirm effect remains smaller in absolute point estimate than one third of the simultaneously measured M-confirm writer effect under both primary readouts. This is a magnitude-selectivity rule, not proof of no task effect; report the T interval even if it includes zero.
- **Writer selectivity fails** if either primary T/M absolute ratio is at least one third. Do not replace T with another task cell after failure; use fresh WP1 controls for any new selection.
- Suppressor T co-movement is expected from the accepted arbitration interpretation. Report its sign/magnitude; do not apply a quietness gate or call it welfare-selective.
- Historical-to-repaired differences quantify format crossing sensitivity. No equivalence claim is made unless a separate equivalence margin is justified independently.

## Part 2: E27 UNKNOWN adjudication and score closure

### Inputs

- All existing E27 player `say` actions, not only the three UNKNOWN rows.
- Existing labels are hidden from adjudicators.
- Deterministic input IDs derived from raw-history hash, seed, step, and message hash.

### Adjudication protocol

1. Rejudge the full message set, not only UNKNOWNs, with a version-pinned independent model family and the versioned SUPPORT/CHAT/TASK rubric.
2. Randomize presentation order. Judge receives prior user context and player message only; never condition, variant, seed, edit status, existing label, or aggregate result.
3. Persist full requests/responses, attempts, parse failures, and labels.
4. Draw a stratified blinded human audit of at least 30 messages: 10 previously SUPPORT, 10 CHAT/TASK where available, and all prior UNKNOWNs (UNKNOWNs do not replace the 30). The human sees the same context/message fields and rubric.
5. Report model-vs-historical and model-vs-human confusion matrices and agreement. Resolve remaining label disagreements using a third adjudicator under a predeclared majority rule; never hand-edit silently.

### Primary statistic

For each paired seed:

`(suppressor - baseline)_distress - mean((suppressor - baseline)_excited, (suppressor - baseline)_resolved)`

Report all eight seed values, mean, seed-bootstrap interval as descriptive small-n uncertainty, and exact sign test. Also report total says and SUPPORT share by condition/variant.

### Interpretation fixed in advance

- Incremental distress specificity is supported only if the paired interaction is positive in at least 7/8 seeds and its effect is not driven by one seed under LOFO. The bootstrap interval and sign test are reported but not treated as well-calibrated large-sample inference.
- Otherwise the result remains unresolved or variant-general engagement. Do not increase seeds after seeing which side of the rule fails; a powered replication requires new preregistered seeds and is a separate experiment.
- Baseline distress selectivity and edit-specific distress selectivity remain separate claims.

## Deliverables

- Self-contained result bundles with input/model/prompt hashes and per-example arrays.
- Experiment-log entries preserving both historical and repaired results.
- Adversarial review before paper/showcase changes.
