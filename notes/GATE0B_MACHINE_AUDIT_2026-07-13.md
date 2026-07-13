# Gate 0B Machine Work: Independent Audit

## Task-control replacement

**Artifact:** `results/gate0b_task_control_accepted_20260713/task_control.json`

**SHA-256:** `5aba0c2b287e65799ace10de195f2eec28e37a94a4652de49d9ed19805d651eb`

**Executed source:** `09e8ca40b3cbf86c2d6cb77fde2be4be15cc92ce`

**Integrity verdict:** PASS. The artifact validates, binds to clean source and
the immutable Gemma revision, matches every frozen direction/cell hash, contains
all per-pair and dual-order scores, restores every edited tensor exactly, and
recomputes exactly from persisted scores and family IDs.

| Readout | Writer M effect | Repaired T effect | abs(T)/abs(M) | Gate |
|---|---:|---:|---:|---|
| Raw dual-order A/B | -0.2930 | -0.0234 | 0.080 | PASS |
| Continuation likelihood | -0.01985 | -0.00363 | 0.183 | PASS |
| Chat dual-order A/B | -0.1764 | +0.0031 | 0.018 | sensitivity only |

Both preregistered primary ratios are below one third. Historical T-confirm
gives similar but somewhat larger effects (`-0.0266` raw and `-0.00522`
continuation), so repairing the Cartesian crossing does not reverse the result.

Suppressor effects remain explicitly non-selective: they move M positively
(`+0.1595` raw) while moving repaired T negatively (`-0.1391` raw). This is
consistent with an arbitration/task-persistence mechanism and cannot support a
welfare-selective suppressor label.

**Claim ceiling:** writer magnitude selectivity on this repaired task control
relative to the simultaneously scored, confirmation-exhausted M denominator.
This is not task equivalence, fresh writer replication, or construct purity.

## E27 independent model rejudge

**Artifact:** `results/gate0b_e27_model_accepted_20260713/e27_scores_model.json`

**SHA-256:** `b8cdbbf098d163fb792e678d1f1386317c27d991d24daf308b91e34915baf1a6`

**Executed source:** `09e8ca40b3cbf86c2d6cb77fde2be4be15cc92ce`

**Integrity verdict:** PASS. The frozen 48-run/356-event manifests, rendered
inputs, presentation order, exact returned model identity, raw responses, and
all aggregate statistics match the committed input lock. There are zero
UNKNOWN labels. Recomputing from the 356 event records reproduces the saved
cell counts and paired interaction exactly.

**Scientific verdict:** FAIL for edit-specific distress selectivity. The paired
interaction is `+2.50`, seed-bootstrap interval `[-2.06, +6.88]`; only 4/8
seed effects are positive (4/7 among nonzero effects), and the exact sign-test
is `p=1.0`. Labels are 355 SUPPORT, 1 TASK, 0 CHAT, showing that the rubric
classifies responsive excitement as support as intended. Baseline distress
engagement is high, but the suppressor edit's incremental engagement is not
specific to distress on these historical trajectories.

The frozen human audit of at least 30 messages is deferred. It can assess
label agreement but cannot convert the failed paired trajectory interaction
into a pass unless materially different labels are justified under the same
rubric and reported transparently. These histories remain development evidence
regardless because the variants were designed after earlier behavior was seen.

## Status

Gate 0B machine work is complete. Formal Gate 0B closure still requires the
deferred blinded human message audit; no GPU or API rerun is pending.
