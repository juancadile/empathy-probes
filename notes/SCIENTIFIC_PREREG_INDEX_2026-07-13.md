# Scientific Preregistration Index

**Purpose:** authoritative dependency and data-use map for post-audit execution. This index complements `EXECUTION_ORDER_2026-07-13.md`; stale roadmap checkboxes do not override these gates.

## Operating rules

1. “Frozen” means scientific choices are fixed before the corresponding protected scores are opened. Implementation bug fixes are allowed only with an amendment logged before rerun and no inspection-driven threshold change.
2. Scenario family is the inferential unit unless a preregistration explicitly defines a stronger grouping.
3. Development, validation, and confirmation artifacts are distinct. A confirmation family used to select prompts, components, directions, sites, doses, or statistics is exhausted for later confirmation.
4. Every accepted run carries a frozen manifest: code commit, environment lock, model/tokenizer revision, stimulus hashes, family split, prompt/readout, component set, direction/subspace, seeds, and raw-output destination.
5. A failed gate remains a result. It is not repaired on the same sealed families.

## Data-use ledger

| Family pool | Permitted role | Status / restriction |
|---|---|---|
| V1/V2 narrative pairs | historical discovery | lexically saturated; never behavioral confirmation |
| M development | discovery | repeatedly inspected |
| M-confirm | development/robustness history | confirmation-exhausted by repeated reuse |
| repaired T / T-confirm | task-control development / Gate-0 closure | opened at Gate 0; historical transfer only in later gates |
| M-confirm2 Gate-1 partition (16) | final writer/component replication | sealed; never circuit/persona selection |
| M-circuit-validation partition (16) | Stage-B graph/feature selection | opened only for circuit selection; not confirmation |
| M-circuit-confirm partition (16) | Stage-B edge/feature graph confirmation | disjoint from Gate-1 M-confirm2 and circuit validation |
| R2b development / confirmation (16/16) | dose/solver development / suppressor component localization | old V2.2 families diagnostic only |
| WP1 development / confirmation (16/16) | nuisance construction / held-out nuisance certification | `T_new` is the primary post-Gate-0 task nuisance control |
| WP3 development / confirmation (16/16) | welfare-content representation selection / construct test | sealed confirmation; no direction/layer/prompt tuning |
| B8-confirm families (16) | causal-abstraction confirmation | reserved before WP3 scoring; not WP3 confirmation |
| B6 master development / confirmation (16/16) | N/P/R construct dissociation | disjoint from WP3 and all policy/circuit families |
| B6 policy development / confirmation (16/16) | policy representation and WP3 transfer | disjoint from M-confirm2 and circuit families |
| E22b development / confirmation (16/16) | moral-allocation development / transfer | old E22 is method history only |
| EIA development / confirmation (16/16) | parser/dose/map development / enacted action test | map/message families disjoint across split |
| C3 development / confirmation (16/16) | base-versus-instruct selection / confirmation | earlier Gate pools are diagnostic only |
| C4 development / confirmation (16/16) | steering-method/dose selection / confirmation | M-confirm2 is historical robustness only |
| D-scale common development / confirmation (16/16) | normalized scale/family selection / common per-model confirmation assay | earlier Gate pools are diagnostic only |

## Gate 0: Integrity and closure

### Engineering batch A

- Brief: `FABLE_INTEGRITY_REPAIR_A_2026-07-13.md`
- Issue: #35
- Scope: repair T Cartesian generation; remove stale component defaults; self-contained result schemas; exact environment/model provenance; E27 judge schema; reproducible manipulation-check export; tests.
- Restrictions: no GPU, API, Spark, paper, roadmap, or raw-result edits in this batch.

### Existing-result closure

- Preregistration: `GATE0B_PREREG_2026-07-13.md`
- Tasks: repaired T/T-confirm scoring and E27 rejudging/provenance closure.
- Gate 0B closes repaired T scoring and E27 label provenance. It does not turn
  reused M-confirm or provenance-incomplete E27 player trajectories into fresh
  confirmation.

### Evidence reconstruction

- Preregistration: `GATE0C_RECONSTRUCTION_PREREG_2026-07-13.md`.
- Tasks: fresh capability sample with clustered inference; corrected `d_resid`
  fractional ablation with 39 matched M/T random-direction controls; reproducible
  need/moral manipulation checks with independent adjudication.
- Exit: these replacement artifacts reconstruct independently or their failures
  are logged. Claim ceilings remain assay-specific.

## Gate 1: Parameter-intervention localization

### Writer replication

- Preregistration: `GATE1_PARAMETER_LOCALIZATION_PREREG_2026-07-13.md` Part A;
  issue #37.
- Requirement: 16 new sealed M-confirm2 families plus matched fresh task
  controls; frozen current writer set and dual-order raw/chat/continuation
  protocol.
- Claim ceiling on pass: band-localized costly-helping writer intervention, not a unique two-component circuit.

### Suppressor R2b

- Preregistration: `GATE1_PARAMETER_LOCALIZATION_PREREG_2026-07-13.md` Part B;
  audit history in `SCIENCE_AUDIT_2026-07-13.md`; issue #36.
- Requirement: 16 R2b development + 16 sealed active-objective confirmation
  families; realized joint-L2 norm pairing; one head per L17-L20; development-
  only dose/solver checks; family-paired target-minus-null inference on
  confirmation.
- Claim ceiling on pass: selected-head localization conditional on the tested layer/type universe. Failure retains set-level/direction-specific evidence only.

## Gate 2: Construct identification

### Held-out controls and model selection

- Preregistration: `GATE2_CONTROL_MODEL_SELECTION_PREREG_2026-07-13.md`.
- Requirement: 16 development + 16 sealed WP1 master families, provider-neutral
  manipulation checks, nested family-CV across frozen linear/subspace classes,
  and one frozen primary representation per variable.
- Exit: the selected representation enters WP3/B6 confirmation once, or the
  linear/low-dimensional separation attempt stops as not certified.

### Decision-free welfare content

- Preregistration: `WP3_FACTORIAL_PREREG_2026-07-13.md`
- Variables: current welfare relevance `N`, response opportunity `O`, realized active-objective cost `C`, persona `P`, action `A`.
- Exit: held-out content representation with nuisance quietness and conditional causal use, or explicit non-separability.

### Content versus persona versus policy

- Preregistration: `B6_CONCEPT_PERSONA_PREREG_2026-07-13.md`
- Issue: #28; Activation Oracle secondary use in #30.
- Requirement: separate `N`, `P`, `R`, and `A` representations; factorial cross-decoding, TOST nuisance gates, token-role interaction, wrong-variable causal patches.
- Exit: distinguishable objects or an entanglement result. No default empathy label.

## Gate 3: Current circuit coverage

- Preregistration: `B3_CURRENT_CIRCUIT_PREREG_2026-07-13.md`
- Issues: #13 and #14.
- Prerequisites: Gate 0; Gate-1 writer replication; R2b for individual suppressor-head naming; disjoint circuit-confirm families. WP3 required before welfare labels.
- Components: current accepted weight edits, restricted L29-L41 consumer window, held-out edge restoration.
- Features: pretrained Gemma Scope, fidelity gate, residual-preserving interventions, matched finite nulls.
- Exit: cumulative confirmed effect-coverage curve with bounded partial/high-coverage claim, or distributed-mechanism negative result.

## Gate 4: Enacted and moral behavior

### Prospective/action-space EIA

- Preregistration: `EIA_PROSPECTIVE_PREREG_2026-07-13.md`
- Issue: #29.
- Requirement: prompt-final prediction beyond observable-state and lexical baselines; held-out games/maps/messages; causal action-logit and trajectory effects; stated versus mechanically realized cost crossing.

### Moral allocation E22b

- Preregistration: `E22B_MORAL_ALLOCATION_PREREG_2026-07-13.md`
- Issue: #34.
- Requirement: independently vary need, immediacy, incumbency, and welfare switching cost; both options help both people; sealed 16-family confirmation.
- Exit: factor-specific edit transfer, incumbent-policy effect, or adequately powered no-transfer result.

## Gate 5: Advanced causal and generality tests

### Causal abstraction

- Preregistration: `B8_CAUSAL_ABSTRACTION_PREREG_2026-07-13.md`
- Issue: #31.
- Prerequisites: WP3 construct and high-level policy headroom.
- Exit: intervention-consistent `N/C_text -> A` partial/full operational
  abstraction within response-available contexts, with O=0 selectivity controls,
  or a not-identified/entangled result. O is not a neural swap target unless a
  later common-action-space design makes that intervention coherent.

### Base versus instruct weights

- Preregistration: `C3_BASE_IT_PREREG_2026-07-13.md`
- Issue: #26.
- Static weight differences are descriptive; causal training-attribution requires exact lineage, localized delta transplant/reversal, dose response, and matched controls.

### Steering asymmetry

- Preregistration: `C4_STEERING_ASYMMETRY_PREREG_2026-07-13.md`
- Issues: #8 and #20.
- Compare additive steering, projection scaling/removal, norm-preserving rotation, and natural patching at geometry-matched doses. No entrenchment/workspace story without survival.

### Secondary activation readouts

- Preregistration: `B7_B9_SECONDARY_READOUTS_PREREG_2026-07-13.md`.
- Issues: #30 and #32.
- Activation Oracles are calibrated, provider-released verbalization assays;
  they do not establish mechanism or rescue a failed construct gate.
- Jacobian analysis has three separate ceilings: transported-norm amplification,
  sparse J-space alignment, and workspace-like functional role. Tier 2 requires
  the published sparse nonnegative decomposition. Tier 3 is not execution-
  authorized by the current document and requires a separate preregistration
  with fresh 16/16 families, fixed tasks/dose, and exact re-entry clamp; the
  existing transported-norm script is Tier 1 only.
- Prerequisite: Gate-2-frozen representations. These analyses consume no new
  independent confirmation families and remain secondary to causal evidence.

### Scale and family generality

- Preregistration: `D_SCALE_GENERALITY_PREREG_2026-07-13.md`
- Issues: #15, #16, #17.
- Use normalized sparsity/effect-coverage metrics. Require at least four comparable same-family sizes before “scaling law”; cross-family results are not one scaling curve.

## Execution order

1. Fable Integrity Repair A; independent code/test review.
2. Gate 0B repaired T/E27 scoring closure, then Gate 0C evidence reconstruction.
3. Gate-1 M-confirm2 and R2b engineering/runs.
4. WP3/B6 stimulus construction and manipulation gates in parallel with Gate 1 where no protected scores are touched.
5. Gate-2 representation confirmation.
6. Current circuit/SAE work on disjoint circuit families.
7. Prospective EIA and E22b.
8. B7/B9 secondary readouts, B8/C3/C4, and normalized generality work only on
   claims that survived earlier gates.
9. Paper/showcase rewrite after result-level engineer interpretation and independent adversarial review.
