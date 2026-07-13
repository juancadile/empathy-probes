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
| repaired T / T-confirm | task-control development / Gate-0 closure | score only after Integrity Repair A and Gate 0B freeze |
| M-confirm2 Gate-1 partition | final writer/component replication | sealed; never circuit/persona selection |
| M-circuit-confirm partition | Stage-B edge/feature graph confirmation | must be disjoint from Gate-1 M-confirm2; generate new families if not partitioned before scores |
| R2b fresh 16 families | suppressor component localization | sealed; old V2.2 families diagnostic only |
| WP3 confirmation families | welfare-content construct test | sealed; no direction/layer/prompt tuning |
| B6 policy-confirm families | concept/persona/policy dissociation | disjoint from M-confirm2 and circuit families |
| E22b confirmation families | moral-allocation transfer | old E22 is method history only |
| EIA confirmation maps/seeds/messages | prospective/enacted action test | disjoint from parser/dose/map development |

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

- Authority: `EXECUTION_ORDER_2026-07-13.md` Gate 1A and issue #37.
- Requirement: at least ten new sealed M-confirm2 families plus matched fresh task controls; frozen current writer set and dual-order raw/chat/continuation protocol.
- Claim ceiling on pass: band-localized costly-helping writer intervention, not a unique two-component circuit.

### Suppressor R2b

- Authority: `SCIENCE_AUDIT_2026-07-13.md` R2b section and issue #36.
- Requirement: sixteen fresh active-objective families; realized joint-L2 norm pairing; one head per L17-L20; dose monotonicity; family-paired target-minus-null inference.
- Claim ceiling on pass: selected-head localization conditional on the tested layer/type universe. Failure retains set-level/direction-specific evidence only.

## Gate 2: Construct identification

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
- Exit: intervention-consistent `N/O/C -> A` partial/full operational abstraction or not-identified/entangled result.

### Base versus instruct weights

- Preregistration: `C3_BASE_IT_PREREG_2026-07-13.md`
- Issue: #26.
- Static weight differences are descriptive; causal training-attribution requires exact lineage, localized delta transplant/reversal, dose response, and matched controls.

### Steering asymmetry

- Preregistration: `C4_STEERING_ASYMMETRY_PREREG_2026-07-13.md`
- Issues: #8 and #20.
- Compare additive steering, projection scaling/removal, norm-preserving rotation, and natural patching at geometry-matched doses. No entrenchment/workspace story without survival.

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
8. B8/C3/C4 and normalized generality work only on claims that survived earlier gates.
9. Paper/showcase rewrite after result-level engineer interpretation and independent adversarial review.
