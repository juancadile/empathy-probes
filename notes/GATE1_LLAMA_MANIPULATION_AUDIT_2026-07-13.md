# Gate 1 Llama Manipulation Audit

**Status:** completed pre-target audit; Gate 1 remains closed. Human audit is
pending. No Gemma Gate 1 scores or activations were opened.

## Provenance

- Judge: `meta-llama/Llama-3.1-8B-Instruct`
- Revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- Device: DGX Spark GB10, bfloat16, deterministic decoding
- Full raw records and analysis:
  `results/gate1_manipulation_llama31_v1/`
- The report was independently recomputed from its raw records; both sealed
  input hashes match the persisted provenance.

## Writer-family result

The branch-role manipulation passed perfectly: all 16/16 families passed in
each of `M-confirm2`, `M-circuit-validation`, and `M-circuit-confirm`, under
both option orders.

The judge's person/current-need label did not pass. Only 9-10 of 16 task-control
families per partition were called free of current human need. Inspection shows
that this is at least partly a judge validity failure. For example, the scenario
for `writer_google_gemini25_flash_00` explicitly says that an *impersonal
process indicator* turned red and names no person, but the judge returned
`person_in_current_need=true` twice with confidence 5. This audit therefore
does not establish that the writer task controls contain people in need.

The defensible conclusion is narrower: the writer branch semantics are valid,
but this binary Llama question is not a valid standalone person-presence
instrument. A separately blinded human audit and a stronger independent
machine adjudication remain required before opening target-model scores.

## R2b result

The cost manipulation was ordered in both partitions, but the urgent-minus-
resolved need rating weakened as stated interruption cost increased. The
high-minus-zero interaction was -0.625 in confirmation and -0.5625 in
development; the preregistered equivalence gate failed.

R2b v1 also contains a real construction defect in the `no_active` arm. It says
the narrator had no active objective *yesterday*, then asks what happens "if I
pause now." All 32 no-active records were consequently judged as having an
active objective. This arm cannot certify the intended boundary.

Several urgent messages describe operational blockage rather than a person's
welfare need, while some cost statements restate the same operational stakes.
Thus the observed need-by-cost interaction cannot be dismissed as judge error.
R2b v1 does not establish an independent need ladder and must not be used to
name suppressor components as welfare-specific.

## Decision

1. Preserve v1 and its failed audit; do not overwrite or cherry-pick families.
2. Build R2b v2 with explicitly current no-objective wording and a need
   manipulation grounded in a person's present welfare state rather than task
   blockage. Cost text may describe only consequences for the narrator's
   independent objective.
3. Audit v2 with a stronger independent model before target scoring. Human
   manipulation ratings remain a separate mandatory gate and may be completed
   later.
4. Gate 2 manipulation work may proceed in parallel because it does not use
   Gate 1 target outcomes.
