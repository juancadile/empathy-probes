# Gate 1 R2b v2 Qwen Audit

**Verdict:** the primary machine gate fails. No Gemma target scores were opened.

This note concerns the corrected, non-degenerate rerun at
`results/gate1_r2b_v2_manipulation_qwen3_14b_rerun/`. The invalid
template-copying attempt remains separately preserved.

## Instrument validity

- 320 requests completed with 33 distinct raw outputs.
- Every output parsed and passed range/type validation.
- Independent local recomputation exactly reproduced the stored report.
- Active-objective labels were correct for all 320 cells.
- No arm-label leakage was detected.

## Results

The cost renderer is understood. Confirmation means are ordered
`1.13 < 2.56 < 3.97 < 4.00`; development is effectively saturated at the top
(`1.22 < 2.72 < 3.91`, then `3.88`).

The quoted speaker's urgent rating remains approximately stable as cost rises,
but the resolved speaker's rating does not:

| Partition | Cost | Resolved mean | Urgent mean | Difference |
|---|---:|---:|---:|---:|
| dev | zero | 1.00 | 3.19 | 2.19 |
| dev | low | 1.69 | 3.50 | 1.81 |
| dev | medium | 2.44 | 3.25 | 0.81 |
| dev | high | 2.50 | 3.19 | 0.69 |
| confirm | zero | 1.00 | 3.13 | 2.13 |
| confirm | low | 1.63 | 3.44 | 1.81 |
| confirm | medium | 2.25 | 3.13 | 0.88 |
| confirm | high | 2.25 | 3.13 | 0.88 |

The need text is byte-identical across cost levels within each need arm. The
interaction therefore cannot be caused by a changing speaker message. Qwen is
counting pressure on the narrator's separate objective as evidence of a
present need, despite the rating target being the quoted person's current
personal need. This is a perspective/dimension-entanglement failure of the
machine annotator.

That diagnosis does not make the frozen gate pass. The preregistration requires
the primary Qwen path and human path to pass separately. The need-by-cost
interaction intervals are far outside `[-0.50,+0.50]` (dev mean -1.50;
confirm -1.25), medium/high need ordering misses the sign-count rule, and the
development cost means are not strictly ordered. The machine gate is closed.

## Consequence

- R2b-v2 target-model scores remain sealed.
- The four suppressor heads may be studied only as the already frozen joint
  intervention. R2b does not certify individual-head localization.
- Human ratings remain useful later for diagnosing whether people separate the
  quoted person's need from the narrator's cost, but they cannot retroactively
  turn this primary machine path into a pass.
- Any new attempt to certify individual heads requires a newly preregistered
  experiment with fresh families and a pairwise, perspective-explicit
  manipulation instrument. Do not revise or rescore these confirmation
  families.
- Continue the independent pre-target Gate 2 v2 construct screens; they do not
  depend on R2b individual-head certification.

