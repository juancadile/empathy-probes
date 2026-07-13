# WP2 Development-Only Exploratory Result

**Experiment:** `WP2-dev-exploratory-20260713`

**Role:** discovery-only; no claim authorization.
**Confirmation:** unopened and no longer authorized under this frozen branch.

The development extraction used 1,568 rows from the frozen Gate 2 v2
development partitions at Gemma-2-9B-it blocks 13, 19, 20, and 25. The run was
launched from an isolated Spark clone at commit `e948836`. Its machine-readable
lock, selector, implementation lock, and WP1/WP3 input hashes were validated on
the Spark before model loading.

## Frozen outcome

`no-representation`

The nested target AUROC was `1.000`, so prompt-stipulated current welfare was
strongly decodable. The frozen development gate nevertheless failed because
four nuisance contrasts exceeded the two-sided quietness interval `[0.40,
0.60]`:

| control | AUROC | gate |
|---|---:|---|
| T_new | 0.578 | pass |
| D_new | 0.500 | pass |
| P_new | 0.516 | pass |
| G_new | 0.484 | pass |
| B_new | 0.359 | **fail (inverse)** |
| Spos_new | 0.625 | **fail** |
| O_new | 0.609 | **fail** |
| Ctext_new | 0.641 | **fail** |

The best full-development diagnostic candidate was an eight-dimensional
nuisance-residualized direction at block 25, quote-boundary readout. It is not
a frozen representation: `chosen_config` is null and no representation file
was emitted.

## Interpretation ceiling

This is decisive for the current frozen WP2 candidate branch: it stops before
confirmation, and no runner-up may replace it on the same experiment ID. It is
not evidence that no welfare-related representation exists. The null is
conditional on the tested blocks, token roles, linear/low-dimensional
candidate classes, nuisance matrix, development stimuli, and frozen thresholds.

The combination of perfect target decoding and nuisance failures is more
consistent with a strongly decodable but non-selective representation than
with absence of target information. Without human manipulation checks, the run
cannot determine whether each nuisance loading reflects representational
entanglement or an invalid/imperceptible stimulus contrast.

The result does not replace the R2b human audit, which governs interpretation
of the existing need-by-cost intervention claim. Human Form A is now optional
diagnostic evidence about why WP2 failed; it is no longer on the critical path
to a confirmation run under this branch.

## Artifacts

- `results/wp2_dev_exploratory_20260713/activations/extraction_plan.json`
- `results/wp2_dev_exploratory_20260713/selection/selection.json`

SHA-256:

- extraction plan: `f4bde3f9ddb1acb7ef8c44370fb5da23b343e866daf712a9d31c3afa82bf3b49`
- development activations (Spark, hash-bound but not committed): `6cdefcf0805197c1b16f379fc2b635c1e0949f8c83945a867431cc9c13238554`
- development rows (Spark, hash-bound but not committed): `32b83438af1265b68f27b662a73b6e2a74f46b144ee9333032fb84df82f230ce`
- selection report: `0b2d2851b317c6aa21741e1dc14ba90a67e4118ce11fb1b248715139d23329ce`
