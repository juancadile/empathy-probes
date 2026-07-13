# Gate 0C Fractional-Ablation Artifact: Independent Audit

**Artifact:** `results/gate0c_fractional_accepted_20260713/e17b.json`

**Artifact SHA-256:** `f22b7ea8a23a62a556fabcd55953056c749a30ac567307c667228c8eefcdb560`

**Executed source commit:** `9bca5cdc787a362576e0a76e6cc5d9e1056d813b`

**Audit verdict:** PASS for integrity, reconstructibility, and the
preregistered raw-readout gate.

## Independent checks

- The local artifact and log hashes match their Spark sources byte-for-byte.
- The accepted artifact passes `validate_e17b_artifact` independently.
- Start and final source bindings are clean and point to the same commit.
- Direction, model, block, component registry, M-confirm, and repaired
  T-confirm hashes/family manifests match the frozen Gate 0C protocol.
- The 39 child seeds reproduce exactly from the frozen PCG64 master stream;
  all 39 random-direction records and hashes are present.
- Every target fraction and random control contains complete per-pair scores
  for 48 M-confirm and 40 T-confirm pairs under all three readouts.
- Re-running `analyze_fractional_protocol` from only the persisted scores and
  family labels reproduces the saved gate report exactly.
- The accepted-artifact validator permits both scientific pass and failure;
  acceptance is based on completeness/provenance rather than outcome.

## Results

| Readout | M target Z (family 95% CI) | Largest null Z | p | T Z | abs(T)/abs(M) |
|---|---:|---:|---:|---:|---:|
| raw dual-order | 3.192 [2.579, 3.805] | 0.061 | 0.025 | -0.174 | 0.055 |
| chat dual-order | 1.589 [1.219, 1.976] | 0.053 | 0.025 | -0.186 | 0.117 |
| continuation | 0.062 [0.032, 0.091] | 0.007 | 0.025 | -0.024 | 0.381 |

The raw target dose effects are monotonic:
`0.000, 0.594, 1.346, 2.243, 3.192`. Every raw family effect and every
leave-one-family-out mean is positive; the minimum LOFO effect is `2.980`.
The target exceeds all 39 random directions, giving the preregistered finite
minimum plus-one p-value `1/40 = 0.025`. Chat and continuation retain the
predicted sign and independently exceed their random-direction controls.

## Claim ceiling

This supports causal dependence of the historical M-confirm forced-choice
assay on the selected block-20 direction relative to isotropic directions.
The raw T/M effect ratio is small (`0.055`) but descriptive only; no equivalence
margin was preregistered. Continuation has weaker descriptive selectivity.

The result does not establish a welfare-pure representation, a complete
component/SAE circuit, mediation of enacted behavior, or fresh confirmation:
M-confirm is confirmation-exhausted and this is an audit-driven reconstruction.
