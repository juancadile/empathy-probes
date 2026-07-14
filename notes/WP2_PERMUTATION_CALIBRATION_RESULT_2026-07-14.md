# WP2 Paired-Label Permutation Calibration Result

**Experiment:** `WP2-paired-permutation-calibration-20260713`  
**Raw permutations:** 32/32 complete  
**Confirmation opened:** no  
**Claim authorized:** no

## Primary result: observed site selection does not converge

The corrected pre-result primary statistic is the mean pairwise absolute block
distance among the four outer-fold winners. Lower values indicate tighter
depth convergence.

| summary | observed | paired-label null |
|---|---:|---:|
| mean pairwise block distance | 14.33 | median 12.08 |
| block span | 26 | median 24 |
| unique `(block, role)` sites | 4/4 | 4/4 in 28 of 32 runs |

The observed lower-tail plus-one permutation p-value is **0.545** for both
mean pairwise distance and block span. The selector therefore converges no
more tightly on the real, decodable target than on targets destroyed by paired
within-family label exchange.

This directly supports the all-block result's primary diagnosis: the outer
fold choices at blocks 28/15/2/7 are selection instability, not evidence for a
common representational site. It does not establish that the model contains no
welfare representation.

## Apparent-pass counts are descriptive only

The observed search has 62 configurations satisfying the conjunctive
target-plus-quiet threshold. Permuted runs have median 0, mean 2.56, and range
0--39; one of 32 permuted nested screens passes.

No p-value is computed or interpreted for this comparison. Target-label
exchange destroys target decodability, so a low null pass count is expected by
construction from the target-AUROC threshold. Calibrating threshold
multiplicity requires a separately locked, matched-decodability decoy target.
That experiment is outside the narrow V2 scope.

## Interpretation ceiling

The result says the tested selector's depth/site choice is not more stable
than under a null target. It cannot distinguish absence of a stable
representation from incoherent/trivial targets or confounded controls. Human
Form A remains the instrument for evaluating that construct-validity caveat;
it is an optional upgrade to the conditional negative result, not a blocker to
the V1 correction.

## Audit trail

The original specification incorrectly designated apparent-pass count as the
primary statistic. The flaw was identified and frozen in
`WP2_PERMUTATION_CALIBRATION_AMENDMENT_2026-07-13.md` while zero permutation
outputs existed. Raw runs remained unchanged; only the inferential priority
and canonical aggregator changed. All 32 raw files and the corrected aggregate
are tracked under `results/wp2_permutation_calibration_20260713/` and bound by
`MANIFEST.json`.
