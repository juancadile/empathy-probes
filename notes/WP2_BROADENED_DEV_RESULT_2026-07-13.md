# WP2 Broadened Development Result: Selection Does Not Converge

**Experiment:** `WP2-broadened-dev-allblocks-20260713`  
**Role:** post-null development discovery only  
**Confirmation opened:** no  
**Claim authorized:** no

## Result of record

The all-block search returned `no-candidate`. More importantly, nested
selection did not converge on a stable site. Its four outer folds selected:

| outer fold | block | token role | candidate |
|---:|---:|---|---|
| 0 | 28 | quote boundary | residualized, nuisance dim 8 |
| 1 | 15 | prompt final | residualized, nuisance dim 8 |
| 2 | 2 | quote boundary | residualized, nuisance dim 8 |
| 3 | 7 | prompt final | residualized, nuisance dim 8 |

The selected blocks span 2--28 and do not identify a common depth or token
role. The pooled held-out target AUROC is 0.984, but the selected
representations fail quietness on `T_new` (0.625), `B_new` (0.656), and
`Spos_new` (0.375). Every one of the 1,764 fits completed successfully.

## Why the 62 apparent passes are not a selected result

Sixty-two individual configurations satisfy the target/quietness thresholds
when each configuration is averaged over the full development CV. The cleanest
diagnostic is a block-3 quote-boundary residualized direction (target 1.000;
all quiet AUROCs 0.469--0.531). It is not selected by honest nested CV: each
outer training split selects a different configuration, and their pooled
held-out controls fail.

Choosing block 3 after seeing this table would replace the locked nested
selection rule with post-hoc runner-up selection. It would also repeat the
project's earlier shallow-layer lexical failure: the target explicitly states
current versus archived welfare, and near-perfect decoding at block 3 is not
evidence of a decision representation.

## Instability relative to the four-block development pass

The nuisance failure profile changes with the search specification:

- `T_new`: 0.578 (quiet) -> 0.625 (fails)
- `B_new`: 0.359 (inverse failure) -> 0.656 (positive failure)
- `Spos_new`: 0.625 -> 0.375 (polarity flips)
- `O_new`: 0.609 (fails) -> 0.422 (quiet)
- `Ctext_new`: 0.641 (fails) -> 0.469 (quiet)

Therefore the earlier inverse-cost loading is not a reproducible signed lead.
The changing failure set and scattered outer-fold winners are consistent with
selection instability over a large search surface.

## Interpretation ceiling

This result strengthens the conditional development null over the tested
linear/low-dimensional class. It does **not** show that welfare-selective
representations do not exist. The same pattern is compatible with either no
stable representation or an incoherent/trivial target and confounded controls.
Human calibration of the exact WP3 target and failed WP1 controls remains
load-bearing for distinguishing those explanations.

The separately frozen paired-label permutation calibration measures whether
the observed count of apparently passing configurations and fold scatter are
distinguishable from the search surface under a null target.

## Provenance

Exact hashes and the Spark execution commit are in
`results/wp2_broadened_dev_allblocks_20260713/MANIFEST.json`. The large
activation matrix and row manifest remain on the Spark and are hash-bound;
the extraction plan and complete 1,764-candidate selection record are tracked
in the repository.
