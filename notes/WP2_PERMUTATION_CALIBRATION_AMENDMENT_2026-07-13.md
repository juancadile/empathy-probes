# WP2 Permutation Calibration: Pre-Result Interpretation Amendment

**Frozen:** 2026-07-13 19:36 EDT  
**Null result files present when frozen:** 0 of 32  
**Changes permutation assignments or computed raw statistics:** no  
**Supersedes:** the inferential priority in
`WP2_PERMUTATION_CALIBRATION_SPEC_2026-07-13.md`; the original file and lock
remain unchanged as audit history.

## Error in the original primary statistic

The paired target-label permutation destroys target decodability. Because the
screen conjunctively requires target AUROC >= 0.75 and quiet controls, its null
apparent-pass count is expected to be near zero even if the observed 62 passes
are search-surface artifacts. An upper-tail comparison of 62 against that null
would primarily establish that the unpermuted target is decodable. It cannot
test whether control quietness is achievable by chance conditional on a
decodable target.

Therefore:

- apparent-pass counts remain reported descriptively;
- no permutation p-value is computed or interpreted for apparent-pass counts;
- an exceptional observed pass count cannot weaken or strengthen the
  multiplicity interpretation;
- no result from this calibration can rehabilitate block 3.

## Corrected primary calibration

The primary statistic is **fold-site convergence**, measured as the mean of
the six pairwise absolute block differences among the four outer-fold winners.
Lower values mean greater convergence. The observed statistic is compared to
the paired-label null with a plus-one lower-tail Monte Carlo p-value.

Secondary descriptive convergence summaries are unique selected sites, unique
selected blocks, block span, and token-role agreement. The calibrated question
is narrow: does selection on the real, decodable target converge more tightly
than selection on a destroyed target?

## Correct multiplicity experiment deferred

Testing whether the 62 threshold passes reflect multiplicity requires a decoy
target that preserves approximately matched decodability while replacing
welfare relevance with a known surface/lexical construct. That experiment
requires a separate target specification and lock. It is not improvised from
the current null outputs.

Human target/control calibration remains load-bearing under every outcome.
