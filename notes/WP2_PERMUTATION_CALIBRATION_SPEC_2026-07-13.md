# WP2 Paired-Label Permutation Calibration

**Status:** frozen after the observed all-block result and before any
permutation result. This is a calibration of the development search surface,
not a new representation search.

## Question

Are the observed 62 full-development threshold passes and the scatter of the
four nested-CV winners unusual under a null target with the same family,
source, arm, token-role, fold, activation, and candidate-grid structure?

## Fixed permutation

- Run 32 permutations, indexed `0..31`, using master seed `2459917311`.
- For each index, independently exchange `current_actual` and
  `archived_actual` labels within each of the sixteen WP3 observation
  development families. A family's exchange applies to every variant and both
  token roles. Reject and deterministically redraw the identity and global
  inversion assignments.
- Do not permute activations, rows between families, source labels, folds,
  controls, diagnostic cells, or any confirmation data.
- Re-run the identical 1,764-candidate full-development CV and identical
  four-fold nested selection from the locked all-block experiment.
- Preserve one compact result per permutation. Do not retain or inspect a
  permutation's full candidate table while deciding how many permutations to
  run.

Thirty-two permutations give a minimum attainable plus-one Monte Carlo
p-value of `1/33 = 0.0303`. This is sufficient for the stated calibration and
was fixed before the first null run.

## Fixed summaries

For each permutation report:

1. number of individually valid full-development configurations satisfying
   target AUROC >= 0.75 and every two-sided quiet AUROC within 0.10 of chance;
2. nested target and quiet AUROCs and whether the nested screen passes;
3. the four selected `(block, token_role)` sites, their unique-site count, and
   block span;
4. the best full-development diagnostic configuration under the unchanged
   selector.

Compare the observed result to the permutation distribution using plus-one
Monte Carlo tail probabilities. The primary calibration is the upper tail for
the observed apparent-pass count. Fold scatter is descriptive: fewer unique
sites and a smaller block span indicate greater convergence.

## Interpretation ceiling

- An observed pass count and fold scatter typical of permutations supports the
  interpretation that the 62 apparent passes are compatible with search-surface
  multiplicity and unstable selection.
- An exceptional observed count or markedly greater fold convergence weakens
  that interpretation and must be reported without changing the selector.
- Neither outcome proves existence or nonexistence of a welfare representation,
  validates the target/controls, authorizes confirmation, or permits selecting
  block 3. Human calibration remains load-bearing.

All 32 outputs are retained, including failures. No stopping based on interim
results is permitted.
