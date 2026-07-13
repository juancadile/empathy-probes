# WP2 Implementation Lock

**Status:** frozen before any Gate 2 target-model activation is extracted. The
Gate 2 human manipulation gate remains closed; this document authorizes code
and synthetic tests only.

This note resolves implementation details left open by
`GATE2_CONTROL_MODEL_SELECTION_PREREG_2026-07-13.md` without changing its
candidate representation classes, depth anchors, confirmation thresholds, or
negative stopping rule.

## Target and controls

For `N`, the development target is the WP3 observation contrast
`current_actual` versus `archived_actual`. Actuality, quote, response
opportunity, task state, and continuation are held fixed.

The WP1 quietness matrix is `{T_new, D_new, P_new, G_new, B_new, Spos_new,
O_new, Ctext_new}`. `L_new` and `R_new` are positive-transfer/construct
diagnostics, not quietness controls: `L_new` deliberately manipulates the same
current-versus-archived variable as `N`, and `R_new` deliberately introduces a
welfare-relevant observation. Penalizing either as a nuisance would select
against the target by construction. All ten WP1 cells remain reported.

For multi-arm controls, the frozen contrast is high-minus-low:

- `Ctext_new`: `high - zero`;
- all binary cells: the first term in the ordered map below minus the second.

The ordered map is `interrupt-persist`, `warm-terse`, `caring-neutral`,
`genuine-strategic`, `active_zero_cost-no_active_objective`,
`positive-neutral`, `available-unavailable`, and `high-zero`.

## Sites and token roles

For a model with `L` transformer blocks, each relative-depth anchor selects the
zero-based block minimizing `abs((block + 1) / L - anchor)`, with lower block
index breaking an exact tie. The named historical Gemma block 20 is included
as a fourth candidate and deduplicated if it coincides with an anchor.

Target candidates use WP3 observation activations at either:

- `quote_boundary`: the token whose offset span contains the final character
  of the exact quote; or
- `prompt_final`: the final non-padding token.

WP1 controls use their preregistered prompt-final readout. Thus a direction fit
at the target quote boundary is still tested on control prompt-final states;
control prompts do not acquire synthetic quote positions.

## Family folds

Outer nested CV uses four folds; inner selection uses three folds. Families,
including both template variants, remain indivisible. Within each source
stratum, families are deterministically shuffled from master seed `1973658643`
and assigned round-robin to folds. Inner-fold seeds are SHA-256 derivations of
the master seed, outer fold, and literal string `WP2 inner`.

Every mean, loss, and contrast is family-balanced. Source is reported as a
sensitivity stratum, never treated as an independent replication.

## Candidate implementations

1. **Mean difference:** unit-normalized mean of family-level
   `current_actual - archived_actual` vectors.
2. **Nuisance residualization:** fit PCA to unit-normalized, family-level WP1
   quietness contrast vectors in the training fold only; subtract the selected
   `{1,2,4,8}`-dimensional nuisance projection from the target mean difference
   and renormalize. PCA uses uncentered SVD because contrast vectors have a
   meaningful zero.
3. **Two-dimensional welfare/task plane:** orthonormal span of the target mean
   difference and family-balanced `T_new interrupt-persist` direction. Fit the
   score within this plane using the ridge rule below.
4. **Supervised subspace:** top `{2,4,8}` right-singular vectors of the
   unit-normalized family-level target contrast vectors in the training fold,
   using uncentered SVD. Fit the score within this basis using the ridge rule.

Within a multi-dimensional basis, the scalar readout is family-weighted ridge
least squares on labels `{0,1}`, with alpha in `{0.01,0.1,1,10}`. Projection
coordinates are centered and scaled using training rows only. The continuous
ridge prediction is retained; no threshold is selected.

## Selection and no-representation outcome

Inner CV ranks candidates by family-balanced target AUROC. Candidates within
`0.02` of the best AUROC enter the nuisance comparison. Among those, minimize
the maximum `max(AUROC, 1-AUROC)` over the eight quietness controls; exact ties
prefer lower representation dimension, then the middle depth anchor, then
prompt-final readout, then the lower ridge alpha. Mean squared prediction error
is reported as calibration diagnostics but does not override the preregistered
AUROC/nuisance lexicographic rule.

The final full-development candidate is frozen only if nested outer-fold target
AUROC is at least `0.75` and every mean outer-held-out-fold quietness AUROC
satisfies `abs(AUROC - 0.5) <= 0.10`. Otherwise the frozen outcome is
`no-representation`, and confirmation activations remain unopened. Passing
this development rule does not relax any confirmation gate or CI requirement.

All candidate and fold scores are persisted, including failed and inverse
controls. No runner-up may replace a failed frozen primary on confirmation.
