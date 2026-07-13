# WP2 Broadened Development Search Specification

**Status:** frozen after the four-block development null and before any
all-block activation extraction. This is a new, explicitly post-null discovery
experiment. It cannot reopen the original WP2 branch or authorize confirmation.

## Question

Does any candidate from the already specified WP2 linear/low-dimensional class
pass the development selectivity screen at a block omitted by the original
four-site search?

## Fixed search

- Model/revision: `google/gemma-2-9b-it` at
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.
- Data: the unchanged WP1/WP3 development partitions. Confirmation partitions
  remain unopened.
- Sites: residual-stream post-block activations at every block `0..41`.
- Token roles: `quote_boundary` and `prompt_final` exactly as implemented in
  the original WP2 lock.
- Candidate classes, dimensions, ridge grid, nuisance matrix, folds, seed,
  target/control arms, and lexicographic selection: unchanged from
  `WP2_IMPLEMENTATION_LOCK_2026-07-13.md`.
- Persist every candidate and fold score. Do not report only the best site.

## Why the class is not broadened further

There are sixteen target development families. Some inner-CV training folds
contain only about eight target-family contrasts, so supervised dimensions
above eight are rank-deficient by construction. Flexible nonlinear models on
this pool would add researcher degrees of freedom without enough independent
families to estimate them. Higher-dimensional or nonlinear searches require
fresh target/control development families and a separate preregistration; they
are not smuggled into this depth sweep.

Pretrained SAE feature analysis and DAS are separate WP4 experiments, not
candidate classes in this search. They require their own fidelity, null, and
held-out confirmation gates.

## Outcomes

- `candidate-found`: the frozen development screen passes at some site. This
  motivates fresh preregistration and fresh confirmation families only. No
  representation is frozen from this post-null search.
- `no-candidate`: no all-block candidate passes. This strengthens the
  conditional development null over the tested class but does not establish
  nonexistence.

Both outcomes remain conditional on human validation of the exact WP3 target
and the four failed WP1 controls. Target AUROC is described as decoding
prompt-stipulated current welfare status, never as evidence of empathy or an
internally generated welfare variable.
