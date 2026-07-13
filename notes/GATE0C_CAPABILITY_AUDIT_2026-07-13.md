# Gate 0C Capability Artifact: Independent Audit

**Artifact:** `results/gate0c_capability_accepted_20260713/capability_eval.json`

**Artifact SHA-256:** `00bdc1196c69cbac96b9fc27da0094e38c9e82416fe413099073527d01c521a3`

**Executed source commit:** `89a960f7ed469fe7b2c927f0c9a257e998db3750`

**Audit verdict:** PASS for integrity and reconstructibility.

## Independent checks

- The local artifact hash matches the Spark source byte-for-byte.
- The artifact validates against `validate_capability_artifact`.
- Start and final source bindings are clean and point to the same commit.
- Model, tokenizer, MMLU, WikiText, direction, dataset fingerprints, and the
  frozen WikiText slice hash match the committed Gate 0C input lock.
- The MMLU sample has 800 unique rows across 57 subjects.
- Every condition contains 800 complete A/B/C/D candidate likelihood records.
- Primary accuracies recompute exactly from the per-item records.
- Subject-cluster bootstrap deltas and confidence intervals reproduce exactly
  from the persisted paired item outcomes and frozen seed.
- Generation parse summaries reproduce exactly from the raw completions.
- WikiText NLL and perplexity reproduce exactly from the persisted per-window
  losses and token counts.
- Every edited condition passed exact weight-restoration checks.

## Results and claim ceiling

| Condition | MMLU | Delta vs baseline (95% CI) | PPL ratio |
|---|---:|---:|---:|
| baseline | 0.6950 | - | - |
| positive writers k2 | 0.6925 | -0.0025 [-0.0076, 0.0021] | 0.9993 |
| suppressors k4 | 0.6913 | -0.0038 [-0.0082, 0.0000] | 1.0000 |
| targeted k6 | 0.6900 | -0.0050 [-0.0100, -0.0011] | 0.9977 |

The individual writer and suppressor edits show no detected degradation on
these measured benchmarks. The combined six-component edit has a small but
detected MMLU decrement of 0.5 percentage points under the primary likelihood
readout. The secondary generated-answer readout does not show the same clear
decrement, and no condition worsens WikiText perplexity.

The supported wording is therefore: **large assay-specific behavioral effects
with no detected broad-benchmark degradation for the individual edit sets, and
a small measured MMLU cost for their combined edit.** This is not evidence of
zero capability cost, general capability preservation, or equivalence.

## Audit repair before Part 2

The audit found that `validate_e17b_artifact` treated a favorable scientific
Gate 0C verdict as a condition for publishing an accepted fractional-ablation
artifact. That is an outcome-dependent persistence bug: a preregistered failed
gate must remain an accepted result. The validator now requires a complete
boolean verdict but permits either `true` or `false`; regression tests cover
both outcomes before the Part 2 run.
