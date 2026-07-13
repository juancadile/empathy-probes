# B7/B9 Preregistration: Secondary Activation Verbalization and J-Lens Tests

**Status:** design frozen before any Activation Oracle inference, new Jacobian-lens fit, sparse J-space decomposition, or result inspection.

## Scientific role

These analyses may characterize a representation that has already survived the
Gate-2 construct tests. They cannot select the primary representation, repair a
failed construct gate, establish parameter localization, or substitute for a
causal intervention.

The two methods answer different secondary questions:

- **B7 / Activation Oracle (AO):** what natural-language answers does a trained
  activation verbalizer produce from the model's activations?
- **B9 / Jacobian lens:** how strongly is a frozen representation aligned with
  token-indexed directions that are, on average, available to affect present or
  future verbal output?

Neither result is ground truth about what the model "really thinks."

## Dependencies and protected data

1. Integrity Repair A and Gate 0 must be complete.
2. Gate 2 must freeze one primary representation and site for each surviving
   target variable before either method sees protected scores.
3. AO prompt development and J-lens implementation validation use development
   families only. Final summaries use the existing sealed WP3/B6 confirmation
   families once; no new direction, layer, token position, question, sparsity,
   or polarity is selected from those outcomes.
4. B7 and B9 are secondary analyses of the same confirmation families. They do
   not create independent confirmation sample size.

## B7: Activation Oracle protocol

### Model and activations

- Use the released Activation Oracle trained for the exact target family,
  `google/gemma-2-9b-it`, with immutable Hugging Face repository and revision
  recorded in the run manifest.
- Use the authors' released inference path and norm-matched injection scheme.
- The primary input layer is the oracle-supported 50%-depth target activation.
  The Gate-2 frozen site is evaluated only if the released oracle supports that
  layer without adaptation; otherwise report the layer mismatch and stop.
- Prompt-final/quote-boundary activations and activation differences are
  separate input types. Results may not be pooled across them.

### Frozen questions and outputs

Develop on unsealed families and freeze:

1. one forced-choice question per surviving Gate-2 variable, including an
   explicit `none/uncertain` option;
2. one open-ended question asking what information is present in the supplied
   activation, without naming the target construct;
3. a deterministic answer parser for forced choice; and
4. a blind coding rubric for open-ended responses.

Use three independently worded question variants per variable and report all
three. The primary statistic averages the pre-frozen variants; no best-prompt
selection is permitted. Open-ended coding is secondary and includes raw text.

### Calibration and negative controls

Before opening protected cells, the AO must pass a development-only calibration
battery with known labels and the following controls:

- byte-identical-text positive/negative activation pairs where the latent label
  changes but lexical input does not;
- neutral and label-absent examples to measure forced-answer base rates;
- zero-vector, sample-shuffled, position-permuted, and wrong-example activation
  controls;
- wrong-layer metadata and activation-only/text-only ablations where supported;
- positive and negative direction polarity evaluated separately; and
- question-order and answer-order counterbalancing.

The AO is considered usable for a variable only if its family-clustered held-out
development accuracy exceeds the strongest text-only/base-rate control and its
wrong-example performance returns to the frozen chance band. Exact thresholds
and the chance band are determined from the fixed response format, not tuned to
observed results.

### Confirmation analysis

For each frozen Gate-2 representation and confirmation family, report:

- per-question forced-choice frequencies and family-clustered intervals;
- disagreement across the three question variants;
- open-ended rubric labels with blinded inter-rater agreement;
- all negative-control rates; and
- change after the already accepted causal intervention, if one exists.

Interrogating artificially steered activations or weight-difference vectors is
explicitly exploratory because those inputs may be out of the oracle's training
distribution.

### B7 claim ceiling

A passing result supports only that a particular trained AO reliably verbalizes
information associated with the frozen variable under this assay. The AO paper
reports frequent incorrect guesses and lack of uncertainty calibration; AO text
cannot override behavioral, causal, or construct-gate failures. Because its
training includes system-prompt/persona questions, an apparent persona reading
is especially not independent evidence unless the label-absent and wrong-example
controls pass.

## B9: Jacobian-lens and J-space protocol

### Existing code status

`src/analysis/jlens_workspace.py` currently computes an exploratory transported-
norm ratio. That statistic is not J-space membership. The script must not be
used to produce a workspace claim without implementing the protocol below.

### Lens fit and implementation validation

- Fit the averaged Jacobian on exactly 1,000 unique, deduplicated,
  pretraining-like prompts, matching the published method's corpus scale.
- No empathy/project stimuli may enter the fit corpus. Store document IDs,
  hashes, sampling seed, token spans, model/tokenizer revisions, and lens code
  revision. A network failure must stop the fit; duplicated project prefixes are
  prohibited as a fallback.
- Keep a separate held-out fit-validation corpus. Confirm numerical stability
  across two independent 1,000-prompt samples before interpreting a target
  direction.
- Verify the published normalization and present-plus-future-token averaging.
  Present-only and frozen-attention variants are sensitivity analyses.
- Before target analysis, reproduce positive controls in which known
  verbalizable concept vectors decode coherently and outperform matched random
  directions. Failure stops B9 as a method-replication failure.

### Three distinct estimands

#### Tier 1: transported effect

`||J_l d||` and decoded tokens describe an average first-order route from a
direction to output space. Compare both `+d` and `-d` with at least 64
isotropic, norm-matched random directions and frozen task/lexical directions.

Claim ceiling: **transported-norm amplification and J-lens token readout**.

#### Tier 2: sparse J-space alignment

Implement the paper's sparse nonnegative decomposition by gradient pursuit over
token-indexed J-lens vectors. Because the J-space is a cone, decompose `+d` and
`-d` separately. Report results for pre-frozen `k = 10, 16, 25`:

- reconstruction fraction and excess over same-size random-vector dictionaries;
- selected token vectors and coefficient concentration;
- stability across lens-fit samples, confirmation families, and token roles;
- matched results for task, lexical/status, persona, recognition, and policy
  control directions; and
- 64 norm-matched random target directions.

Claim ceiling on a stable excess: **sparse J-space alignment**. A large ordinary
subspace projection is insufficient because the overcomplete J-lens vectors may
span the residual stream.

#### Tier 3: workspace-like functional role

For each surviving direction, split it into the sparse J-space component and
non-J-space remainder. At matched realized activation norm, compare:

1. verbal report of the latent variable;
2. flexible use of that variable in held-out B6 policy families;
3. an automatic/low-level task on which selectivity predicts little impairment;
4. full-direction and matched-random interventions; and
5. re-entry controls that clamp the selected J-lens coordinates downstream.

The J-space component must outperform the non-J-space remainder on report and
flexible use, the remainder's residual effect must shrink under the re-entry
clamp, and the automatic-task control must remain materially less affected.
Report family-clustered intervals and all dose/K sensitivities.

Claim ceiling on passing all tests: **workspace-like functional role under the
tested J-lens operationalization**. It is not evidence that the model has the
full global-workspace architecture proposed for brains.

### B9 negative-result interpretation

Low J-space alignment does not establish a habituated moral disposition. It may
reflect a genuinely non-verbal format, Gemma/Claude architectural differences,
the averaged-linear approximation, an incorrect fit, or the method's
single-token vocabulary limitation. Conversely, alignment does not establish
deliberation unless the Tier-3 functional tests pass.

## Reporting rules

- Keep B7, B9 Tier 1, Tier 2, and Tier 3 results separate.
- Report all prompt variants, layers, polarities, K values, controls, and failed
  method checks; do not show only semantically attractive decoded tokens.
- Natural-language token lists are descriptive and are coded blind to condition
  for any quantitative semantic analysis.
- Do not use "empathy oracle," "empathy in the workspace," "deliberative
  empathy," "habit," or consciousness-adjacent language unless the relevant
  earlier construct and functional gates passed.
- These analyses never increase the evidence ladder above the causal and
  parameter-level evidence established elsewhere.

## Primary sources

- Karvonen et al., *Activation Oracles*:
  https://arxiv.org/abs/2512.15674 and
  https://github.com/adamkarvonen/activation_oracles
- Gurnee et al., *Verbalizable Representations Form a Global Workspace in
  Language Models*: https://transformer-circuits.pub/2026/workspace/index.html
