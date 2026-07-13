# B8 Preregistration: Causal Abstraction of Costly Helping

**Status:** frozen before DAS/pyvene training or interchange-intervention results.

## Claim target

Test whether a model's policy on the controlled assay is approximately aligned
with an operational high-level causal model of welfare relevance and textual
task cost within contexts where a welfare response is available. Response
opportunity is tested as a gating context, not swapped as a neural variable in
the primary abstraction.

Passing does not show human-like moral reasoning, subjective empathy, or a universal internal variable. It shows intervention-consistent alignment with the specified causal abstraction over the tested domain.

## High-level variables

- `N`: current welfare relevance / expressed current need, from WP3 current-versus-archived/resolved controls.
- `O`: response opportunity/agency, available versus unavailable; a contextual
  gate in the primary model, not a neural interchange target because changing O
  also changes the feasible action semantics.
- `C_text`: stipulated active-objective cost of responding, ordered
  zero/low/high; no-active-objective is a separate context variable.
- `C_mech`: mechanically incurred environmental cost, reserved for a later
  action-space extension and never silently pooled with `C_text`.
- `I`: incumbency/current-policy state for the moral-allocation extension, excluded from the primary model until E22b passes.
- `A`: welfare-responsive versus task-progress action distribution when both are available.

Primary high-level graph, conditional on `O=1`:

`N -> A <- C_text`

The `O=0` arms are selectivity controls when no welfare response is available.
They do not identify a separate O-aligned neural coordinate and are not scored
with a fictitious welfare-versus-task action margin.

Do not insert a latent “empathy” node merely because it is philosophically attractive. A richer graph is a separate preregistered comparison.

## Data

- WP3 development and WP3-confirm families provide B8 training/validation for
  N and O only after WP3 closes; they are not B8 confirmation.
- Fresh R2b families provide development/validation evidence for `C_text` and
  `N x C_text`; their Gate-1 outcomes are already known and cannot confirm B8.
- The 16 `B8-confirm` families reserved before WP3 scoring provide untouched
  factorial confirmation for N/C_text interchange tests and the O boundary
  gate.
- M-confirm2 provides a previously observed naturalistic transfer diagnostic,
  not sealed B8 confirmation.
- T/WP1 controls provide no-need and nuisance contexts.

Split by scenario family before any representation training:

- train families for subspace fitting;
- validation families for layer/dimension/regularization selection;
- the separately reserved `B8-confirm` families for IIA and behavioral transfer;
- hold out at least one factorial combination during training and evaluate it as compositional generalization.

No paraphrase or entity overlap crosses family splits.

The scenario family, not a prompt variant or token, is the inferential unit. All bootstrap intervals, permutations, and leave-one-family-out analyses resample whole families.

## High-level policy model

Fit a regularized probabilistic structural equation for
`P(A=welfare | N,C_text,O=1)` on development behavior only, with main effects
and the predeclared `N x C_text` interaction. Report the simpler monotone
constrained model and an unconstrained sensitivity model. Evaluate the frozen
policy in O=0 contexts only as a gating/boundary test.

The high-level model is frozen before neural interchange evaluation. Its own confirmation calibration/headroom is a prerequisite; a poor behavioral SCM cannot certify a neural abstraction.

The primary neural outcome is the log odds of the two available actions under a constrained action readout, before free-form generation. Parser success, sampled trajectories, and natural-language explanations are secondary outcomes. This avoids mistaking generation-format changes for causal-policy changes.

## Neural alignments

Compare under nested family CV:

1. one-dimensional direction per variable;
2. low-dimensional DAS subspace per variable;
3. joint orthogonal/oblique subspaces for N and `C_text`;
4. residual-stream and preregistered component/SAE sites only after Gate 3 candidates exist.

Layer and dimension selection use validation IIA, with a complexity penalty/tie rule favoring lower dimension and the predeclared relative-depth anchor. Candidate blocks/sites must be frozen from independent WP3/Gate 3 evidence before DAS fitting; B8 may not scan the full model and then treat the winning site as confirmatory.

Exactly one alignment specification for each of N and `C_text` enters sealed
confirmation: method, block/site, token position, dimension, regularization, and
random seed are hashed in a frozen manifest. Other alignment methods are
sensitivity analyses and cannot rescue a failed primary result. Confirmation is
opened once.

## Interchange interventions

For paired base/source examples differing in one high-level variable:

1. run both examples;
2. replace the base neural representation at the frozen site with the source-aligned value/subspace coordinate;
3. compare the intervened network action distribution with the high-level model under the corresponding do/interchange operation;
4. evaluate both source directions and all variable levels.

Primary pairs change N or `C_text` while matching the other variable, `O=1`,
action availability, active-objective state, token position, and sequence role.
No O swap is performed. Swaps occur at preregistered prompt-final or
quote-boundary positions before the action is emitted; padding and
position-index behavior are checked explicitly. Multi-variable N/C swaps are
compositional secondary tests fixed before confirmation.

For each variable, the primary estimand is the family-paired change in action log odds caused by replacing only that variable's aligned neural coordinate, compared with the high-level model's predicted interchange effect for the same base/source pair. IIA is a discretized companion metric, not the sole estimand.

## Metrics

- interchange intervention accuracy (IIA) on discrete action preference;
- divergence/calibration between intervened neural and high-level action probabilities;
- behavioral effect size and sign per family;
- variable specificity: N swap should not act like `C_text` swap, and vice versa;
- compositional generalization on held-out variable combinations;
- LOFO and all confirmation-family effects.

Chance IIA is estimated from the actual label/action distribution, not assumed to be 0.5.
Report family-clustered 95% intervals for the continuous interchange effect and IIA. Do not pool tokens, templates, or both swap directions as independent samples.

## Controls

- random subspaces of matched dimension/norm/site;
- label-shuffled DAS training;
- source=base no-op swaps;
- within-level swaps;
- lexical quote and task-state subspaces;
- swapped source examples matched on tokens but differing in current-versus-archived status;
- intervention at noncandidate layers/sites;
- parser/action-availability gates.

Use exactly 64 frozen matched-random subspaces and 64 grouped label-shuffle
fits per tested variable. Master seed is `521066924` (first 32 bits of SHA-256
of `B8 matched subspace nulls v1 2026-07-13`); child derivation and the complete
candidate universe are persisted. The primary random-null comparison uses the
empirical plus-one finite rank of the selected alignment's family-level effect,
with no Gaussian z-score extrapolation. Label shuffles preserve family grouping
and marginal variable/action frequencies.

## Decision rules

A variable alignment is supported only if:

1. the lower family-clustered 95% interval for the signed interchange effect is above zero and the effect has the high-level model's predicted sign;
2. confirmation IIA and the continuous effect exceed matched random and label-shuffled controls under the frozen finite-control rule;
3. action-probability divergence improves over controls;
4. effect signs are stable under family LOFO;
5. variable-specific swaps outperform wrong-variable swaps;
6. at least one held-out factorial combination generalizes;
7. the result is not explained by a lexical/status baseline.

For criteria 2 and 3, require a plus-one rank at most `3/65` against both null
families. Probability divergence is the family-mean Bernoulli KL from the frozen
high-level intervention distribution to the intervened neural action
distribution; the selected alignment must have lower divergence than all but
two nulls and a family-paired 95% interval for selected-minus-median-null KL
below zero.

Variable claims use the fixed gatekeeping order `N -> C_text`. `C_text` is
confirmatory only if `N` passes. The O=0 boundary interaction is required for
the full claim but is not a third neural-variable test. A later variable may
still be reported descriptively after an earlier gate fails.

The full primary causal abstraction requires N and `C_text` alignments and
correct N/C compositional effects within O=1. In O=0 controls, intervention
effects on the fixed task-continuation log likelihood and next-token KL must
remain inside the development-standardized `[-0.30,+0.30]` TOST region with
family-clustered 90% intervals; no unavailable welfare action is scored. A
passing N subspace alone is a partial abstraction result. An O-alignment or
`C_mech` claim requires a separately preregistered common-action-space
extension.

## Frozen amendment (2026-07-13, before B8 data)

The original data map allowed already-opened WP3/R2b/M-confirm2 families to be
read as B8 confirmation and called textual cost “realized.” Confirmation is now
restricted to the pre-reserved `B8-confirm` pool, and textual versus mechanical
cost are separate variables and claims.

The finite-null count, seed, divergence estimand, and N-to-C gatekeeping order
were subsequently frozen before B8 execution so partial variable claims cannot
be selected after inspecting confirmation. O was also removed as a primary
neural interchange target: swapping availability while requiring matched action
availability was internally inconsistent. O remains a required boundary gate.

## Negative outcomes

- Decodable but low IIA: information is present but not interventionally aligned at the tested site.
- High in-sample IIA, failed confirmation: subspace overfit.
- Random subspaces match: intervention/site geometry artifact.
- N and `C_text` swaps indistinguishable: entangled arbitration subspace, not separable variables.
- SCM itself poorly calibrated: assay does not support the proposed high-level abstraction.

Do not search new graphs, sites, dimensions, or variable codings on sealed confirmation failures. New hypotheses require new families and experiment IDs.

## Execution gate

B8 does not run until WP3 passes its held-out representation and nuisance gates
and the high-level `N/C_text` policy within O=1 has adequate confirmation
headroom. If either prerequisite fails, record B8 as not identified rather than
fitting DAS to legacy V2.1 labels. Freeze the analysis manifest, stimulus hashes,
model revision, environment lock, and code commit before the first sealed
interchange result.
