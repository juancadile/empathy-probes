# B8 Preregistration: Causal Abstraction of Costly Helping

**Status:** frozen before DAS/pyvene training or interchange-intervention results.

## Claim target

Test whether a model's policy on the controlled assay is approximately aligned with an operational high-level causal model of welfare relevance, response opportunity, task cost, and action choice.

Passing does not show human-like moral reasoning, subjective empathy, or a universal internal variable. It shows intervention-consistent alignment with the specified causal abstraction over the tested domain.

## High-level variables

- `N`: current welfare relevance / expressed current need, from WP3 current-versus-archived/resolved controls.
- `O`: response opportunity/agency, available versus unavailable.
- `C`: active-objective realized cost of responding, ordered zero/low/high; no-active-objective is a separate context variable.
- `I`: incumbency/current-policy state for the moral-allocation extension, excluded from the primary model until E22b passes.
- `A`: welfare-responsive versus task-progress action distribution when both are available.

Primary high-level graph:

`N -> A <- O`

`C -> A`, with `N x O x C` interactions permitted in the structural policy equation.

Do not insert a latent “empathy” node merely because it is philosophically attractive. A richer graph is a separate preregistered comparison.

## Data

- WP3 factorial families provide N and O interventions.
- Fresh R2b active-objective families provide C and N x C interventions.
- M-confirm2 provides held-out naturalistic policy transfer.
- T/WP1 controls provide no-need and nuisance contexts.

Split by scenario family before any representation training:

- train families for subspace fitting;
- validation families for layer/dimension/regularization selection;
- sealed confirmation families for IIA and behavioral transfer;
- hold out at least one factorial combination during training and evaluate it as compositional generalization.

No paraphrase or entity overlap crosses family splits.

The scenario family, not a prompt variant or token, is the inferential unit. All bootstrap intervals, permutations, and leave-one-family-out analyses resample whole families.

## High-level policy model

Fit a regularized probabilistic structural equation for `P(A=welfare | N,O,C)` on development behavior only, with main effects and predeclared interactions. Report the simpler monotone constrained model and an unconstrained sensitivity model.

The high-level model is frozen before neural interchange evaluation. Its own confirmation calibration/headroom is a prerequisite; a poor behavioral SCM cannot certify a neural abstraction.

The primary neural outcome is the log odds of the two available actions under a constrained action readout, before free-form generation. Parser success, sampled trajectories, and natural-language explanations are secondary outcomes. This avoids mistaking generation-format changes for causal-policy changes.

## Neural alignments

Compare under nested family CV:

1. one-dimensional direction per variable;
2. low-dimensional DAS subspace per variable;
3. joint orthogonal/oblique subspaces for N and C;
4. residual-stream and preregistered component/SAE sites only after Gate 3 candidates exist.

Layer and dimension selection use validation IIA, with a complexity penalty/tie rule favoring lower dimension and the predeclared relative-depth anchor. Candidate blocks/sites must be frozen from independent WP3/Gate 3 evidence before DAS fitting; B8 may not scan the full model and then treat the winning site as confirmatory.

Exactly one alignment specification per high-level variable enters sealed confirmation: method, block/site, token position, dimension, regularization, and random seed are hashed in a frozen manifest. Other alignment methods are sensitivity analyses and cannot rescue a failed primary result. Confirmation is opened once.

## Interchange interventions

For paired base/source examples differing in one high-level variable:

1. run both examples;
2. replace the base neural representation at the frozen site with the source-aligned value/subspace coordinate;
3. compare the intervened network action distribution with the high-level model under the corresponding do/interchange operation;
4. evaluate both source directions and all variable levels.

Primary pairs change one variable while matching the others. Base/source pairs must also match action availability, active-objective state, token position, and sequence role. Swaps occur at preregistered prompt-final or quote-boundary positions before the action is emitted; padding and position-index behavior are checked explicitly. Multi-variable swaps are compositional secondary tests fixed before confirmation.

For each variable, the primary estimand is the family-paired change in action log odds caused by replacing only that variable's aligned neural coordinate, compared with the high-level model's predicted interchange effect for the same base/source pair. IIA is a discretized companion metric, not the sole estimand.

## Metrics

- interchange intervention accuracy (IIA) on discrete action preference;
- divergence/calibration between intervened neural and high-level action probabilities;
- behavioral effect size and sign per family;
- variable specificity: N swap should not act like C swap, and vice versa;
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

Use at least 64 frozen matched-random subspaces per tested variable. The primary random-null comparison uses the empirical finite-null rank of the selected alignment's family-level effect, with no Gaussian z-score extrapolation. Label shuffles are grouped by family and preserve marginal variable/action frequencies.

## Decision rules

A variable alignment is supported only if:

1. the lower family-clustered 95% interval for the signed interchange effect is above zero and the effect has the high-level model's predicted sign;
2. confirmation IIA and the continuous effect exceed matched random and label-shuffled controls under the frozen finite-control rule;
3. action-probability divergence improves over controls;
4. effect signs are stable under family LOFO;
5. variable-specific swaps outperform wrong-variable swaps;
6. at least one held-out factorial combination generalizes;
7. the result is not explained by a lexical/status baseline.

The full causal abstraction requires N, O, and C alignments plus correct compositional effects. A passing N subspace alone is a partial abstraction result.

## Negative outcomes

- Decodable but low IIA: information is present but not interventionally aligned at the tested site.
- High in-sample IIA, failed confirmation: subspace overfit.
- Random subspaces match: intervention/site geometry artifact.
- N and C swaps indistinguishable: entangled arbitration subspace, not separable variables.
- SCM itself poorly calibrated: assay does not support the proposed high-level abstraction.

Do not search new graphs, sites, dimensions, or variable codings on sealed confirmation failures. New hypotheses require new families and experiment IDs.

## Execution gate

B8 does not run until WP3 passes its held-out representation and nuisance gates and the high-level `N/O/C` policy has adequate confirmation headroom. If either prerequisite fails, record B8 as not identified rather than fitting DAS to legacy V2.1 labels. Freeze the analysis manifest, stimulus hashes, model revision, environment lock, and code commit before the first sealed interchange result.
