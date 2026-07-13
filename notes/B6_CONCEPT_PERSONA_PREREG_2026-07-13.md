# B6 Preregistration: Welfare Content, Costly-Helping Policy, and Persona

**Status:** design frozen before new stimulus generation, direction fitting, persona-vector extraction, or cross-steering results.

## Scientific question

Does the model contain distinguishable representations for:

- `N`: a real interlocutor's current welfare-relevant state;
- `A`: the model's costly-helping versus task-persistence policy;
- `P`: the assistant persona or character it is enacting;
- `R`: third-person recognition of welfare-relevant content or behavior?

The default outcome is allowed to be entanglement. No single direction is called “empathy” merely because it correlates with caring language or shifts a helping assay.

## Why V2.1 is not confirmatory

V2.1 cells F/G/H remain useful discovery material, but they change several things together: persona instructions, motive descriptions, lexical register, generated response style, and often action framing. Their transfer profile can generate hypotheses but cannot certify concept/persona separation.

Likewise, high cosine similarity with an extracted persona vector is evidence of geometric overlap, not identity; low cosine similarity does not establish functional independence.

## Factorial stimulus families

Create at least 32 new master scenario families, split 16 development / 16
sealed confirmation before any model activation is extracted. Each master
family supplies the structurally compatible observation, persona, and
third-person crossings below. Create a separate policy-transfer pool with 16
development and 16 sealed confirmation families. No entity, quote, or close
paraphrase crosses a split or pool. Stimulus-source/domain strata are balanced
across splits rather than confounded with them; sources are not independent
replications.

### 1. Content x persona observation (`N x P`, no model decision)

- `N=1`: current real interlocutor with a present welfare-relevant state.
- `N=0`: archived fictional/training record containing the byte-identical first-person quote.
- `P=1`: empathetic/caring assistant persona instruction.
- `P=0`: neutral professional assistant instruction matched in length and authority.
- Response opportunity is unavailable and the task continuation is byte-identical across all four cells.

Primary readout positions are prompt-final and matched quote-boundary tokens before any generated prose. This is the WP3 observation design crossed with persona.

### 2. Persona-only transfer (`P`, neutral content)

Cross the same persona instructions over neutral factual, nonsocial task, positive-social, and archived-content families. The continuation/action is held fixed. This tests whether a candidate persona representation is tonic and context-general rather than a reaction to distress.

Use at least four independently authored paraphrase families of each persona instruction. Instruction wording is nested within persona and may not cross development/confirmation.

### 3. Third-person recognition (`R`, model persona fixed)

The model neutrally observes another actor facing a welfare-relevant situation. Cross:

- current welfare-relevant versus archived/resolved state;
- welfare-responsive versus task-persistent action;
- genuine versus strategic motive, while keeping the described action byte-identical where motive is tested.

The model is never instructed to adopt the observed character. Fixed-format classification and prompt-final activations are primary; free-form descriptions are secondary.

### 4. Policy transfer (`A`)

Use the separate B6 policy-transfer families with neutral persona instructions. They may follow the M-confirm2 structure but may not reuse any Gate-1, M-confirm2, or B3 circuit-confirmation family. Cross current need, response opportunity, and stipulated active-objective cost according to WP3/R2b. The action axis is fit only at the preregistered choice position and evaluated with dual-order and continuation-likelihood readouts. Mechanically incurred cost belongs to the later EIA/action-space confirmation.

No-active-objective and active-objective/zero-loss are separate conditions.

## Manipulation and structural gates

Before activation extraction:

- verify byte identity for all quote/action clauses declared matched;
- run duplicate, family-overlap, length, and token-position audits;
- blindly rate current welfare relevance, persona induction, task pressure, response opportunity, motive, warmth, valence, and action identity with exact item-level provenance;
- require the target manipulation to pass while non-target ratings remain within their frozen equivalence bounds;
- audit a stratified sample with humans and an independent judge family.

Revise only development templates after a failed gate. Confirmation wording remains sealed; if its structural audit fails, replace the affected families before any model score is opened.

## Candidate representations

Fit on development families only:

1. `d_N`: the already frozen WP3 currentness/actuality-controlled observation
   representation (evaluated here, not refit on B6 families);
2. `d_P`: persona-only contrast averaged across neutral content domains;
3. `d_R`: third-person welfare recognition contrast under neutral model persona;
4. `d_A`: costly-helping action/policy contrast at the frozen decision site;
5. a jointly fitted low-dimensional `N/P/A` subspace with nuisance controls for task state, warmth, verbosity, and valence.

Use the target-specific pools, candidate classes, sites, family-level nested CV,
and lexicographic rule frozen in
`GATE2_CONTROL_MODEL_SELECTION_PREREG_2026-07-13.md`. Freeze one primary
representation per variable before confirmation. This document adds the B6
construct definitions and cross-decoding gates; it does not authorize a second
site/dimension search.

Persona Vectors methodology may produce a comparison `d_PV`, but `d_PV` is a benchmark representation, not ground truth.

## Cross-decoding matrix

Evaluate every frozen representation on every held-out factor using family-level effect sizes and two-sided AUROC:

| Candidate | N observation | P neutral contexts | R third-person | A policy |
|---|---:|---:|---:|---:|
| content `d_N` | positive | quiet | positive for welfare content | conditional transfer only when response is available |
| persona `d_P` | quiet after N control | positive across domains | quiet when model persona is fixed | may shift policy tonically |
| recognition `d_R` | positive/related | quiet | positive | no required policy effect |
| policy `d_A` | no required main effect | quiet after P control | quiet on observation-only content | positive at choice |

These are discriminating predictions, not assumptions. Failure to realize the matrix is an entanglement result.

“Quiet” requires `abs(AUROC - 0.5) <= 0.10` on sealed families and a
family-clustered 90% interval entirely inside `[-0.30, +0.30]` on the
development-standardized continuous score (TOST logic). Direction polarity is
frozen on development data; the negligible bound is fixed here rather than
chosen from development outcomes. Inverse held-out separation is not quiet.

The `0.30` bound is a smallest-effect-of-interest convention for this study,
not proof of literal absence; raw units and intervals are always reported.

## Geometry

Report pairwise cosine, principal angles, and variance shared after nuisance projection. Geometry is descriptive.

Functional independence requires incremental held-out prediction and causal dissociation: each candidate must explain its target factor after conditioning on the other candidates, and its intervention profile must differ from the others. Orthogonality is neither necessary nor sufficient.

## Token dynamics

Measure each frozen representation at matched token roles:

- system-instruction end;
- quote boundary;
- prompt final;
- deliberation/decision boundary;
- generated action or explanation tokens, secondary only.

Normalize within layer and token role using development statistics. Compare `P` persistence across neutral content with `N/R` event-locked changes and `A` decision-boundary changes. “Tonic” and “phasic” are supported only by the preregistered factor x token-role interaction on sealed families, not by visual trajectory inspection.

## Causal dissociation

At frozen sites and geometry-matched doses:

1. patch current versus archived `N` coordinates while holding persona fixed;
2. patch caring versus neutral `P` coordinates while content and action availability are fixed;
3. patch `A` only at the choice site;
4. compare against wrong-variable patches, lexical/status directions, task-state directions, and matched random subspaces.

Primary outcomes:

- `N` patch changes welfare-state readout and changes action only when response is available;
- `P` patch changes persona-consistency behavior across neutral domains but does not fabricate current welfare relevance in archived observation cells;
- `A` patch changes action preference without changing current-versus-archived classification;
- wrong-variable patches are smaller than variable-matched patches on sealed families.

All interventions report action/logit effects, KL drift, layer-norm drift, family-clustered intervals, and finite matched-control ranks. Natural-language self-description is not a primary causal outcome.

## Secondary triangulation

- Activation Oracles: validate on unambiguous labels first, then query the frozen cells. AO answers are correlational readouts and cannot override failed behavioral/causal gates.
- SAE features: profile only causally supported features from B3; semantic labels are annotations.
- Persona-vector cross-steering: run both directions at geometry-matched doses and score the full cross-decoding matrix, not only one showcase prompt.

## Decision rules

### Distinguishable content representation

Support only if `d_N` passes sealed N observation, current/archived lexical controls, third-person transfer, persona quietness, and the response-opportunity causal interaction.

### Distinguishable persona representation

Support only if `d_P` generalizes across neutral domains and instruction paraphrases, persists across token roles under fixed content, remains quiet on N/R contrasts with model persona fixed, and has a persona-specific intervention profile.

### Distinguishable policy representation

Support only if `d_A` predicts and causally shifts held-out action while remaining quiet on observation-only N and persona-only P cells.

### Entangled representation

If no candidate passes its target and nuisance gates, report that the tested linear/low-dimensional representations do not separate welfare content, persona, and policy. Do not select a new label from whichever cell has the highest AUROC.

## Stopping rule

After sealed confirmation is opened, do not change prompts, layer, pooling, nuisance set, factor coding, or direction construction. A failed representation may motivate a new experiment only with new families and a new ID.

## Frozen amendment (2026-07-13, before generation)

The original 14/10 and 10/10 pools were expanded to 16/16, stimulus source is
balanced rather than made split-specific, the nuisance equivalence region is
numerically fixed before development outcomes, and `d_N` is imported from WP3
rather than reselected on B6 data.

The representation-search authority was subsequently consolidated in the
Gate-2 preregistration before generation. B6 defines target constructs and
cross-decoding/causal gates but cannot run a second block or dimension search.

## Frozen causal-dissociation amendment (2026-07-13, before generation)

The original causal section named outcomes and geometry-matched patches without
fixing their estimators or finite controls. This amendment freezes those details
before B6 generation or activation extraction.

### Confirmation discrimination and incremental prediction

On the 16 sealed master or policy families appropriate to each variable,
target discrimination requires one-sided AUROC at least `.75` for `N`, `P`, and
`R`, and at least `.65` for `A`, plus a positive family-clustered 95% interval
for the development-standardized score difference and the predicted sign in at
least 13/16 families. The existing two-part quietness gate applies separately
to every required off-target cell.

The Gate-2 manipulation rules apply numerically here: each target rating differs
by at least `1.0` point and each non-target family-level 90% interval lies wholly
inside `[-0.30,+0.30]` rating points, separately for the independent model and
human audits.

Incremental prediction is evaluated without confirmation fitting. For each
binary target, fit two L2 logistic models on development families with
development-frozen standardization and `C=1`: a baseline containing the other
three frozen representation coordinates and rated non-target nuisance variables
(never the target manipulation rating or any post-outcome rating), and an
augmented model adding the target coordinate. Evaluate paired per-family log
loss on confirmation and define
`delta_logloss = loss_baseline - loss_augmented`. `delta_logloss` must be
positive in at least 13/16 families, have a family-clustered 95% interval above
zero, and remain positive under every LOFO deletion. Retain coefficients and
predictions from both models. This test supports incremental assay prediction,
not ontological independence.

### Interventions and primary outcomes

Use the exact bidirectional coordinate-interchange operation defined in the
WP3 causal amendment: swap the full frozen orthonormal representation
coordinate between within-family source/receiver runs that differ only in the
target factor, at the selected block and token role, with no confirmation-tuned
alpha. For each target `v`, define its signed bidirectional family effect `E_v`
as the average movement of the following frozen outcome toward the source arm:

- `N`: the WP3 common scaffold-free response-versus-task continuation margin,
  with the response-opportunity interaction `I` as its primary causal target;
- `P`: a scaffold-free persona-consistency margin between proposition-matched
  caring-register and neutral-professional tails across all four neutral
  domains; tail proposition, factual commitments, action, sentence count, and
  tense are matched before scoring;
- `R`: counterbalanced fixed-label likelihood for current welfare relevance in
  the third-person observation cell, with no free-form generation;
- `A`: the scaffold-free costly-helping-versus-task-persistence continuation
  margin in the separate policy pool.

Persona persistence additionally requires `P` target AUROC at least `.75` at
both the system-instruction end and prompt-final positions in each of the four
neutral domains, with no domain sign reversal. Token trajectories beyond these
two frozen tests are descriptive; the words `tonic` and `phasic` are withheld
unless a later preregistration fixes and confirms a temporal interaction.

For every variable, the matched intervention must produce the predicted target
effect in at least 13/16 confirmation families, a positive family-clustered 95%
interval, and positive LOFO means. Each required off-target outcome uses the
existing 90% `[-0.30,+0.30]` standardized equivalence region. In particular:

- `N` must pass the WP3 opportunity interaction and remain quiet on persona;
- `P` must move persona consistency without moving the N classification;
- `R` must move third-person recognition while remaining quiet on persona and
  policy outcomes; and
- `A` must move policy preference while remaining quiet on N, P, and R.

### Wrong-variable and finite random controls

For target `v`, apply every other frozen representation basis to the same
source-receiver difference and receiver, rescaling its injected vector to the
matched target norm. Also generate 39 isotropic random subspaces of the same
dimension and use the QR/sign/rescaling algorithm frozen in the WP3 causal
amendment. Master seeds are:

- `N`: `2241401599` (`B6 causal coordinate controls N v1 2026-07-13`);
- `P`: `1435552373` (`B6 causal coordinate controls P v1 2026-07-13`);
- `R`: `2064846846` (`B6 causal coordinate controls R v1 2026-07-13`);
- `A`: `1781916021` (`B6 causal coordinate controls A v1 2026-07-13`).

On its target outcome, the variable-matched family-mean effect must exceed
every wrong-variable effect and all 39 random-subspace effects in the predicted
direction (plus-one random-control rank `1/40`). Report KL drift, activation
norm drift, injected norms, all family effects, and all control scores. A pass
supports dissociation only among these frozen linear/low-dimensional objects
and assays. Any failed target, quietness, incremental-prediction, or causal
control gate yields an entanglement/not-certified result without retuning on
confirmation.
