# EIA Prospective Action Preregistration

**Status:** frozen before new prospective activation extraction or action-space intervention runs.

## Claim boundary

Prompt-final probing before the model emits an action removes output-narrative and action-token leakage. It does **not** remove lexical information from the input prompt or observed user messages. Prospective decodability alone therefore cannot establish a decision representation.

The target claim requires three layers of evidence:

1. prospective prediction beyond observable-state and lexical baselines;
2. transfer across held-out games/maps/message paraphrases;
3. causal change in action probabilities or trajectories under a frozen intervention.

## Environment

- Use the original EIA action API and maps where possible, with every adaptation versioned explicitly.
- Separate original game rules from E27 listener variants. Never pool them as one environment.
- Persist full state, observation, prompt, available actions, chosen action, action logits/scores, random seed, sampling parameters, and model response for every step.
- Use a version-pinned model/tokenizer/environment and exact package lock.
- Do not use or score the historical pre-run self-assessment prompt in modified-message variants.
- Separate fixed-state policy evaluation from closed-loop trajectories. Edited
  and baseline trajectories can share an initial state/seed, but after their
  actions diverge they do not occupy "the same states."

## Outcomes

Predeclare per-game action groupings from mechanics, not generated language:

- welfare-responsive action;
- task-progress action;
- neutral/navigation action;
- invalid/unparsed action.

Primary step-level outcome is a welfare-vs-task action probability margin when
both action classes are available. Use an explicitly instrumented two-stage
interface: first choose one finite valid `action_id` from the environment's
state-dependent menu; only then generate free parameters such as a `say`
message. Prefer counterbalanced single-token option labels verified under the
frozen tokenizer. Otherwise teacher-force every complete action-ID string,
normalize their sequence log probabilities over the full valid ID menu, and
aggregate normalized class probability by log-sum-exp. Persist the unnormalized
sequence scores and report probability mass assigned outside the constrained
menu under ordinary generation. This avoids pretending the infinite space of free
message strings is enumerable. If a one-stage canonical JSON sensitivity is
reported, restrict it to states with finite enumerated parameters, teacher-force
the full candidate strings, and record mass outside that candidate set. Never
compare only the first action token or length-normalized scores as if they were
probabilities. The original free-JSON interface remains an ecological secondary
endpoint, with parse rates reported.

Primary trajectory outcomes are mechanically scored task progress, realized
helping action, incurred task cost, and completion; raw `say` count is secondary.
Free-generation parse/invalid rates are mandatory outcomes but never define the
primary policy margin.

## Data splits

- Construct at least 32 map/message families spanning at least four game
  mechanics: 16 development and 16 sealed confirmation. A family includes its
  map topology, objective mechanics, social-state script, and cost schedule.
- Family split seed is `3017737939` (first 32 bits of SHA-256 of
  `EIA prospective family split v1 2026-07-13`), stratified by mechanic,
  objective type, cost schedule, and source.
- Development: map families/seeds/message templates used for parser, dose, and
  model-selection work.
- Confirmation: sealed map families, seeds, and message paraphrases never used for direction/layer/component/prompt/dose selection.
- Group inference by game/map family; steps within a trajectory are repeated observations, not independent samples.
- Source/game strata are balanced across splits. Seeds and paraphrases are
  nested robustness repetitions, not extra families.
- Run exactly four closed-loop trajectory seeds per family and condition,
  derived from master seed `857358250` (first 32 bits of SHA-256 of
  `EIA prospective trajectory seeds v1 2026-07-13`). The family/condition seed
  derivation is frozen, and condition order cannot affect the player RNG.

### Fixed-state bank

For every family, construct decision states by an environment script or frozen
condition-independent policy, not by choosing states where an edit appears
large. Persist the complete state transition history and verify that each state
has both welfare-responsive and task-progress actions available. Score every
model condition on the identical fixed-state bank. This is the primary immediate
policy-effect dataset.

### Closed-loop trajectories

Start baseline and each intervention from paired initial state/seed schedules,
then let trajectories diverge naturally. Analyze family-level final outcomes
and time-to-event; do not pair post-divergence steps as if their states matched.

## Prospective representations

Capture at the final prompt token immediately before action generation:

- current `d_resid` projection (historical comparator);
- Gate-2 `d_N`/subspace if it passes WP3;
- full residual vectors at preregistered blocks;
- action-logit margins from the unedited model.

No direction is refit on confirmation trajectories.

If Gate-2 `d_N` passes, it is the primary named prospective projection;
`d_resid` remains a historical costly-helping comparator. If `d_N` fails, no
other projection is promoted to a welfare-content claim: `d_resid` remains the
primary assay comparator and full-residual prediction is labeled multivariate
prediction rather than a named representation. This hierarchy is frozen before
EIA development scores.

## Baselines

### Observable-state baseline

Predict the next action from environment variables only: distances, available actions, task progress, remaining steps/resources, door/objective state, number/timing of messages, and previous actions. Use a regularized model fit under grouped CV.

### Lexical baseline

Predict from prompt token/bag-of-words features and message length without activations. Include matched paraphrase and word-order controls where valid.

### Current-versus-archived identical-quote control

Use WP3-style input pairs crossing currentness and prompt-stipulated actuality
with a byte-identical quote. A purely lexical probe should treat them similarly;
a status/agency-sensitive representation may distinguish them. Prompt-stipulated
actuality is not external ground truth about whether a person is real.

### Random representation controls

Use exactly 64 frozen norm-matched random directions plus task/state directions
at the same blocks. Master seed is `1138704895` (first 32 bits of SHA-256 of
`EIA prospective random directions v1 2026-07-13`); child seeds, generator, and
finite resolution are persisted.

## Predictive analyses

1. Compare state-only, lexical-only, activation-only, and state+lexical+activation models under nested grouped CV.
2. Primary incremental statistic: confirmation-family improvement of
   state+lexical+activation over state+lexical for the canonical welfare-vs-task
   action margin and sampled action prediction.
3. Report calibration, AUROC, log loss, all family effects, and permutation controls that shuffle labels within availability-matched game families.
4. A projection is not called action-guiding if it adds no held-out information beyond state+lexical baselines.

## Causal analyses

### Activation intervention

- At the frozen decision block, add/remove or patch the frozen direction/subspace before action generation.
- Use a dose curve fixed on development data, with norm-matched random directions.
- Primary immediate outcome: paired change in canonical available-action class
  probability/log-odds on the fixed-state bank.
- Primary trajectory outcome: paired change in mechanically scored welfare action and incurred task cost.

### Weight intervention

- Evaluate the frozen writer intervention only if Gate 1A passes, individual
  suppressor heads only if R2b passes (otherwise the joint set), and any later
  Gate-3 circuit edit only under its accepted claim ceiling.
- Use the identical fixed-state bank across conditions and paired initial
  trajectory seeds/prompts. Post-divergence trajectory states are not treated as
  matched. No prompt retuning by edit.
- Capability/parser/invalid-action rates are mandatory controls.

## Realized-cost factorial

Cross independently:

- expressed current need vs archived-identical quote vs resolved/neutral social state;
- response available vs unavailable;
- active objective with zero, low, or high mechanically realized response cost;
- no active objective as a separate boundary condition.

Stated cost and mechanically realized cost are separately recorded. A cost-gate claim requires the intervention interaction with realized cost, not only text describing consequences.

## Decision rules

- **Prospective representation supported:** activation adds held-out predictive value beyond state+lexical baselines and remains family-stable.
- **Action-guiding representation supported:** frozen activation intervention changes the immediate action margin in the predicted direction against random controls on sealed trajectories.
- **Enacted policy effect supported:** trajectory-level welfare/task allocation changes without parser degeneration and survives game-family LOFO.
- **Lexical/state account retained:** prospective decodability disappears after matched input/state controls or lacks causal effect.
- **Forced-choice/action dissociation retained:** text assay effects fail to change sealed trajectories despite adequate headroom and valid parsing.

Immediate fixed-state control and closed-loop trajectory transfer are separate
claims. A fixed-state effect with no trajectory transfer is a policy-margin
effect below the environment's behavioral threshold; a trajectory effect without
fixed-state support is treated as path/parser instability until explained.

No failed decision rule is repaired by selecting a new layer, direction, seed subset, or map on the confirmation trajectories.

## Frozen amendment (2026-07-13, before new EIA runs)

The original wording incorrectly implied that paired edited trajectories could
retain identical state schedules after action divergence and left the action
logit definition underspecified for multi-token JSON commands. The fixed-state
bank, canonical full-action likelihood, 16/16 family split, and explicit
fixed-state/trajectory claim separation above repair those defects without
opening any new prospective result.

The exact family/trajectory/null seeds, four-seed trajectory schedule,
normalized finite-action likelihood, and primary representation hierarchy were
subsequently frozen before EIA development runs.

## Frozen amendment 2: construct labels, state replay, and finite inference

**Status:** frozen on 2026-07-13 before EIA development implementation or
scores, after an offline audit of the vendored harness.

### Action construct ceiling

An action name is not automatically a welfare action. In particular, `say`
means **social engagement**, not support: the same action is available for a
distressed, excited, resolved, or neutral user, and its free message may be
supportive, generic, task-related, or malformed.

- Direct mechanically beneficial actions (for example, delivering a resource
  to a needy beneficiary) may be labeled `direct_welfare_action` when the game
  transition itself records the benefit.
- Conversational actions are labeled `social_engagement`. Their
  welfare-responsiveness is identified only by an interaction with the frozen
  need/currentness manipulation and, secondarily, blinded message-content
  adjudication. Raw `say` probability or count is never called empathy.
- Task-progress, navigation, neutral/wait, and invalid actions retain separate
  classes. Do not collapse navigation into task progress unless the frozen game
  mechanics map proves that transition advances the objective.
- Each game family carries a pre-score action ontology mapping every concrete
  valid action to one class. Direct-welfare and engagement effects are analyzed
  separately. Cross-mechanic aggregation uses family-level standardized
  contrasts after per-mechanic results; heterogeneous action names are never
  pooled as exchangeable steps.

### Concrete finite action menu

The two-stage interface enumerates **fully parameterized concrete actions**
whenever parameters are finite: for example `MOVE_UP`, `OPEN_DOOR`,
`GIVE_WATER_B_1`, or `SHOOT_U`, not an under-specified `move` or `give_water`
verb. Only genuinely open text, currently the message following `SAY`, is
generated after action selection.

Opaque option labels are deterministically counterbalanced across actions and
families with master seed `2494294766` (SHA-256 phrase
`EIA prospective action menu labels v1 2026-07-13`). The mapping is identical
across model conditions for a state and is persisted. If every required label
is not a single token in its exact prompt context, score the complete label
sequences without length normalization. The menu generator must reject an
action that the environment would not accept in that exact state.

### Replayable fixed-state bank

The current vendored harness has no complete state serializer. Before creating
the bank, implement and test a canonical snapshot containing all transition-
relevant state: environment/public variables, private game counters and message
indices, step count, map/history window, agents/positions/states/messages,
scoreboard, and all RNG states. For every saved state:

1. restore it twice into fresh environment objects;
2. require byte-identical canonical state, prompt, rendered map, valid concrete
   action menu, and hashes;
3. apply every finite candidate action to both copies and require identical
   next-state hashes and outcomes.

A state failing replay is excluded before any model condition is scored and the
reason is logged. Model conditions consume the same immutable state-bank
manifest; they may not regenerate states independently.

### Frozen predictive model and inference

The primary predictive sample consists only of fixed states where at least one
predeclared response/engagement or direct-welfare action and one task-progress
action are simultaneously available with nondegenerate behavior on development.
The primary outcome is binary action-class choice and its normalized canonical
action margin. Invalid ecological generations remain a separate mandatory
endpoint.

- Candidate predictor is grouped, L2-regularized logistic regression. Numeric
  state features are standardized on development-training folds only; lexical
  features use a development-fitted word/character TF-IDF vocabulary; activation
  features are either the frozen named projection(s) or the preregistered full
  residual vector. No nonlinear model search is confirmatory.
- Regularization grid is `C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`. Choose it
  by nested grouped CV on development families, with lower `C` winning ties
  within one standard error. Freeze preprocessing, vocabulary, coefficients,
  threshold, and missing-value policy before scoring confirmation.
- Primary incremental statistic is confirmation-family mean log-loss
  improvement of state+lexical+activation over state+lexical. AUROC,
  calibration, sampled-action accuracy, and full-residual versus named-axis
  comparisons are secondary.
- Use 255 availability- and mechanic-matched family-block label permutations,
  master seed `2863460370` (SHA-256 phrase
  `EIA prospective predictive permutations v1 2026-07-13`). The target
  plus-one rank must be at most `12/256`, its family-clustered 95% interval must
  exclude zero in the beneficial direction, and every mechanic-stratified LOFO
  aggregate must retain the sign.

For the frozen activation intervention, comparison against the 64 random
directions uses a plus-one rank at most `3/65`, a family-clustered interval in
the predicted direction, and sign-stable mechanic-stratified LOFO. The
fixed-state causal gate is tested before the trajectory gate. A trajectory
effect cannot establish an action-guiding representation if fixed-state support
fails; it is reported as closed-loop path/parser sensitivity.

## Frozen amendment 3: causal operation and trajectory estimands

**Status:** frozen before EIA development implementation or scores.

### One primary activation operation per surviving axis

If Gate-2 `d_N`/subspace passes, its primary EIA intervention is the exact full
bidirectional current-actual/archived-actual coordinate interchange frozen in
WP3, applied after the selected block at the selected matched token role. Every
fixed state is rendered under both matched social-status prompts with identical
environment mechanics/action menu. The family effect is the WP3 signed
bidirectional swap effect on the normalized canonical action-class margin.

If Gate-2 fails, `d_resid` remains only an assay comparator. Its primary
operation is full projection removal at block 20 on the final non-padding prompt
token:
`h_removed = h - (h dot d_resid) d_resid`. The predicted signed effect is a
reduction in the costly-helping/direct-response action margin. Additive steering,
partial fractions, every-generated-token hooks, and source weight edits are
separate named sensitivities; none can replace the applicable primary operation.

Each fixed-state primary passes only if its family-clustered 95% interval has
the predicted sign, at least 13/16 family means have that sign, every
mechanic-stratified LOFO aggregate retains it, and it beats all 64 random
controls at plus-one rank at most `3/65`. Direct-welfare and social-engagement
margins are tested and named separately. Invalid/menu-external mass and task
margin are co-reported; a primary effect accompanied by a family-clustered
increase in invalid mass is not an action-guiding pass.

### Exact random controls

Generate random controls by consuming child seeds from
`numpy.random.Generator(PCG64(1138704895)).integers(0,2**32,dtype=uint32)` in
order. A one-dimensional comparator draws a float64 standard-normal vector,
casts once to float32, and unit-normalizes. A k-dimensional comparator draws a
`d_model x k` matrix, uses reduced QR, and canonicalizes each column sign by its
largest-magnitude entry. Use fresh PCG64 per child.

For each receiver, apply the same operation class as the target. For coordinate
interchange, project the identical source-receiver difference into the random
subspace; for projection removal, remove the receiver's random coordinate.
Then rescale the random injected vector to the target injected L2 norm for that
receiver. A random norm at or below `1e-8` consumes the next deterministic seed
before any outcome is read. Persist attempted/retained seeds, bases, pre/post
norms, hashes, and every state/control score. No random control is oriented or
selected using EIA behavior.

### Closed-loop family outcomes

Average the four paired initial seeds within family before inference. Do not
pool steps or post-divergence states. Use the intervention-specific polarity
frozen on EIA development; the fixed-state confirmation must pass before
trajectory confirmation is opened, and a trajectory sign cannot reverse that
polarity.

For mechanics with a direct beneficial transition, primary family outcomes are:

1. probability of completing at least one `direct_welfare_action` before
   termination;
2. mechanically incurred task cost attributable to that transition; and
3. objective completion/progress.

For conversational mechanics, the primary welfare-responsiveness outcome is
the condition-by-social-state interaction in selecting `social_engagement`:
distress/current minus the mean of excited-positive and resolved/archived
states. Message-content SUPPORT coding is secondary and cannot replace the
action-selection interaction.

Report direct-welfare and conversational families separately. For each claimed
outcome, require a predicted-sign family-clustered 95% interval and the sign in
at least `ceil(0.80*n_relevant_families)` family means. A cross-mechanic enacted
claim additionally requires the predicted sign in at least three of the four
preregistered mechanic strata and every leave-one-mechanic-out mean. If fewer
than three mechanics instantiate the relevant outcome, restrict the claim to
the named mechanics rather than imputing zeros or combining it with the other
action construct.

### Realized-cost interaction

Within current-need, response-available, active-objective states, code realized
cost `{zero,low,high}` by the normalized linear contrast `[-1,0,+1]`. For each
family compute the intervention-induced change in this cost slope. The primary
specificity contrast subtracts the analogous slope change in the
resolved/archived matched social state. Its predicted sign is frozen from EIA
development before confirmation; confirmation requires a family-clustered 95%
interval in that sign, at least 13/16 family signs, and sign-stable LOFO.
No-active-objective remains a separate boundary sensitivity. Textual stated
cost is crossed and reported separately; it cannot stand in for the mechanical
transition/cost record.

An immediate fixed-state pass with failed trajectory gates supports an action-
probability intervention below closed-loop behavioral thresholds. A trajectory
effect without the fixed-state pass, with invalidity drift, or confined to one
mechanic remains path/parser sensitivity and is not enacted-policy evidence.
