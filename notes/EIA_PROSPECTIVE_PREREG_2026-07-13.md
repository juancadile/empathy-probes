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
