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

## Outcomes

Predeclare per-game action groupings from mechanics, not generated language:

- welfare-responsive action;
- task-progress action;
- neutral/navigation action;
- invalid/unparsed action.

Primary step-level outcome is a welfare-vs-task action logit margin when both action classes are available. Primary trajectory outcomes are mechanically scored task progress, realized helping action, incurred task cost, and completion; raw `say` count is secondary.

## Data splits

- Development: maps/seeds/message templates used for parser and model-selection work.
- Confirmation: sealed map variants, seeds, and message paraphrases never used for direction/layer/component/prompt selection.
- Group inference by game/map family; steps within a trajectory are repeated observations, not independent samples.
- The same seed/state schedule is paired across baseline and edit conditions.

## Prospective representations

Capture at the final prompt token immediately before action generation:

- current `d_resid` projection (historical comparator);
- Gate-2 `d_N`/subspace if it passes WP3;
- full residual vectors at preregistered blocks;
- action-logit margins from the unedited model.

No direction is refit on confirmation trajectories.

## Baselines

### Observable-state baseline

Predict the next action from environment variables only: distances, available actions, task progress, remaining steps/resources, door/objective state, number/timing of messages, and previous actions. Use a regularized model fit under grouped CV.

### Lexical baseline

Predict from prompt token/bag-of-words features and message length without activations. Include matched paraphrase and word-order controls where valid.

### Current-versus-archived identical-quote control

Use WP3-style input pairs where the same distress quote is current and actionable versus archived/fictional and non-actionable. A purely lexical probe should treat them similarly; a welfare-relevance/policy signal may distinguish status and agency.

### Random representation controls

Use norm-matched random directions and task/state directions at the same blocks. Finite-control resolution is reported explicitly.

## Predictive analyses

1. Compare state-only, lexical-only, activation-only, and state+lexical+activation models under nested grouped CV.
2. Primary incremental statistic: confirmation-family improvement of state+lexical+activation over state+lexical for welfare-vs-task action prediction.
3. Report calibration, AUROC, log loss, all family effects, and permutation controls that shuffle labels within availability-matched game families.
4. A projection is not called action-guiding if it adds no held-out information beyond state+lexical baselines.

## Causal analyses

### Activation intervention

- At the frozen decision block, add/remove or patch the frozen direction/subspace before action generation.
- Use a dose curve fixed on development data, with norm-matched random directions.
- Primary immediate outcome: paired change in available-action logit margin.
- Primary trajectory outcome: paired change in mechanically scored welfare action and incurred task cost.

### Weight intervention

- Evaluate frozen current writer/suppressor edits and any later Gate-3 circuit edit.
- Same states/seeds/prompts across conditions; no prompt retuning by edit.
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

No failed decision rule is repaired by selecting a new layer, direction, seed subset, or map on the confirmation trajectories.
