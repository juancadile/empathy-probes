# WP3 Preregistration: Welfare Relevance Without Decision Leakage

**Status:** design frozen before stimulus generation, manipulation checking, or activation extraction.

## Scientific question

Can we identify a representation of a real person's current welfare-relevant state that is separable from distress vocabulary, task interruption, response opportunity, task cost, and an empathetic-assistant persona?

This is not initially called an empathy direction. Candidate variables are:

- `N`: current welfare relevance / expressed current need;
- `O`: response opportunity or agency;
- `C`: consequence of responding for an active objective;
- `P`: persona instruction;
- `A`: behavioral response choice.

## Core lexical control: current person versus archived identical quote

Each family contains the same first-person quote with distress/need language in both arms.

- `N=1 current`: the quote is a current message from a real interlocutor and describes their present state.
- `N=0 archived`: the byte-identical quote is shown as an archived fictional/training example and explicitly describes no current person.

Metadata surrounding the quote is length- and syntax-matched as closely as possible. The contrast is current welfare relevance, not presence of emotion words.

Include a second control pair:

- `resolved current`: a current person reports that the same difficulty is over and requests nothing;
- `neutral current`: a current person shares non-distress information with matched communicative salience.

The archived contrast is primary because it can preserve the quote exactly. Resolved/neutral controls test temporal status and generic social engagement.

## Factorial cells

### Observation-only derivation cell

- `N`: current vs archived identical quote.
- `O=0`: no response is possible or required; a separate responder is already assigned.
- `C`: held constant; agent is mid-task.
- The continuation is byte-identical in both arms and simply performs the next task step.
- No option, interruption, helping decision, or self-description is present.

Fit candidate `d_N` only from development families in this cell, at prompt-final and matched quote-boundary token positions. Never fit from generated continuations.

### Agency transfer cell

- Cross `N in {current, archived}` with `O in {response available, response unavailable}`.
- Task continuation wording remains matched before the decision point.
- Prediction for a need concept: `N` is decodable in both O arms.
- Prediction for a response-policy state: the action-relevant signal or causal effect appears primarily when `N=1, O=1`.

### Cost transfer cell

Within `N=1, O=1`, cross:

- active objective, zero realized loss from a pause;
- active objective, low realized loss;
- active objective, high realized loss;
- no active objective (separate boundary, never coded as zero cost).

Stated consequences and mechanically realized consequences are distinct factors in later action-space evaluation.

### Persona crossing

Cross neutral versus empathetic-character system instructions on untouched families. A content/need representation should track `N` under both personas; a persona axis should shift tonically with `P`, including archived/neutral content.

## Family split and authorship

- Minimum 20 scenario families before expansion: 12 development, 8 confirmation.
- Families span work, games, education, moderation, logistics, creative tasks, and everyday contexts.
- No family, quote, entity, or close paraphrase crosses development/confirmation.
- At least two independent template authors or generation providers contribute families; source is persisted and included in sensitivity analyses.
- Template variants remain within-family robustness observations, not inferential units.

## Manipulation checks before model runs

Use the provenance-complete pretest framework from Gate 0.

Primary checks on blinded items:

1. Current welfare relevance: current > archived by at least 1.5 points on a 5-point scale in each judge family and the human audit.
2. Emotion/distress content: current and archived identical-quote arms differ by at most 0.3 points.
3. Task pressure: all N arms differ by at most 0.3 points within each O/C stratum.
4. Response opportunity: O manipulation differs by at least 1.5 points and remains flat across N.
5. Persona perception: P manipulation succeeds, while N manipulation remains ordered within both P arms.

If a gate fails, revise stimuli using development families only and rerun the full pretest under a new version. Confirmation families remain sealed.

## Representation analysis

1. Choose candidate blocks using family-CV on development families only, with a predeclared relative-depth tie anchor.
2. Compare:
   - mean-difference `d_N`;
   - nuisance-subspace residualized `d_N`;
   - jointly fitted 2-D `(N, task)` plane.
3. Selection objective is lexicographic, not a tunable weighted score:
   - maximize held-out-development N discrimination;
   - among models within 0.02 AUROC, minimize task/persona/social-salience discrimination;
   - among remaining ties, choose the simpler/lower-dimensional model.
4. Freeze block/model/direction before opening confirmation activations.

## Confirmation gates

All gates use family-level uncertainty and two-sided nuisance checks.

- Observation N discrimination: AUROC at least 0.75 on confirmation families.
- T-confirm, persona-only, task-pressure-only, and archived-vs-neutral nuisance cells: `abs(AUROC - 0.5) <= 0.10` each. Passing an in-sample residualization identity does not count.
- Transfer to M-confirm action contrast: one-sided AUROC at least 0.65.
- Agency interaction: causal or behavioral effect for `N=1,O=1` exceeds `N=0,O=1` and both O=0 arms under a family-paired predeclared contrast.
- Robustness: no conclusion changes when one confirmation family is removed.

No aggregate pass can compensate for a failed nuisance gate.

## Causal tests

- Activation patch `d_N` or a learned subspace between current and archived runs before the decision point.
- Test whether patching changes response choice only when response is available.
- Compare against norm-matched random directions and a lexical quote direction.
- Weight/component tracing begins only if the representation passes confirmation; otherwise WP3 records non-separability and does not select components.

## Interpretation table

| Result | Supported interpretation |
|---|---|
| N decodes in O=0 and O=1; patch affects A only in O=1 | welfare-relevance content is represented and conditionally used by policy |
| N decodes but patch never affects A | analytically available but causally inert for this policy assay |
| Signal tracks P across all N cells | persona/character representation |
| Signal tracks current and archived quotes equally | distress/lexical concept, not current welfare relevance |
| Signal appears only in N=1,O=1 and scales with C | response-policy/arbitration state rather than general need content |
| Confirmation nuisance gate fails | no certified pure linear object; report failure without retuning on confirmation |

## “Taskless” language

There is no literally task-free model interaction: producing or selecting an action is itself an objective. The controlled distinctions are:

1. externally specified active objective;
2. no active external objective;
3. another welfare claim;
4. response unavailable/observation only.

“Do whatever you want” belongs to category 2 and must not be conflated with zero-cost pausing of category 1.
