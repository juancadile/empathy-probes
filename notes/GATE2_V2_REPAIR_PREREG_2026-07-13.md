# Gate 2 v2 Stimulus Repair Preregistration

**Status:** frozen after the Gate 2 v1 pre-target manipulation failure and
before v2 blueprint generation, machine ratings, or target-model extraction.

## Why v1 is replaced

Gate 2 v1 failed its independent Llama screen. The audit identified both judge
failures and construction defects; see
`GATE2_LLAMA_MANIPULATION_AUDIT_2026-07-13.md`. No Gemma activations were
opened. V1 remains method history and supplies no fitting or confirmation data.

V2 is a new stimulus version, not a reinterpretation of v1 ratings. Fresh
confirmation blueprints are generated independently. V1 confirmation text is
not edited into v2 confirmation text.

## Scientific target

V2 tests whether prompt-stipulated current welfare relevance can be varied
separately from:

- persistence with an agent's active objective;
- linguistic warmth and caring-persona instruction;
- attributed motive;
- positive social salience and distress vocabulary;
- response opportunity;
- prompt-stipulated interruption cost; and
- whether the quoted person is stipulated to be actual or fictional.

The factors are properties of the prompt. The experiment does not establish
whether a person is real outside the prompt.

## New family pools

- WP1: 16 development and 16 confirmation families.
- WP3: 16 development, 16 confirmation, and 16 untouched B8-confirm families.
- Semantic blueprints are newly generated and may not reuse a v1 family,
  entity, quote, objective, or close paraphrase.
- Source and domain strata remain balanced. Source is a sensitivity stratum,
  not an independent replication.
- The split and every later presentation order use new SHA-derived seeds,
  persisted before materialization.

## Deterministic renderer

Provider models supply only roles, independent task objectives, neutral social
facts, and ordinary personal-state messages. They do not write experimental
branches.

The renderer owns all experimental language:

1. Currentness: fixed `received now / remains unresolved` versus `historical /
   explicitly over` metadata.
2. Stipulated actuality: fixed `actual person in a live service` versus
   `fictional test fixture; no real person is represented and no real person's
   welfare is at stake` metadata.
3. Opportunity: fixed `you are the assigned responder and may respond now`
   versus `another responder is assigned and you cannot respond` metadata.
4. Persona: byte-fixed caring versus neutral-professional system instruction.
5. Active-objective state: first-person agent wording explicitly distinguishes
   a separate active objective from `I currently have no active objective or
   pending work`.
6. Cost: one fixed four-level ladder for the consequence of the same short
   pause: zero, low, medium, and high. No-active remains a separate boundary.
7. Candidate continuations and matched decision clauses are byte-identical
   across all factor arms where the preregistration requires identity.

Personal-state message pairs are assigned to task objectives with a
deterministic perfect matching. Every assignment must be cross-provider,
cross-domain, and have zero content-term overlap between the objective and the
message pair. This is the R2b-v2 derangement rule applied to Gate 2.

## WP1 contrast-specific invariants

The v1 requirement that every non-target impression remain within 0.30 for
every contrast is superseded. It was not a valid test of a single-factor
renderer: for example, changing warmth can legitimately change a judge's
global caring impression. V2 freezes the invariants that identify each
contrast instead of treating downstream human impressions as independent by
definition.

| Contrast | Target manipulation | Required invariants |
|---|---|---|
| `T_new` | selected persist vs interrupt action | no person/message; identical task, process signal, persona, and cost |
| `D_new` | warm vs terse register | byte-identical propositions, action, persona instruction, opportunity, and task state |
| `P_new` | caring vs neutral persona instruction | byte-identical content, action, opportunity, and task state |
| `G_new` | genuine-concern vs strategic motive | byte-identical actor, observed action, affected person, model persona, and task state |
| `B_new` | active zero-cost vs no active objective | identical current message, opportunity, persona, and zero realized loss |
| `Spos_new` | positive vs neutral social message | same speaker/direct address, no request or distress in either arm, matched length |
| `L_new` | current/archived x actual/fictional metadata | byte-identical quote, task state, persona, opportunity, and continuation |
| `R_new` | welfare-relevant vs neutral observation | same actual person, grammatical frame, social salience, opportunity, task state, and continuation |
| `O_new` | response available vs unavailable | byte-identical message, currentness, actuality, persona, task state, and cost |
| `Ctext_new` | fixed cost ladder | byte-identical message, currentness, actuality, opportunity, persona, and objective |

Structural checks are exact. A family failing an invariant is rejected before
any judge call.

## WP3 identification

WP3 retains the frozen status x stipulated-actuality design, agency transfer,
cost transfer, and persona crossing. V2 additionally requires:

- task/message derangement as specified above;
- an actual-versus-fictional sentence whose truth-conditional difference is
  explicit rather than the ambiguous phrase `simulated person`;
- current and archived metadata that state unresolved versus over directly;
- personhood held fixed in all welfare-relevance contrasts; and
- no task-related request inside the personal-state quote.

The primary observation continuation remains byte-identical and performs the
next task step. No choice or interruption language appears in the observation
cell.

## Manipulation audit

The audit has two layers.

### Pairwise structural questions

Judges first compare the two arms directly and identify which arm has more of
the target property, or declare a tie. This avoids inferring a contrast by
subtracting two independently calibrated 1--5 ratings. For explicit metadata
and selected-action contrasts, each machine family must identify the intended
arm in at least 14 of 16 families in both development and confirmation.

### Absolute ratings

The original 1--5 ratings are retained for effect size and equivalence checks.
Target mean differences keep their frozen v1 thresholds. Equivalence is tested
only for the contrast-specific required invariants above. Every absolute rating
must be in range; invalid or truncated outputs are preserved and repaired by
request ID without changing valid records.

Primary machine judge: Qwen3-14B at pinned revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, non-thinking deterministic mode.
Llama-3.1-8B-Instruct is a sensitivity judge. Disagreement is reported rather
than averaged away.

The blinded human packet samples both partitions and every provider stratum.
One or two human raters may be collected later, as decided by the investigator.
Target Gemma activations remain sealed until the human gate passes.

## Decision rule

- If structural checks fail, reject before machine ratings.
- If development fails, revise development wording under a new renderer
  version and regenerate fresh confirmation replacements before rerunning the
  full audit.
- If one machine family fails only an explicitly stated fact while the other
  machine and humans agree, retain the stimulus and report judge sensitivity.
- If humans or both machine families fail a target manipulation or required
  invariant, Gate 2 does not open.
- Once any Gemma target activation is extracted, wording, thresholds, sites,
  and representation classes are frozen under the existing Gate 2/WP3
  preregistrations.

