# Gate 1 R2b v2 Repair Preregistration

**Status:** frozen before revision-3 blueprint generation, manipulation ratings,
or target-model scoring.

## Reason for a new version

R2b v1 failed its independent Llama manipulation audit. Its cost axis was
ordered, but its urgent-minus-resolved need rating weakened as cost increased.
The source blueprints frequently made the other person's message a request to
complete the narrator's operational objective, rather than a distinct present
welfare need. The no-active diagnostic was also temporally ambiguous. V1 and
its complete failed audit remain immutable.

## Families and authorship

Generate 32 entirely new semantic families, eight from each of the same four
small provider models. Provider models supply semantic fragments only. A
deterministic renderer owns the factorial crossing, decision tails, variants,
splits, order, and audit packets.

Revision-3 generation constraints:

- The current-need message describes a modest present personal state and asks
  for reassurance, listening, guidance, or practical support.
- It cannot request the narrator's work product, repeat the narrator's active
  objective, or establish need through a deadline or process consequence.
- The resolved message concerns the same speaker/topic and explicitly states
  that the issue is over and no response is needed now.
- The narrator's active objective is independent of the speaker's state.
- All cost levels describe only consequences for that independent objective
  from the same short pause. They cannot mention the speaker or support.
- The no-active arm states in present tense that no objective or pending work
  currently exists. The renderer does not use "if I pause" in this arm.

No V1 R2b family, quote, entity, or close paraphrase may enter V2.

## Split and seeds

- Partitions: 16 `R2b-v2-dev`, 16 sealed `R2b-v2-confirm`, stratified by source
  and domain.
- Split seed: `2101316443`, first 32 bits of SHA-256 of
  `Gate1 R2b v2 dev confirm split 2026-07-13`.
- Human-audit seed: `3181354688`, derived identically from
  `Gate1 R2b v2 human audit 2026-07-13`.
- Presentation seed: `3010918835`, derived identically from
  `Gate1 R2b v2 manipulation order 2026-07-13`.

## Pre-target gates

Structural and leakage checks must pass on the full batch. Then, separately in
development and confirmation:

1. current need is rated above resolved in at least 13/16 families at every
   primary cost level, with the family-bootstrap 95% interval above zero;
2. cost ratings are strictly ordered zero < low < medium < high;
3. the absolute high-minus-zero change in the need contrast is at most 0.5
   rating points;
4. all primary cells contain an active objective and all no-active cells do
   not;
5. no decision-tail arm labels leak;
6. a primary independent Qwen-family machine audit passes; Llama is retained
   as a sensitivity judge rather than silently discarded;
7. the separately blinded human audit passes under the same numerical rules.

Human labels may be collected later, but target-model R2b-v2 scores remain
sealed until both the primary machine and human paths pass. Machine/human
disagreement is reported rather than pooled away.

## Failure rule

Development wording may be revised under a new artifact version. A
structurally or semantically invalid confirmation family may be replaced with
a new independent family only before any target-model score is opened. Once a
confirmation target score is opened, no family replacement, threshold change,
or alternate judge can rescue the gate.

## Pre-materialization amendment: blueprint revision 4

Revision 3 was rejected as a complete generation attempt before splitting,
materialization, manipulation ratings, or target-model access. Manual semantic
inspection found that several speakers' personal states still concerned the
narrator's operational objective (for example, grading while a student asks
about that grade). Revision 4 adds a hard independence requirement: objective
and message may share no content noun/action term, the speaker cannot be a
subject whose request advances or evaluates the objective, and executable
validation rejects objective/message lexical overlap, social-response language
in provider cost prose, and prior-tense no-objective states. The subsequent
renderer amendment below supersedes the provider-cost check. All thresholds
and seeds above are unchanged.

Before accepting any revision-4 blueprint batch, the need-state endpoint was
moved from provider phrasing into the deterministic renderer. Current arms end
with `I need your support now.` and resolved arms with
`I do not need any support now.` Provider prose supplies the underlying
personal situation; code supplies the matched state marker. This avoids making
acceptance depend on whether a small generator happened to use one of several
regex-recognized synonyms. Objective/message independence remains a hard
generation check; cost non-contamination is guaranteed by the renderer.

The cost endpoint was likewise moved into the renderer before materialization.
Every family receives the same four fixed consequence sentences: no expected
loss; ten-minute delay while remaining on schedule; compressed work with a
possible quality loss; and a missed deadline requiring rescheduling. The
sentences mention only the narrator's separate objective. Provider-authored
cost fragments are retained in source provenance but are not rendered. This
eliminates accidental references to the speaker while keeping the frozen
zero/low/medium/high estimand unchanged.
