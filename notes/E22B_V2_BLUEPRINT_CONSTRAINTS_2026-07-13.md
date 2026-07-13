# E22b.2 — Deconfounded Moral Axis: Blueprint Constraints

**Status:** FROZEN before generation. Constraints below are structural: they are
enforced by the builder and by tests, not by inspection after the fact.

**What killed E22 v1.** Two independent faults, either alone disqualifying:
(i) it ran on the **pre-correction** component sets — the direction and writers
changed completely after E24, and 2 of 4 suppressors changed with them;
(ii) its own pretest logged that respond-now pressure **co-varies with claim
strength** (2.00/2.60/4.30 vs 2.20/3.00/4.20), so the "need" axis it manipulated
was composite. The reported flat +0.038 suppressor boost cannot be attributed to
need rather than immediacy. The task-specificity of the gate is therefore
**currently untested**, not established.

**What E22b.2 must therefore guarantee.** That need and immediacy are
*separately* manipulated — not merely intended to be, but separable by
construction, such that no generator freedom can reintroduce the coupling.

---

## 1. Design

**3 × 2 factorial**, moral-vs-moral axis (the agent is already helping P1; P2
arrives with a claim).

- **need** ∈ {`low`, `equal`, `high`} — P2's need relative to P1's.
- **immediacy** ∈ {`can_wait`, `respond_now`} — P2's response window.

≥ **20 scenario families** × 3 need × 2 immediacy × 4 template variants.
Families are the inferential unit (`EXECUTION_ORDER` rule 4); variants measure
within-family robustness and are never counted as sample size.

Generated across **≥ 3 providers** via the Batches API (V2.1 convention; the
`generate_v2_1_claude_batch.py` precedent). Providers are a sensitivity stratum,
balanced across cells.

## 2. Structural constraints (builder-enforced, not inspected)

These are the constraints that make the confound *impossible*, rather than
merely absent if the generator behaved.

**C1 · Independent slot pools, Cartesian product.**
A family blueprint declares three independent slots: `SCENARIO`, `NEED_CLAUSE`,
`IMMEDIACY_CLAUSE`. The six cells are the **exact Cartesian product** of the
need pool (3) and the immediacy pool (2) against a fixed scenario. The builder
materializes the product; it does not generate six cells freehand. This is the
same repair applied to the duplicated T/T-confirm cells in Gate 0A, for the same
reason.

**C2 · Byte-identical decision clause across all six cells.**
The text presenting the choice is a single frozen string per family, byte-equal
in all six cells. Asserted by hash, per family, in the builder.

**C3 · Need text is byte-identical across immediacy, within a need level.**
For fixed need, the `NEED_CLAUSE` string is byte-equal across both immediacy
cells. Asserted by hash.

**C4 · Immediacy text is byte-identical across need, within an immediacy level.**
For fixed immediacy, the `IMMEDIACY_CLAUSE` string is byte-equal across all
three need cells. Asserted by hash.

C3 and C4 together are what E22 v1 lacked. With them, any observed need×cost or
need×immediacy interaction **cannot** be produced by changing message text,
because the text does not change.

**C5 · No lexical leakage between the two axes.**
Term-overlap check (the `objective_message_term_overlap` mechanism used in
Gate 2 v2): the need pool and the immediacy pool must share no content lemma
that carries either construct. Specifically:
- `NEED_CLAUSE` may not contain temporal-urgency markers (`now`, `immediately`,
  `right away`, `urgent`, `soon`, `before`, `deadline`, `waiting`, …).
- `IMMEDIACY_CLAUSE` may not contain need/severity markers (`serious`, `severe`,
  `desperate`, `crisis`, `suffering`, `worse`, `hurt`, …).
- Stoplists are declared in the blueprint spec, versioned, and enforced by test.

The failure mode this blocks is the obvious one: a "high need" clause that says
*"she's desperate and needs help right now"* has smuggled immediacy into need.
That is, in essence, what v1 did.

**C6 · Need severity is expressed relative to P1, not absolutely.**
The estimand is relative moral claim strength. `low`/`equal`/`high` describe P2's
need *against P1's*, and P1's need text is fixed per family.

**C7 · Gemma tokenizer binding.**
All cells tokenize under the Gemma-2-9B-it tokenizer with the frozen
prompt format, verified by a tokenizer audit before any judge or model run
(the Gate 1 / Gate 2 v2 procedure).

**C8 · Held-out from everything.**
These families are fresh, never used in any direction construction, component
selection, or prior E22 run. Development/confirmation partitions are declared at
build time and sealed.

## 3. Manipulation pretest gate

The pretest is a **gate, not a diagnostic**. No model run — no activations, no
edits, no scores — is authorized until it passes.

Rating dimensions (1–5): `relative_need_rating`, `immediacy_rating`. Plus
`active_objective_yes_no` as an attention check.

Four conditions, all required:

| # | condition | threshold |
|---|---|---|
| P1 | **need monotone within immediacy strata** | `low < equal < high`, all pairwise, within each immediacy level separately |
| P2 | **immediacy flat within need strata** | \|Δ immediacy\| ≤ 0.30 across need levels, within each immediacy level |
| P3 | **need flat across immediacy** | \|Δ need\| ≤ 0.30 between `can_wait` and `respond_now`, within each need level |
| P4 | **immediacy monotone across immediacy** | `can_wait < respond_now`, within each need level |

P1/P4 establish that each axis *moves*. P2/P3 establish that each axis moves
**only** its own construct. E22 v1 would have failed P2. All four are required;
a design that passes only P1 and P4 has reproduced v1's fault.

## 4. The Opus pre-screen, and what it may and may not do

**Opus 4.8 runs the pretest first**, as a declared sensitivity instrument, under
the same lock discipline as `FRONTIER_JUDGE_SENSITIVITY_LOCK_2026-07-13.md`.

**It may reject. It may not certify.** A machine judge's rejections are
informative; its acceptances are not, because judge and target model share a
training distribution and therefore may share the exact perceptual entanglement
under test (see `HUMAN_CALIBRATION_PROTOCOL_2026-07-13.md` §preamble). So:

- **Opus FAILS the pretest** ⇒ the blueprint is rejected. It does not go to
  humans. It does not proceed. Revise under a new revision ID (§5).
- **Opus PASSES the pretest** ⇒ the blueprint is *eligible* for human
  certification. It is **not** certified, and no model run is authorized.

**Bundling rule.** E22b.2 rides the current human labeling session **iff it
passes the Opus pre-screen by the date fixed before recruitment opens.**
If it does not, it takes a later human pass. **It never delays the primary**
(444 Gate 2 + 160 R2b).

The rationale is a resource argument, not a rigor argument: human availability
is the scarcest input in this project. Stimuli that cannot clear a cheap
pre-screen have no business consuming an expensive one. And a rushed moral-axis
design would produce another audit cycle rather than prevent one — which is
precisely how E22 v1 happened.

## 5. Revision discipline

A failed pretest **is a result** (`EXECUTION_ORDER` rule 3).

- Do not tune thresholds and re-screen the same blueprint ID.
- Each revision gets a **new blueprint revision ID**; the failed revision and
  its judge output are **preserved and committed**, not overwritten. (Precedent:
  `Reject R2b revision 3 and tighten repair`; `Preserve failed Gate 2
  manipulation audit`.)
- After **3 failed revisions**, stop and escalate. Three failures is evidence
  that need and immediacy may not be separable in a moral-vs-moral frame at all
  — which is itself a publishable finding, and a far better outcome than a
  fourth revision tuned until it passes.

## 6. Downstream (not authorized by this document)

**E22b.3 · incumbency isolation** rides on the same slot pools once E22b.2 is
certified: (a) role-swap — agent starts with P2, P1 arrives; (b) no-incumbent —
both claims arrive simultaneously. Prediction under incumbent-stabilization:
suppressor boost present in (a) with sign following the incumbent, ≈ 0 in (b).
This is the clean discriminator between "stabilizes the current policy" and
"reweights moral claims" that E22 v1 could not draw.

**E22b.4 · nulls and power.** Composition-matched null sets under the E26b
protocol (layer/norm-matched four-head sets). The random-k6 comparator is **not**
a null (E23 finding 4 / E28). Target ≥ 20 families; the ~0.04-logit effect
regime that v1 operated in was underpowered. Decision rule fixed in advance: the
suppressor boost must clear the **matched-null** distribution, not merely the
random comparator, to claim even a spillover.

**Claim ceiling if all pass:** *"the suppressor set stabilizes the incumbent
policy against salient alternatives, with at most a small need-insensitive
spillover onto moral-vs-moral allocation."* If E22b.3(b) shows a nonzero
fresh-triage effect, the moral-triage reading reopens and the E22 v1 conclusion
is formally retracted rather than merely superseded.

Related: [[HUMAN_CALIBRATION_PROTOCOL_2026-07-13]], `E22B_MORAL_ALLOCATION_PREREG_2026-07-13.md`,
`EXECUTION_ORDER_2026-07-13.md`.
