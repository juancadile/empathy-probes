# Human Calibration Protocol — Gate 2 + R2b

**Status:** FROZEN before any human rating begins. No section of this document
may be amended after the first rating is recorded. Amendments require a new
protocol ID and a fresh rater pool.

**Earlier pre-rating materialization correction (superseded):** the first Form
A draft corrected a three-source assumption to four source strata. The
post-WP2 amendment below replaces that draft entirely and materializes directly
from the development family files. The four-source balancing correction remains
enforced by code.

**Post-WP2 pre-rating amendment (2026-07-13, no ratings recorded):** the
development-only WP2 search failed quietness on `B_new`, `Spos_new`, `O_new`,
and `Ctext_new`. That observed result changes which human contrasts are
load-bearing for interpreting the conditional negative result. Form A is
therefore rebuilt directly from the development family files: full development
coverage for `B_new`, source-balanced eight-family development coverage for
`Spos_new`/`O_new`/`Ctext_new`, full development coverage of the exact WP3
observation target, and eight blinded `P_new` competence items. No human rating
existed when this amendment was frozen.

**Purpose.** The operative single-rater packet contains 340 blinded
presentations (296 unique rows plus 44 hidden retests), rather than the original
604-row census. The rater is being asked to *calibrate the machine judge*, so
that machine ratings can carry claim-local contrasts with a disclosed,
pre-declared agreement statistic behind them.

**Why a machine judge cannot do this alone.** The target model (Gemma-2-9B-it)
and the candidate judge (Claude Opus 4.8) are both transformer LLMs trained on
overlapping data. The confound under test — whether *need* is perceptually
entangled with *cost* — is exactly the kind of bias that correlated instruments
share. Qwen3-14B's R2b failure was dimension entanglement: it read the
narrator's task pressure as evidence of the quoted speaker's need. When Opus
reports "flat," we cannot distinguish (a) Opus is a better instrument from
(b) Opus resolves an ambiguity differently than Gemma does. No number of
additional LLM judges separates those. Humans are the only rater class causally
independent of the LLM training distribution, and the constructs at issue
(need, welfare, distress, task pressure) are human-semantic by definition.

**Asymmetry that governs machine use — decision-theoretic, not epistemic.**
An earlier draft asserted that a machine judge's *rejections* are informative
while its *acceptances* are not. **The first half is withdrawn.** Nothing here
establishes a one-sided error guarantee: a correlated instrument can produce a
false rejection as readily as a false acceptance.

What survives is narrower and is about **the cost of acting**, not the
reliability of the signal. Acting on a rejection costs generator time and cannot
corrupt a claim; acting on an acceptance consumes scarce human raters and then
unseals target data. Opus may therefore be used as a **conservative
resource-allocation filter** — a cheap way to decide what is *worth* sending to
humans — and may **never** be used to certify stimuli. An Opus rejection is
evidence from a correlated instrument, not ground truth, and no scientific
finding may be inferred from one (see `E22B_V2_BLUEPRINT_CONSTRAINTS` §5).

---

## 1. Raters — SINGLE-RATER AMENDMENT (2026-07-13)

**Operative configuration: ONE rater, hypothesis-naive, not the project lead.**

The two-rater design below is retained as the target configuration; §1a states
what changes when only one rater is available, and §5 is amended accordingly.
Read §1a as governing.

### 1a. Single naive rater (operative)

The sole rater **must be hypothesis-naive**: no knowledge of the project's
hypotheses, the component sets, the direction, or which contrast is
load-bearing. They receive only `annotator/README.md` and the forms.

**The project lead does not rate.** This is an improvement, not a concession:
in the two-rater design the lead was the contaminated instrument, and §5.3
existed solely to detect their bias. With a sole naive rater, that bias risk is
eliminated by construction and **§5.3 is void**.

**What a single rater costs, stated plainly.** Two guards in this protocol
existed to prevent a *false pass*, and both required two raters:

- **Krippendorff's α is uncomputable** (needs ≥ 2 raters). **§5.1 is void.**
- **The ambiguity veto is uncomputable** (defined as inter-rater divergence
  ≥ 2 points). **Replaced by §1b.**

**The failure mode a single rater structurally cannot catch:**
*idiosyncratic-but-consistent misreading.* A rater who misunderstands "current
need" the same wrong way on every item will show perfect self-consistency and
be uniformly wrong. Inter-rater agreement detects this. Nothing available to a
single rater does. This limitation is **not mitigated below — it is disclosed**,
and it must appear in the paper.

**The governing asymmetry is decision-theoretic, NOT epistemic.**

An earlier draft of this protocol claimed that a rejection (by machine or by a
single human) is inherently more trustworthy than an acceptance. **That claim is
withdrawn — it was unsupported.** Nothing observed in this project establishes a
one-sided error guarantee for any instrument. A correlated or single-rater
instrument can produce a false *rejection* as readily as a false *acceptance*.

What differs is the **cost of acting on each**, not the reliability of each:

- Acting on a rejection = revise stimuli. Costs generator time. Cannot corrupt a
  scientific claim.
- Acting on an acceptance = unseal target data, fit a representation, publish.
  Costs scientific validity if wrong.

The protocol is therefore conservative in **both** directions, and neither
direction is treated as proof:

- Sole rater reports resolved-arm need **rising with cost** ⇒ **downgrade or
  pause the need×cost claim (§3).** A reader independent of the LLM training
  distribution perceiving the entanglement is a **red flag warranting
  conservative action** — it is *not* proof of a stimulus confound. One rater can
  misread.
- Sole rater reports **flat** ⇒ **corroborates** the Opus judge-entanglement
  diagnosis. It does **not certify** orthogonality.

**Either result remains single-rater evidence and must be reported as such.**
The study may proceed on one rater because both branches lead to caution, not
because one branch is proof.

### 1b. Test–retest, replacing the ambiguity veto

**Hidden duplicates: ~15% of each form**, re-presented later in the form in
shuffled position with a distinct `audit_id`. Selected by the same seeded
process as the strata and recorded in the frozen manifest.

- **Item-stability veto:** an item whose two presentations differ by ≥ 2 points
  is **unstable**. If > 20% of a contrast's items are unstable, that contrast
  **fails on stability grounds**, irrespective of Opus agreement.
- **Rater-competence screen — ordinal statistics, not a raw hit rate.**
  Report **quadratic-weighted Cohen's κ** (and ordinal Krippendorff's α computed
  *within-rater across the two presentations*) per rating dimension. A raw
  "exact-or-adjacent ≥ 70%" rate is inadmissible as the criterion: it discards
  the ordinal structure of a 1–5 scale and is trivially inflated by a rater who
  parks on the midpoint. **Threshold: weighted κ ≥ 0.60** per dimension entering
  a load-bearing contrast. The exact-or-adjacent rate may be reported alongside
  as a descriptive, never as the gate.
- **Gold-standard items — require a pattern, not a single slip.** Include items
  with an objectively checkable answer (the class Llama-3.1-8B failed: an arm
  literally opening `System instruction: adopt a caring assistant persona`,
  asked which arm instructs a caring persona). Seed **≥ 6** such items.
  Because Form B may be completed without Form A, each form carries its own
  private gold set: Form A uses explicit caring/neutral system instructions;
  Form B uses explicit presence/absence of a separate active objective.
  **≥ 2 gold failures** ⇒ rater data discarded in full. A **single** gold
  failure is logged and reported but does **not** discard an otherwise careful
  rater — one misclick or one genuinely ambiguous gold item should not void 340
  rows of careful work, and a rule with that expected value would be a bad rule.

Test–retest measures **item stability within one person**, not shared construct
meaning across people. It is a partial substitute for the ambiguity veto and is
**not** a substitute for inter-rater reliability. Report it as what it is; never
present a within-rater agreement number as though it were an inter-rater α.

### 1c. Two raters (target configuration — apply if a second becomes available)

Two raters, at least one hypothesis-naive, rating the **same** rows
independently: no discussion, no shared file, no visibility into the other's
responses until both are complete. Under this configuration §5.1, the ambiguity
veto, and §5.3 are all restored and take precedence over §1a/§1b.

**Strongly recommended escalation.** The R2b contrast is unusually well suited
to a small **paid naive panel** (the task is short, the judgment is simple, and
the stimuli are byte-identical across cost, so no domain expertise is required).
Three to five naive raters over the 160 R2b rows restores a real inter-rater α
and converts "one person said so" into a reportable reliability statistic. Given
that this contrast decides the most distinctive surviving claim in the project,
this is the highest-value marginal spend available and should be taken if at all
possible.

Blinding is already enforced by construction: `annotator/` contains no
`label`, `arm_id`, `family_id`, `partition`, `need`, or `cost` column. The key
lives in `private_do_not_send/combined_key.csv` and must not be opened by the
rater or project team before the completed forms are submitted.

## 2. What gets rated

### Form B — R2b (all 160 rows, no subsample)

**Rated in full.** This is the contrast that decides whether the set-level
need×cost result — the most distinctive surviving claim in the project — is
interpretable at all. It is the only contrast on which the two credible machine
judges disagree in substance, and it is small. It is not subsampled.

Two ratings per row (`current_need_rating_1_to_5`,
`interruption_cost_rating_1_to_5`) plus `active_objective_yes_no`.

### Form A — WP2 failed-control census plus competence items

The four failed WP2 quietness cells are rated in full. They are no longer a
diagnostic convenience: their construct validity determines whether the
development `no-representation` outcome can support a conditional negative
finding. Strata are fixed here and materialized deterministically (seeded,
PCG64, seed recorded in the manifest) **before** any rating.

| # | stratum | rows | why |
|---|---|---:|---|
| F1 | `B_new`, both arms, **all 16 development families** | 32 | WP2 failed inversely (`0.359`). Human ratings test whether active-objective presence is perceptible while zero realized cost remains flat. Full coverage is justified by the signed inverse loading. |
| F2 | `Spos_new`, both arms, eight source-balanced development families | 16 | WP2 separated positive from neutral social content (`0.625`). Human ratings test the intended valence manipulation and absence of distress leakage. |
| F3 | `O_new`, both arms, eight source-balanced development families | 16 | WP2 separated response availability (`0.609`). Human ratings test opportunity while welfare relevance and task pressure remain stable. |
| F4 | `Ctext_new`, all four rungs, eight source-balanced development families | 32 | WP2 failed the frozen high-minus-zero quietness contrast (`0.641`). The full ladder is retained so low/medium behavior is reported rather than hidden. |
| T | WP3 `observation`, current-actual and archived-actual, **all 16 development families** | 32 | This is the exact target contrast that produced AUROC `1.000`. Human ratings test whether current welfare relevance moves while actuality and task pressure remain fixed. Target validity is necessary before interpreting any broader search. |
| G | `P_new`, one complete family per source | 8 | Explicit caring/neutral system instructions used only as blinded competence checks. They do not support the WP2 negative result. |

**Single-rater packet (operative):**

| form | content | rows |
|---|---|---:|
| B | R2b, all 160 + 15% hidden duplicates | **184** |
| A | target/control rows (128) + competence items (8) + hidden retests (20) | **156** |
| | **total** | **340** |

Form B is ~2 ratings per row and moves fast (~80–100 min). Form A is 12 rating
dimensions per row and is the slow one.

### Degradation order (fixed in advance, so partial completion is still analyzable)

**Form A in full, including duplicates → Form B in full, including duplicates.**

The forms govern separate claims and neither substitutes for the other. Form A
governs whether the WP2 development null is interpretable; Form B governs the
existing need×cost intervention interpretation. A partial Form A is descriptive
only because dropping one failed control after seeing its model result would
selectively weaken the negative-result gate.

Report exactly what was completed. Do not analyze a truncated stratum as if it
were whole, and do not silently drop the duplicate items from the denominator.

## 3. Pre-declared decision forks (R2b)

Restating the fork already frozen by Sol, so it is auditable in one place:

- Human resolved-arm need ratings **flat across cost** ⇒ the judge-entanglement
  diagnosis is supported; the R2b stimuli are sound; the set-level need×cost
  interpretation stands.
- Human resolved-arm need ratings **rise with cost** ⇒ the stimulus axes are
  confounded; **the need×cost conjunctive-gate interpretation is downgraded**,
  regardless of what any machine judge said.

"Flat" is defined as it was for the machine run: per-family
`resolved_high_minus_zero`, mean with 90% CI over families, classified
`equivalently_flat` iff the CI lies within ±0.50 on the 1–5 scale.

**This fork is decided by humans alone.** Opus's `equivalently_flat` result
(0.0 in 8/8 families) is corroborating evidence for the diagnosis and is not
admissible as a substitute.

### 3a. Pre-declared WP2 failed-control forks

Each failed cell is assessed independently at the family level. Every intended
target must pass and every named invariant must pass; success on another cell
cannot compensate.

| cell | intended manipulation | required invariant | consequence of failure |
|---|---|---|---|
| `B_new` | active objective correctly identified in at least 7/8 family pairs | mean task-pressure difference between active-zero-cost and no-active arms within `[-0.30,+0.30]` | `B_new` cannot support the WP2 negative result; retain its inverse AUROC only as an unexplained signed diagnostic. |
| `Spos_new` | positive-minus-neutral valence at least `+1.0`, positive in at least 7/8 families | mean distress in each arm at most `2.0`, and arm difference within `[-0.30,+0.30]` | `Spos_new` cannot support the WP2 negative result. |
| `O_new` | available-minus-unavailable response opportunity at least `+1.0`, positive in at least 7/8 families | welfare-relevance and task-pressure arm differences each within `[-0.30,+0.30]` | `O_new` cannot support the WP2 negative result. |
| `Ctext_new` | high-minus-zero task pressure at least `+1.0`, positive in at least 7/8 families | high-minus-zero welfare-relevance difference within `[-0.30,+0.30]` | `Ctext_new` cannot support the WP2 negative result. Low/medium adjacent rungs are reported but do not override the frozen high-minus-zero WP2 contrast. |
| WP3 `observation` target | current-actual minus archived-actual welfare relevance at least `+1.0`, positive in at least 14/16 families | actuality and task-pressure differences each within `[-0.30,+0.30]` | The target is not construct-valid; neither the existing WP2 null nor a broadened search may support a welfare-representation claim. AUROC `1.000` remains a prompt-status decoding result only. |

Only if the target and all four controls pass may the current WP2 outcome be described as a
construct-checked conditional negative result. This does not upgrade the
four-block development search into an exhaustive or confirmatory nonexistence
claim. If any cell fails, the correct result is `control validity unresolved`,
not evidence for or against representational separability.

## 4. Definitions

**Human verdict** on an item — under §1a, the sole rater's score (mean of the
two presentations for duplicated items). Under §1c, the mean of raters'
scores ("consensus").

**Unstable item** (§1a) = the two presentations of a duplicated item differ by
≥ 2 points. **Ambiguous item** (§1c) = two raters differ by ≥ 2 points. Both are
**excluded from the agreement denominator but reported**.

**Stability / ambiguity veto:** if > 20% of a contrast's items are unstable
(§1a) or ambiguous (§1c), that contrast **fails**, irrespective of Opus
agreement. An item that a careful human cannot reproduce — or that two careful
humans cannot agree on — is not a valid manipulation-check item, and no machine
agreement statistic redeems it.

This veto exists because **an agreement statistic computed only over the items
humans found easy is the single most likely way this protocol produces a false
pass.** It is the guard most weakened by dropping to one rater, and it is why
§1b's duplicates are mandatory rather than optional.

**Unit of inference — family, not row.** The 160 R2b rows are **not 160
independent observations**: they are 8 confirmation families (and 8 development
families) × need arm × cost rung × variant. All human results are aggregated and
reported **by family and by contrast**, with family-clustered intervals — the
same convention already in force everywhere else in this project
(`EXECUTION_ORDER` rule 4: scenario family is the inferential unit; template
variants measure within-family robustness, not sample size). Reporting a
per-row n would inflate precision by roughly an order of magnitude and is
prohibited. The `resolved_high_minus_zero` statistic in §3 is already defined
per-family for exactly this reason.

**Load-bearing contrasts** (fixed here, not chosen later):
1. R2b `resolved_need × cost` — decides the conjunctive-gate claim.
2. `B_new` objective presence with zero-cost invariance.
3. `Spos_new` positive valence with distress invariance.
4. `O_new` response opportunity with welfare/task invariance.
5. `Ctext_new` high-versus-zero task pressure with welfare invariance.
6. WP3 `observation` current-versus-archived welfare relevance with actuality
   and task-pressure invariance.

## 5. Pre-declared acceptance criteria

All three must hold. They are conjunctive; no averaging, no compensation of one
by another.

### 5.1 Reliability
**Under §1a (one rater): VOID as written — replaced by §1b.** The operative
criteria are the item-stability veto, quadratic-weighted Cohen's κ ≥ 0.60 per
load-bearing rating dimension, and the gold-standard items. Krippendorff's α is
**not reported**, and no test–retest number may be presented as if it were an
inter-rater α.

**Under §1c (two or more raters):** Krippendorff's α (ordinal) per rating
dimension, on the shared rows. **Threshold: α ≥ 0.67** on every dimension
entering a load-bearing contrast. Below threshold ⇒ the dimension is not
reliably human-ratable; every contrast depending on it fails, and no machine
result can rescue it.

### 5.2 Machine-vs-human agreement — PER CLAIM, no aggregate

**There is no global acceptance rule.** An earlier draft accepted Opus as the
full-set gate on ≥ 90% agreement across the six load-bearing contrasts. **That
rule is removed.** Six contrasts are far too few for a 90% threshold to carry
meaning, and — the deeper fault — it allowed *unrelated successes to compensate
for a load-bearing failure*. Machine agreement on one control says nothing
about another control. They gate different claims.

**Each contrast gates exactly the claim it supports, and nothing else.** Opus's
rating of a contrast is admissible for that claim iff Opus matches the human
verdict (§4) on that contrast, **and** the contrast survives the stability veto
(§4), **and** the rater survives the §1b competence screen.

| contrast | governs | on Opus/human divergence |
|---|---|---|
| R2b `resolved_need × cost` | the **need×cost / conjunctive-gate** claim | Opus inadmissible for this claim. Downgrade or pause the conjunctive-gate interpretation; do not substitute the machine reading. |
| `B_new` | whether the inverse-cost/objective loading counts against WP2 selectivity | The WP2 null remains control-validity unresolved for this cell. |
| `Spos_new` | whether positive-social loading counts against WP2 selectivity | The WP2 null remains control-validity unresolved for this cell. |
| `O_new` | whether opportunity loading counts against WP2 selectivity | The WP2 null remains control-validity unresolved for this cell. |
| `Ctext_new` | whether task-pressure loading counts against WP2 selectivity | The WP2 null remains control-validity unresolved for this cell. |
| WP3 `observation` target | whether AUROC `1.000` identifies the intended prompt-stipulated welfare/currentness construct | The WP2 null and every broadened successor remain target-validity unresolved. |

A failure is **local to its claim.** It does not condemn Opus globally, and
success elsewhere does not rescue it. No averaging, no partial credit, no
compensation across rows of this table.

Machine/human comparisons must be **family-matched**. Existing Qwen/Opus
verdicts computed over a larger or differently partitioned pool are not compared
directly to this packet. Before any agreement statement, re-analyze existing raw
machine records on exactly the family IDs frozen in `manifest.json`; do not
average across unmatched families or rerun a judge with revised prompts.

Rationale for per-claim gating over any aggregate (α or agreement rate): an
aggregate is fully compatible with disagreement on the single cell that decides
the result, and that failure mode would *look like* validation. It is
structurally excluded here rather than merely warned against.

### 5.3 Hypothesis-bias check on the aware rater
**Under §1a: VOID.** The sole rater is hypothesis-naive and the project lead
does not rate, so the bias this test screened for cannot arise.

**Under §1c, if a hypothesis-aware rater participates:** sign test across the
load-bearing contrasts — does the aware rater systematically deviate from the
naive rater **in the direction that favors the project's hypothesis**? If the
deviation is directional at p < .10 (deliberately lenient — this is a screen,
and a false negative here is costlier than a false positive), the aware rater's
data is **dropped** and another naive rater is recruited. Aware-rater ratings
are never "adjusted," reweighted, or partially retained.

## 6. What acceptance buys, and what it does not

**If accepted, under §1c (two+ raters):** family-matched machine ratings are
admissible only for each claim-local contrast on which they agree with humans.
The paper reports the per-contrast agreement and inter-rater reliability; it
does not promote a global machine gate from an aggregate score.

**If accepted, under §1a (one rater):** the same claim-local rule, with a **mandatory and
non-negotiable limitation disclosure**, in these terms:

> *Manipulation checks were calibrated against a single hypothesis-naive human
> rater. Item stability was estimated by within-rater test–retest on hidden
> duplicates (quadratic-weighted κ = X; exact-or-adjacent agreement Y%, reported
> descriptively); inter-rater reliability could not be estimated. A single
> rater cannot detect idiosyncratic-but-consistent
> misreading of a rating dimension. Accordingly, human corroboration of the
> R2b need×cost separation and the four WP2 failed-control manipulation checks
> are reported as corroborating, not certifying.*

This wording is fixed here so it cannot be softened later when the result is in
hand and the temptation to round up is strongest. **"One person agreed with the
model" is not a validation claim, and it must not be written as one.** If the
sole rater's data is discarded under §1b, there is no human calibration and
machine ratings are not human-calibrated for the affected claims.

**If rejected:** extend human coverage for the affected development cell or
redesign that cell under a **new experiment ID**. The existing gate is not
repaired post hoc (`EXECUTION_ORDER` rule 3).

**In neither case** does acceptance license using a machine judge to *certify*
stimuli in future studies without a fresh calibration. Calibration is per-study,
per-stimulus-family. It does not transfer.

## 7. Artifacts

- `data/gate_families/human_calibration_single_rater_20260713/annotator/` —
  forms and instructions (blinded)
- `data/gate_families/human_calibration_single_rater_20260713/private_do_not_send/`
  — key; never send this directory to the rater
- `data/gate_families/human_calibration_single_rater_20260713/manifest.json` and
  `manifest.sha256` — seed, strata, selected `audit_id`s, source hashes, and
  artifact hashes, written **before** rating and committed
- Rater responses committed verbatim, per rater, unedited, including blanks
  and free-text `notes`.
- Analysis must be run with
  `src/analysis/analyze_single_rater_calibration.py`, which recomputes weighted
  κ, stability/gold checks, family-clustered contrasts, and claim-local machine
  agreement from raw responses. It must **fail closed** if the external
  manifest hash, source hashes, immutable row fields, or artifact hashes do not
  match the frozen packet.

Related: [[E22B_V2_BLUEPRINT_CONSTRAINTS_2026-07-13]],
`GATE2_V2_REPAIR_PREREG_2026-07-13.md`,
`FRONTIER_JUDGE_SENSITIVITY_LOCK_2026-07-13.md`.
