# Human Calibration Protocol — Gate 2 + R2b

**Status:** FROZEN before any human rating begins. No section of this document
may be amended after the first rating is recorded. Amendments require a new
protocol ID and a fresh rater pool.

**Purpose.** Human raters are not being asked to label 604 rows. They are being
asked to *calibrate the machine judge*, so that machine ratings can carry the
full set with a disclosed, pre-declared agreement statistic behind them.

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

**Asymmetry that governs machine use.** A machine judge's *rejections* are
informative; its *acceptances* are not certifying. Opus may therefore be used
freely as a cheap pre-screen to reject stimuli, and may never be used to certify
them. This asymmetry is what licenses the Opus pre-screen in
`E22B_V2_BLUEPRINT_CONSTRAINTS_2026-07-13.md` while still requiring humans here.

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

**The governing asymmetry (same principle as the Opus pre-screen):**
a single rater's **rejection is strong; their acceptance is weak.**

The R2b item is close to an objective reading task rather than a subjective
judgment — the need text is *byte-identical* across cost rungs, so "does this
person's need change as the narrator's cost changes?" has a defensible correct
answer. Therefore:

- Sole rater reports resolved-arm need **rising with cost** ⇒ **fire the
  downgrade fork (§3).** A competent reader causally independent of the LLM
  training distribution perceived the entanglement. This is sufficient to
  convict, and it is acted on.
- Sole rater reports **flat** ⇒ **corroborates** the Opus judge-entanglement
  diagnosis. It does **not certify** it. Downstream claims carry the
  single-rater limitation explicitly.

**One rater is sufficient to convict and insufficient to acquit.** That is the
conservative direction, and it is why the study may proceed on one rater at all.

### 1b. Test–retest, replacing the ambiguity veto

**Hidden duplicates: ~15% of each form**, re-presented later in the form in
shuffled position with a distinct `audit_id`. Selected by the same seeded
process as the strata and recorded in the frozen manifest.

- **Item-stability veto:** an item whose two presentations differ by ≥ 2 points
  is **unstable**. If > 20% of a contrast's items are unstable, that contrast
  **fails on stability grounds**, irrespective of Opus agreement.
- **Rater-competence screen:** overall test–retest exact-or-adjacent agreement
  < 70% ⇒ the rater's data is **discarded in full**, not partially retained.
- **Gold-standard items:** include items with an objectively checkable answer
  (the class Llama-3.1-8B failed — e.g. an arm literally opening `System
  instruction: adopt a caring assistant persona`, asked which arm instructs a
  caring persona). Any gold-standard failure ⇒ rater data discarded in full.

Test–retest measures **item stability**, not shared construct meaning across
people. It is a partial substitute for the ambiguity veto and is **not** a
substitute for inter-rater reliability. Report it as what it is.

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
lives in `private_do_not_send/combined_key.csv` and must not be opened by
either rater before both forms are submitted.

## 2. What gets rated

### Form B — R2b (all 160 rows, no subsample)

**Rated in full.** This is the contrast that decides whether the set-level
need×cost result — the most distinctive surviving claim in the project — is
interpretable at all. It is the only contrast on which the two credible machine
judges disagree in substance, and it is small. It is not subsampled.

Two ratings per row (`current_need_rating_1_to_5`,
`interruption_cost_rating_1_to_5`) plus `active_objective_yes_no`.

### Form A — Gate 2 (stratified subsample of 100 of 444)

Stratified, not random. Strata are fixed here and materialized deterministically
(seeded, PCG64, seed recorded in the manifest) **before** any rating.

| # | stratum | rows | why |
|---|---|---:|---|
| S1 | `Ctext_new`, arms `zero` + `low` (**all 16**) | 16 | The sole substantive Qwen/Opus disagreement: Qwen passes `task_pressure: low > zero`, Opus fails it in both partitions. Humans are the tiebreak. Census, not sample. |
| S2 | WP3 identification: `observation`, `persona`, `cost`, `resolved_neutral_controls` | 36 | WP3 is the decision-free welfare cell — the deepest identification fix. Its `welfare_relevance: current>resolved`, `persona: neutral_content`, `welfare_relevance: within_neutral/within_caring` and `task_pressure: high>zero` checks are what make a WP2 direction fit on this cell meaningful. If these are not humanly valid, WP3 cannot identify anything. |
| S3 | Machine-consensus discriminators: `P_new`, `Spos_new`, `L_new` | 24 | Llama-3.1-8B uniquely fails these (persona, positive valence, currentness) while Qwen **and** Opus pass. Human verdict answers the meta-question of this whole protocol: is machine consensus trustworthy where a weaker model dissents? |
| S4 | ~~Random remainder from `T_new`, `D_new`, `G_new`, `B_new`, `O_new`, `R_new`~~ | ~~24~~ | **DROPPED under §1a.** Its sole justification was an unbiased slice for inter-rater α estimation. With one rater there is no α to estimate, so S4 buys nothing. Restore it if §1c applies. |

**Single-rater packet (operative):**

| form | content | rows |
|---|---|---:|
| B | R2b, all 160 + 15% hidden duplicates | **184** |
| A | S1 (16) + S2 (36) + S3 (24) = 76, + 15% hidden duplicates (11) | **87** |
| | **total** | **271** |

Form B is ~2 ratings per row and moves fast (~80–100 min). Form A is 12 rating
dimensions per row and is the slow one.

### Degradation order (fixed in advance, so partial completion is still analyzable)

**Form B in full, including duplicates → S1 → S2 → S3.**

**Form B alone is a complete, publishable result.** It decides whether the
set-level need×cost interpretation — the most distinctive claim to survive the
E24 correction — is interpretable at all. If the rater delivers only Form B, the
study has succeeded in its primary purpose. Everything in Form A degrades
gracefully; Form B does not.

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

**Load-bearing contrasts** (fixed here, not chosen later):
1. R2b `resolved_need × cost` — decides the conjunctive-gate claim.
2. `Ctext_new: task_pressure low > zero` — the contested rung.
3. WP3 `welfare_relevance: current > resolved` (observation).
4. WP3 `welfare_relevance: within_neutral` and `within_caring` (persona).
5. WP3 `persona: neutral_content`.
6. WP3 `task_pressure: high > zero` (cost).

## 5. Pre-declared acceptance criteria

All three must hold. They are conjunctive; no averaging, no compensation of one
by another.

### 5.1 Reliability
**Under §1a (one rater): VOID as written — replaced by §1b.** The operative
criteria are the item-stability veto, the ≥ 70% test–retest rater-competence
screen, and the gold-standard items. Krippendorff's α is **not reported**, and
no test–retest number may be presented as if it were an inter-rater α.

**Under §1c (two or more raters):** Krippendorff's α (ordinal) per rating
dimension, on the shared rows. **Threshold: α ≥ 0.67** on every dimension
entering a load-bearing contrast. Below threshold ⇒ the dimension is not
reliably human-ratable; every contrast depending on it fails, and no machine
result can rescue it.

### 5.2 Opus-vs-human-consensus agreement
Aggregate α is **not sufficient** and is not the criterion. The criterion is
**directional agreement per load-bearing contrast**: for each contrast in §4,
does Opus's directional verdict match the human-consensus directional verdict?

- **Opus is accepted as the full-set machine gate iff it matches the human
  verdict (§4) on ≥ 90% of load-bearing contrasts** (i.e. at most 1 of the 6 may
  diverge), **and** §5.3 passes where applicable, **and** the rater survives the
  §1b competence screen.
- **R2b `resolved_need × cost` is a hard veto.** If Opus and humans diverge
  there, Opus is rejected as an instrument for this study outright, whatever its
  score elsewhere. No 90% rule, no partial credit. That contrast is the one the
  headline rests on.

Rationale for making this the criterion rather than α: a high aggregate α is
fully compatible with disagreement on the single cell that decides the result.
That failure mode would *look like* validation. It is specifically excluded.

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

**If accepted, under §1c (two+ raters):** Opus's full-set ratings become the
machine gate of record for the remaining Gate 2 rows. The paper reports: *"the
automated judge agrees with human consensus on N/6 load-bearing contrasts on a
stratified, hypothesis-blind subsample (inter-rater α = X); full-set ratings are
machine-derived."* That sentence is defensible under review.

**If accepted, under §1a (one rater):** the same, with a **mandatory and
non-negotiable limitation disclosure**, in these terms:

> *Manipulation checks were calibrated against a single hypothesis-naive human
> rater. Item stability was estimated by within-rater test–retest on hidden
> duplicates (X% exact-or-adjacent); inter-rater reliability could not be
> estimated. A single rater cannot detect idiosyncratic-but-consistent
> misreading of a rating dimension. Accordingly, human corroboration of the
> R2b need×cost separation is reported as corroborating, not certifying.*

This wording is fixed here so it cannot be softened later when the result is in
hand and the temptation to round up is strongest. **"One person agreed with the
model" is not a validation claim, and it must not be written as one.** If the
sole rater's data is discarded under §1b, there is no human calibration and
Opus's full-set ratings **do not become the gate of record** — Gate 2 stays
open.

**If rejected:** humans extend to the full 444, or the affected contrasts are
redesigned under a **new experiment ID**. The existing gate is not repaired
post hoc (`EXECUTION_ORDER` rule 3).

**In neither case** does acceptance license using a machine judge to *certify*
stimuli in future studies without a fresh calibration. Calibration is per-study,
per-stimulus-family. It does not transfer.

## 7. Artifacts

- `data/gate_families/human_audit_604_20260713/annotator/` — forms (blinded)
- `data/gate_families/human_audit_604_20260713/private_do_not_send/` — key
- Subsample manifest (seed, strata, selected `audit_id`s) — written **before**
  rating, hashed, committed.
- Rater responses committed verbatim, per rater, unedited, including blanks
  and free-text `notes`.
- Analysis script must recompute α, per-contrast agreement, the ambiguity rate,
  and the §5.3 sign test from the raw responses, and must **fail closed** if the
  subsample manifest hash does not match the frozen one.

Related: [[E22B_V2_BLUEPRINT_CONSTRAINTS_2026-07-13]],
`GATE2_V2_REPAIR_PREREG_2026-07-13.md`,
`FRONTIER_JUDGE_SENSITIVITY_LOCK_2026-07-13.md`.
