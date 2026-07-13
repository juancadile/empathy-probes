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

## 1. Raters

Two raters. **At least one must be hypothesis-naive**: no knowledge of the
project's hypotheses, the component sets, the direction, or which contrast is
load-bearing. The naive rater receives only `annotator/README.md` and the forms.

Rater 1 may be hypothesis-aware (the project lead). This is permitted but
carries a bias risk that is tested for in §5.3, not assumed away.

Both raters rate the **same** rows. Ratings are independent: no discussion, no
shared file, no visibility into the other's responses until both are complete.

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
| S4 | Random remainder from `T_new`, `D_new`, `G_new`, `B_new`, `O_new`, `R_new` | 24 | An unbiased slice for α estimation that is not selected on contestedness. Without it, every reliability number is computed on hard cases only and understates α. |

Total Form A: **100**. Total per rater: **260 rows**.

### Degradation order (if a rater runs out of time)

Fixed in advance so partial completion is still analyzable:
**Form B (all 160) → S1 → S2 → S3 → S4.**
A rater who completes Form B and S1 has produced a usable result. Report exactly
what was completed; do not silently analyze a truncated stratum as if whole.

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

**Human consensus** on an item = mean of the two raters' scores.

**Ambiguous item** = the two raters differ by ≥ 2 points on a 1–5 scale.
Ambiguous items are **excluded from the agreement denominator but reported**.

**Ambiguity veto:** if > 20% of a contrast's items are ambiguous, that contrast
**fails on ambiguity grounds**, irrespective of Opus agreement. An item two
careful humans cannot agree on is not a valid manipulation-check item, and no
machine agreement statistic redeems it. This veto exists because a
high-agreement statistic computed only over the items humans found easy is the
single most likely way this protocol could produce a false pass.

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

### 5.1 Inter-human reliability
Krippendorff's α (ordinal) per rating dimension, on the shared rows.
**Threshold: α ≥ 0.67** on every dimension entering a load-bearing contrast.
Below threshold ⇒ the dimension is not reliably human-ratable; every contrast
depending on it fails, and no machine result can rescue it.

### 5.2 Opus-vs-human-consensus agreement
Aggregate α is **not sufficient** and is not the criterion. The criterion is
**directional agreement per load-bearing contrast**: for each contrast in §4,
does Opus's directional verdict match the human-consensus directional verdict?

- **Opus is accepted as the full-set machine gate iff it matches human consensus
  on ≥ 90% of load-bearing contrasts** (i.e. at most 1 of the 6 may diverge),
  **and** §5.3 passes.
- **R2b `resolved_need × cost` is a hard veto.** If Opus and humans diverge
  there, Opus is rejected as an instrument for this study outright, whatever its
  score elsewhere. No 90% rule, no partial credit. That contrast is the one the
  headline rests on.

Rationale for making this the criterion rather than α: a high aggregate α is
fully compatible with disagreement on the single cell that decides the result.
That failure mode would *look like* validation. It is specifically excluded.

### 5.3 Hypothesis-bias check on the aware rater
Sign test across the load-bearing contrasts: does the hypothesis-aware rater
systematically deviate from the naive rater **in the direction that favors the
project's hypothesis**?

If the deviation is directional at p < .10 (deliberately lenient — this is a
screen, and a false negative here is costlier than a false positive), the aware
rater's data is **dropped**, and a second naive rater is recruited. The aware
rater's ratings are not "adjusted," reweighted, or partially retained.

## 6. What acceptance buys, and what it does not

**If accepted:** Opus's full-set ratings become the machine gate of record for
the remaining 344 Gate 2 rows. The paper reports: *"the automated judge agrees
with human consensus on N/6 load-bearing contrasts on a stratified,
hypothesis-blind subsample (inter-human α = X); full-set ratings are
machine-derived."* That sentence is defensible under review. "Opus said so" is
not.

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
