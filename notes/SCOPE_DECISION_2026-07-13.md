# Scope Decision — what ships, what dies

**Status:** DECISION. Supersedes the open-ended programs in `ROADMAP.md`
(WP4, circuit tracing, cross-model, E22b) for the purposes of the V2
deliverable. Those items are not "later" — they are **cancelled** unless
V2 ships first and someone chooses to reopen them.

**Author's note:** this document exists because the project had three
different scopes running simultaneously and only ever declared one of them.
Every painful day in this project has been the failure of the scope that was
never going to work, while the scope that was actually delivering went
unnamed.

---

## 0. The organizing purpose: a correction obligation

**V1 is on arXiv and its headline claim is not supported.**

V1 (`Detecting and Steering LLMs' Empathy in Action`) claims empathy-in-action
is *"a linear direction in LLM activation space"* with **AUROC 0.996–1.00**
detection across three model families.

**E04 shows that result is lexically saturated:** a shuffled-token control on
the same dataset reaches **AUROC 0.90–0.99**. The probe was reading vocabulary,
not decisions. The headline detection claim does not survive its own control.

This is not an academic preference about what to publish next. **There is a
paper on a public preprint server making a claim its authors now know to be
confounded.** V2's first job is to replace it under the same arXiv entry so
that readers land on the corrected version.

Everything below is subordinate to that. Any activity that does not move V2
toward arXiv is, for now, out of scope.

---

## 1. The three scopes, named at last

| # | scope | question | verdict |
|---|---|---|---|
| 1 | **Probing** (V1) | Can we decode empathy-in-action? | **Answered, and the answer is "not the way we said."** Decodable, but lexically saturated (E04). Ships as a *correction*. |
| 2 | **Causal / choice-assay** (V2's real engine) | Can parameter edits change matched costly-helping choices? | **YES. This is the result.** It has survived re-derivation and held-out task controls. It is not a general enacted-empathy result. |
| 3 | **Representational / circuit** | Can the tested methods recover a welfare-pure representation or complete circuit? | **NOT FOUND.** The tested linear/low-dimensional searches do not yield a stable, certified-pure representation; full-circuit recovery is outside V2's credible scope. |

**Scope 3 is cancelled for V2. It is not a shipping dependency.** This is a
scope decision, not a claim that further mechanistic work is scientifically
impossible:

- **E24/E25:** the tested direction is not welfare-pure. Held-out certification
  fails on task content. The purity claim is permanently retracted.
- **WP2 (all 42 blocks, 1,764 configs, nested CV):** no stable candidate is
  selected within the tested class, and *selection does not even converge* —
  outer folds chose blocks 28, 15, 2, 7. This result remains conditional on
  target/control construct validity.
- **`WEIGHT_DECOMPOSITION_PLAN.md`:** there is **no privileged input basis for
  "welfare"** the way there is for tokens. Circuit reverse-engineering has been
  carried end-to-end for *induction*, which is privileged at both ends. Ours
  has a privileged forced-choice output contrast but no privileged welfare
  input basis. Worse, **LB1 shows the edited L17–L20 components are
  weak direct logit writers** (ranks 76–409; the leading direct writers are
  L38–L41). A useful account would require unresolved indirect paths.

Additional open-ended GPU work does not advance the bounded correction. Ship
before deciding whether any of these questions deserve a separate project.

---

## 2. The deliverable

**Paper: "Causal control of costly-helping choices without representational
purity."**
V2, replacing V1 at the same arXiv entry.

Three contributions, in order of how much they are worth:

**(a) Methodological — the most valuable thing here.**
- Matched-lexicon minimal pairs (both branches share every emotion word; only
  the enacted decision clause differs).
- **Decodability is cheap:** shuffled-token controls reach 0.90–0.99. Probe
  AUROC on unmatched contrastive sets is not evidence of a behavioral
  representation. This directly corrects V1.
- **The AUROC polarity bug.** One-sided quietness gates read *inverse*
  separation as silence: `T 0.05` is 0.95 two-sided separability with inverted
  polarity. This is a gift to the field — one-sided gates are common, and this
  bug invalidated an entire showcase before it was caught. Publish it loudly.

**(b) Positive — narrow, replicated, honest.**
Editing a mid-late MLP band (L19 dominant, L20 the strongest complement)
causally reduces the matched costly-helping choice: **−0.284, 20× selectivity
over task controls**, with no detected degradation for the individual writer
set on sampled MMLU/WikiText benchmarks,
robust across raw / chat / paraphrase / scaffold-free-continuation readouts,
most extreme of all 28 band sets. Survived re-derivation after the polarity bug.
Plus a **jointly-edited** suppressor set with a cost-contingent profile
(+0.104, 9/10 families) — set-level only; the matched-component null (p=.080)
does **not** localize it to individual heads.

**(c) Negative — the intellectual core.**
Across all 42 blocks, the tested linear/low-dimensional search does not recover
a stable welfare-selective candidate: nested selection chooses blocks
28/15/2/7 and fails nuisance quietness out of fold. The result is conditional
on human validation because incoherent targets or confounded controls can
produce the same instability. It is evidence about the limits of these methods
and stimuli, not evidence that the model contains no welfare representation.

**The claim ceiling, and the title of the paper is basically this sentence:**
*activation-selected parameter edits to a mid-late MLP band causally and
selectively alter matched costly-helping choices in Gemma-2-9B-it — and the
tested methods do not establish that this band represents welfare.*
Not "we found the empathy circuit." Never that.

---

## 3. Two tiers — and Tier 1 does not need humans

This is the most important operational fact in this document.

### Tier 1 — SHIPPABLE NOW. No human audit required.

Discharges the arXiv correction obligation on its own.

| item | status | depends on |
|---|---|---|
| V1 correction (lexical saturation, E04) | **done** | nothing |
| Matched-lexicon methodology (V2.1 suite) | **done** | nothing |
| AUROC polarity bug (E24) | **done** | nothing |
| Writer-band choice-assay control (−0.284, 20×; no detected degradation for the individual set on sampled MMLU/WikiText) | **done** | V2.1 `M_confirm`/`T_confirm` + their own explicit-label audit (E10) — **not** Gate 2 stimuli |
| Direction-specificity vs norm-matched random directions | **done** | nothing |
| WP2 negative result, stated *conditionally* | **done** | can ship with the stimulus-validity caveat disclosed |

**If no human rater ever materializes, this still ships**, with the negative
result stated as conditional on stimulus validity and that limitation
disclosed. That is an honest paper and it corrects the record.

### Tier 2 — the human-gated upgrade

| item | needs | buys |
|---|---|---|
| Cost-gate claim (+0.104, 9/10) | **Form B** — 160 R2b rows | tests whether the set-level cost-gate interpretation survives construct review; one rater corroborates but does not certify |
| WP2 conditional negative result | **Form A** — failed-control census + WP3 target | tests the target/control validity caveat; one rater can strengthen or reject specific contrasts but cannot make the result unconditional |

Tier 2 makes the paper materially better. **It does not gate the correction.**
Nothing else does either.

---

## 4. CANCELLED — do not spend another GPU-hour or human-hour on these

- **WP4** (SAE feature purity, DAS/causal abstraction, 2-D plane). Scope 3.
- **W-series weight decomposition.** Outside the bounded V2 deliverable; the
  direct path is weak and the indirect path would reopen an open-ended program.
- **Circuit tracing, path patching, faithfulness/completeness.** Scope 3.
- **E22b.2 / E22b.3 moral-axis redesign.** A fresh construct-validity assay for
  a claim the paper no longer makes. (The E22 result is retracted as stale and
  confounded; the paper says so and stops there.)
- **Cross-model replication.** E17 already failed at the confirmatory level. The
  paper reports that as a scope limit, which is the honest thing to do anyway.
- **Individual-suppressor-head certification.** Already correctly declined.

Any of these may be reopened **after V2 is on arXiv**, by choice, as new work.
None of them is a prerequisite for anything that ships.

---

## 5. Immediate order of work

1. **Recruit the R2b raters.** Still the only external dependency, and it now
   gates an *upgrade* rather than the paper. This makes it lower-stakes, not
   higher — do it, but do not wait on it.
2. **Permutation calibration complete.** Observed fold-site dispersion is no
   tighter than the paired-label null (14.33 blocks vs null median 12.08,
   lower-tail p=.545); the selector does not converge on a common depth. The
   pass count is descriptive only because target shuffling destroys
   decodability.
3. **Start writing.** Tier 1 is complete. The paper does not need another
   experiment to begin.
4. **Fold in Tier 2 if and when the humans deliver.**

**Rule for the remainder of V2:** if a proposed task does not move the paper
toward arXiv, the answer is no. If it is scope 3, the answer is no even if it
is interesting. *Especially* if it is interesting.

---

## 6. On what actually happened here

The exciting version of this project — *we found the empathy circuit* — died.
It was killed by this project's own controls: the shuffled-token battery, the
polarity audit, the held-out certification, the nested CV, the manipulation
checks. Not by a reviewer, and not after publication.

That is the machinery working exactly as designed. A project that kills its own
best story before it reaches print is doing science; a project that ships the
story and gets corrected later is doing something else. V1 is already the second
kind, through no fault of anyone — it just didn't have the controls yet. V2's
whole purpose is to be the first kind, and to fix V1 on the way past.

Related: [[WEIGHT_DECOMPOSITION_PLAN]], `EXECUTION_ORDER_2026-07-13.md`,
`HUMAN_CALIBRATION_PROTOCOL_2026-07-13.md`, `WP2_BROADENED_DEV_RESULT_2026-07-13.md`.
