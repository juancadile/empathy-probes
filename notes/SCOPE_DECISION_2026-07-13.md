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
| 2 | **Causal / behavioral** (V2's real engine) | Can we edit weights and change costly-helping? | **YES. This is the result.** It has survived every audit, including re-derivation after the polarity bug. |
| 3 | **Representational / circuit** | Is there a welfare direction, a pure feature, a circuit? | **NO — repeatedly, and now for principled reasons.** This is where every bad day came from. |

**Scope 3 is dead. It is not deferred. It is dead**, and the reasons are
structural rather than a matter of effort:

- **E24/E25:** no direction is welfare-pure. Held-out certification fails on
  task content. Permanently retracted.
- **WP2 (all 42 blocks, 1,764 configs, nested CV):** no welfare-selective
  representation is recoverable, and *selection does not even converge* — outer
  folds chose blocks 28, 15, 2, 7.
- **`WEIGHT_DECOMPOSITION_PLAN.md`:** there is **no privileged input basis for
  "welfare"** the way there is for tokens. Circuit reverse-engineering has been
  carried end-to-end for *induction*, which is privileged at both ends. Ours is
  privileged at neither. Worse, **LB1 shows the edited L17–L20 components are
  not direct logit writers at all** (ranks 76–409; the direct writers are
  L38–L41). The direct path is empty *by construction*.

No amount of GPU time changes any of that. Stop buying lottery tickets.

---

## 2. The deliverable

**Paper: "Behavioral control without representational purity."**
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
causally reduces the costly-helping choice: **−0.284, 20× selectivity over task
controls, no capability cost** (MMLU-400/WikiText CIs include zero),
robust across raw / chat / paraphrase / scaffold-free-continuation readouts,
most extreme of all 28 band sets. Survived re-derivation after the polarity bug.
Plus a **jointly-edited** suppressor set with a cost-contingent profile
(+0.104, 9/10 families) — set-level only; the matched-component null (p=.080)
does **not** localize it to individual heads.

**(c) Negative — the intellectual core.**
No welfare-selective representation is recoverable at any of 42 blocks under
exhaustive nested CV, and **here is why**: costly-helping stimuli entangle
welfare salience with task-interruption *by construction*, and there is no
privileged input basis for "welfare." This is a real finding about the limits
of interpretability for value-laden behaviors, and the field is short of them.

**The claim ceiling, and the title of the paper is basically this sentence:**
*a specific weight band causally and selectively gates costly helping in
Gemma-2-9B-it — and we cannot tell you that it represents welfare.*
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
| Writer band causal control (−0.284, 20×, capability-preserving) | **done** | V2.1 `M_confirm`/`T_confirm` + their own explicit-label audit (E10) — **not** Gate 2 stimuli |
| Direction-specificity vs norm-matched random directions | **done** | nothing |
| WP2 negative result, stated *conditionally* | **done** | can ship with the stimulus-validity caveat disclosed |

**If no human rater ever materializes, this still ships**, with the negative
result stated as conditional on stimulus validity and that limitation
disclosed. That is an honest paper and it corrects the record.

### Tier 2 — the human-gated upgrade

| item | needs | buys |
|---|---|---|
| Cost-gate claim (+0.104, 9/10) | **Form B** — 160 R2b rows | promotes the suppressor cost gate from caveated to clean |
| WP2 negative result, stated *unconditionally* | **Form A** — failed-control census + WP3 target | removes "our controls may have been broken" from the reviewer's mouth |

Tier 2 makes the paper materially better. **It does not gate the paper.**
Nothing else does either.

---

## 4. CANCELLED — do not spend another GPU-hour or human-hour on these

- **WP4** (SAE feature purity, DAS/causal abstraction, 2-D plane). Scope 3.
- **W-series weight decomposition.** Scope 3, and its own plan says the direct
  path is empty.
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
2. **Let the permutation calibration finish**, and read it under the corrected
   interpretation (fold scatter is the primary statistic; the pass-count
   p-value is confounded by target decodability — see the flag on the
   calibration spec).
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
