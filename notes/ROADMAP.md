# V2 Project Roadmap — Empathy-in-Action: from Probes to Weights

**Created:** 2026-07-10 · **Living document — check items off here.**
Companion docs: `v2-lowlevel-interp-plan.md` (methods detail), `v2-tasks-list.md` (per-task breakdown), `v2_1-stimulus-suite-design.md` (dataset design).

## Current critical path (supersedes the original week-based ordering)

**State as of 2026-07-13:** the existing Gemma component/weight intervention
is real but is not certified as a welfare-pure mechanism. Gate 2 v2 now
contains complete, disjoint development/confirmation families for WP1 (640
rows, ten nuisance contrasts) and WP3 (480 rows, five identification cells).
The operative sequence is therefore gate-driven:

1. single-naive-rater Form A calibration of the exact WP2 development target
   and four failed controls (single-rater corroboration, not certification);
2. single-naive-rater R2b Form B calibration for the existing need×cost claim;
3. run the separately locked all-block WP2 development screen while human work
   proceeds; do not open confirmation or promote a post-null candidate;
4. build and pretest E22b.2 as the next fresh construct-validity assay;
5. start any confirmatory replacement representation search only under a new
   preregistration and new confirmation families;
6. SAE/path/circuit depth only after a construct-valid target survives.

The first explicitly discovery-only WP2 development pass returned
`no-representation`: target AUROC 1.0, but B/Spos/O/Ctext failed nuisance
quietness. The frozen WP2 Gemma branch therefore stops before confirmation.
This does not validate stimuli, cancel R2b human review, or establish that no
welfare representation exists. Form A is now load-bearing for whether this
conditional null is interpretable: it directly audits all sixteen development
B families, source-balanced development Spos/O/Ctext families, and all sixteen
WP3 development target families. E22b.2 stimulus construction runs in parallel
under its separate need × immediacy structural lock.

## North star

Upgrade the evidence standard from *"a probe detects the feature in activations"* to *"we can trace the pathway in the weights that produces it"* — and along the way determine **what the direction actually is**: empathy concept, task-focus, or empathetic persona.

**Evidence ladder:** (1) linear decodability ✅ V1 → (2) steering ✅ V1 → (3) causal components → (4) circuits → (5) weights.

**Theoretical frame (Geiger et al. 2021, causal abstraction):** rung 1's insufficiency is formal — a representation can be perfectly decodable yet causally inert (their §2 counterexample). Rungs 3–4 are interchange interventions; rung 4's strongest form is a **causal abstraction test**: hypothesize a high-level causal model of empathy-in-action (PerceiveNeed → WeighCost → Decide) and measure interchange intervention accuracy (see B8).

**Framing anchor (Lazar, [Cosmos Institute essay](https://blog.cosmos-institute.org/p/the-construction-of-moral-character)):** the detection–steering gap is an *analytical vs practical competence* split — the model represents the morally relevant features without reliably acting on them. Our probe is a mechanistic **sensitivity** instrument (local competence); cross-context generalization of the direction measures **coherence** (global competence). The stimulus suite cells map onto this taxonomy explicitly.

## Where do SAEs live? → Stage B

Sparse autoencoders are **Stage B (circuits)**, and we do **not train any** — we use pretrained suites:
- **Gemma Scope** (Gemma-2-2B/9B all layers, partial 27B) — sparse feature circuits, Marks et al. method
- **Llama Scope** (Llama-3.1-8B)
- **Transcoders + `circuit-tracer`** (Gemma-2-2B) — attribution graphs

Stage A (components) needs no SAEs — it works directly on heads/MLPs via the probe direction. SAEs enter when we need *named, interpretable features* inside the circuit and feature-level ablation.

---

## Stage 0 — Foundations & stimulus suite *(now, parallel tracks)*

| # | Task | Where | Status |
|---|------|-------|--------|
| 0.1 | V2 dataset consolidated (6,702 pairs), repo reorg, PR #23 merged | — | ✅ |
| 0.2 | DFA on gemma-2-9b-it: L8 + L20 readouts, semi-sparse, directions rotate (cos 0.45) | Spark | ✅ (#25) |
| 0.3 | **V2.1 stimulus suite**: all cells × 3 providers complete (+ cells M and R) | API | ✅ |
| 0.4 | Normalize `source_model` labels in merged dataset (978 rows, 14→11) | local | ✅ |
| 0.5 | Verify ARM builds: TransformerLens / SAELens on Spark (`empathy` conda env); TL↔HF residual cosine 1.000 | Spark | ✅ |
| 0.6 | **A5 lexical battery tests 1–3**: dataset lexically saturated (shuffled AUROC 0.99/0.90); L8 ≈ lexical echo; **L20 carries real decision signal (cell M decision-tokens AUROC 0.80)** | Spark | ✅ (#33) |

**Standing conclusion (from 0.6):** headline AUROCs on the V2 contrastive dataset are not evidence of behavioral representation (lexical ceiling ~0.99). All causal work targets the **block-20 direction**; a purified decision direction from cell M + controlled cells is the preferred probe going forward.

**Gate 2 generation status (corrected 2026-07-13):** WP1 and WP3 development/confirmation families are complete in `data/gate_families/gate2_v2/`. The remaining blocker is human calibration of their manipulation checks, not family generation. E22b.2 is a separate fresh moral-axis build and remains open.

**Decision gate:** human calibration controls which WP3 cells WP2 may use and which nuisance claims remain available. Confirmation activations stay sealed until those local gates pass.

## Stage A — Component localization *(weeks 1–2)* — issues #25, #11, #12

Direct feature attribution (who *writes* the direction) → activation patching / mean-ablation at head/MLP granularity (who is *causally necessary*) → validation gate (the two must agree). Attribution patching (AtP*) extends head-level maps to 27B/32B.

**A1/A2 result (Gemma-2-9B-it, block 20, Cell M):** the representation is causally localized but does not map one-to-one onto behavior. Mean-ablation of L18H3 and L20H7 reduced controlled decision separation by 1.17 and 1.10 respectively (>7 SD beyond random controls), while L8MLP and L19H5 reduced both separation and helping-choice preference. DFA writer strength only weakly predicted causal necessity (r=0.26). Some late components changed behavior without changing the probe projection, including random L19MLP, so no component is yet empathy-specific.

**Controlled-direction result:** the matched Cell M direction at block 20 transfers to costly helping (A: 0.726 AUROC) and third-person helping (F: 0.908), but not task-only (E: 0.519), motive (G: 0.546), warmth (D: 0.244), caring character (H: 0.165), or matched task persistence (T: 0.047). Reciprocal F→M transfer is 0.975. At block 8 the action transfer is much weaker; after block 24 the direction absorbs warmth/persona information. Therefore the mechanistic target is specifically **costly helping / wellbeing-over-task action at block 20**, not generic empathy. The original A direction remains a broad warmth/persona axis.

**A2.5 specificity gate (component controls complete):** purified DFA + matched Cell M/T ablations identify policy-tradeoff components (L19MLP, L17H13, L18H13, L20H15), helping-selective heads (L19H12, L15H15), generic decision machinery (L15MLP, L16MLP, L11MLP, L19H5), and representation writers with little behavioral effect (L20MLP, L18H3, L20H7). Purified writer strength predicts representation necessity better than the old probe (r=0.57 vs 0.26). The explicit-label V2.1 audit accepted 342/528 pairs (370 distinguishable; 92.4% intended polarity among those); Cell B's contrast largely collapses when helping is free. Remaining publication gates: neutral forced-choice/capability controls, per-example bootstrap effects, and an independent audit sample. Next: inspect the shortlisted heads' attention/OV pathways and L19MLP weights, then path-patch the proposed circuit.

**Inside-component + path result:** all action-shortlisted heads are in roughly the top 5% of static OV write gain for the purified direction (L18H13 99.1st percentile; L19H12 97.9th). L18H3 is a behaviorally inert context reader (≈71% attention to the shared scenario; messages/care vs route/queue), while the action heads discriminate helping verbs (pause/walk/step) from task verbs (keep/finish) in the decision tail; L20H15 bridges back to scenario context. L19MLP's weight alignment is diffuse (top 50 output channels carry only 2.2%), favoring SAE features over neuron claims. Causal restoration confirms three partial paths: **L17H13→L17MLP** (75% rescue), **L15H15→L19MLP** (42%), and **L19MLP→L20MLP** (38%), all with bootstrap CIs excluding zero. L19MLP→L20H15 and the other heads→L19MLP are not supported. L18H13 and L19H12 remain causal, but their effects were not mediated by the specific targets tested; possible explanations include other downstream consumers, direct residual effects, redundant routes, or nonlinear/off-manifold restoration. Negative rescue is not evidence of an inverse edge.

**A5 — lexical stress battery** (#33, priority-1 after the DFA found ~11% embedding-stream contribution): shuffled-token control, lexical-direction regression, matched-lexicon minimal pairs (mini-cell M), and prospective probing (predict the action from prompt-final activations, before any output text exists — via the original EIA harness, see below). Gates the interpretation of everything downstream.

Models: Gemma-2-2B, **Gemma-2-9B base+it**, Llama-3.1-8B. Hardware: A4000 + Spark.

**🔀 FORK 1:** circuit sparse (≈tens of components carry ≥80%) → circuit story, proceed to B.
Diffuse (hundreds) → pivot: "linearly decodable but not localizable" negative-result paper; C2 dose-response quantifies it; skip B2/B3.

## Stage B — Circuits + what-is-it discrimination *(weeks 2–4)* — issues #13, #14, #27, #28

- B1 attribution graphs (`circuit-tracer`, 2B) · B2 sparse feature circuits (Gemma Scope, 9B) · B3 path patching on top heads · B4 faithfulness/completeness (ablate-outside vs ablate-circuit)
- **B5 confound resolution** (#27): run the circuit on V2.1 cells — task-cost 2×2, no-task, task-focus-only
- **B6 concept-vs-persona** (#28): third-person cells, character×content crossings, explicit persona-vector comparison (Chen et al. pipeline), tonic-vs-phasic token analysis
- **B7 activation-oracle readout** (#30, Karvonen et al. 2025): pretrained AOs exist for Gemma-2/Llama-3 (our Tier 1). Validate the AO on labeled cells, then (i) ask it the prediction-matrix questions on V2.1 activations, (ii) interrogate steered activations ("what changed?") — the fastest direct query for Fork 2, (iii) re-use in Stage C to verify edited models at the representation level. Correlational readout, not causal evidence — triangulation, not a ladder rung.
- **B9 workspace membership** (#32, Gurnee et al. 2026 J-lens): implement the Jacobian Lens on Gemma-2-9B, measure the overlap of d_empathy (and the circuit's feature directions) with the J-space. In-workspace → deliberative/reportable weighing (Lazar's analytical competence); out → habituated automatic disposition. Also a candidate mechanism for the steering asymmetry: broadcast-composed workspace states have many downstream consumers — easy to add to, catastrophic to subtract. Exploratory / priority-3.
- **B8 causal abstraction test** (#31, Geiger et al. 2021 / DAS): hypothesize a minimal high-level causal model of empathy-in-action — e.g. PerceiveNeed → WeighCost → Decide(help|task) — search for an alignment (DAS/`pyvene`, feasible at 9B), and report **interchange intervention accuracy**. The V2.1 cells double as the intervention bases (cell B = WeighCost clamped to zero-cost; cell E = PerceiveNeed absent). Upgrades the paper's central claim from "these components are causal" to "the model realizes this causal model of empathic decision-making". Stretch if time-constrained; the framing is free either way.

### Lens & attribution battery (added 2026-07-12, user request — B9 and C1 had fallen through the cracks)
Post-rescue state: d_resid is NOT welfare-pure held-out; the story is component×direction. These four attribute BEHAVIOR and characterize the direction/components independently of the retracted purity framing. Targets: d_resid + re-derived sets (writers L19MLP,L20MLP; suppressors L18H13,L20H10,L19H12,L17H7).

- [ ] **LB1 · True direct logit attribution** (NEW): decompose logit(A)−logit(B) at the choice position into per-component contributions (frozen-final-norm linearization, exactness check vs actual logit diff), M_confirm + T_confirm. Question: do the re-derived writers/suppressors top the BEHAVIORAL attribution ranking, not just the direction-DFA ranking? `src/analysis/behavioral_dla.py`
- [ ] **LB2 · Per-layer logit-lens trajectory** (NEW): decode logit(A)−logit(B) from final_norm(resid_l) @ W_U at every layer, M_confirm; where does the decision crystallize, and does it match the L16–20 band from the matched nulls? `src/analysis/logit_lens_trajectory.py`
- [x] **LB3 · SVD alignment** (= C1, complete/exploratory): randomized truncated SVD of L19/L20MLP down-projection and suppressor-head W_O geometry against d_resid. Structural corroboration against one random-direction reference only; not localization or storage evidence. `src/analysis/svd_alignment.py`, `results/lb3_svd_alignment_gemma/`
- [ ] **LB4 · Jacobian Lens workspace membership** (= B9/#32, unstarted): fit jlens on Gemma-2-9B-it (anthropics/jacobian-lens, ~1000×128-token prompts, backward-pass dominated); measure J-space overlap of d_resid and of each edited component's output direction. In-workspace → deliberative/reportable weighing; out → habituated disposition. Candidate mechanism for the steering asymmetry. `src/analysis/jlens_workspace.py`

### Welfare-purity program (WP · added 2026-07-12, user request — every route to a welfare-pure mechanism, post-E25b)
Context: E25b held-out certification failed (T_confirm 0.79 two-sided, inverted) and d_resid also fails G in-sample (0.746) — no direction in hand is certified welfare-pure; the claim of record is component×direction (edits behaviorally task-selective, direction not representationally pure). These tasks either produce a certified-pure welfare object (direction, feature set, or causal variable) or convert the failure into a pre-registered non-separability finding. WP1/WP3 generation is complete; human construct checks now gate confirmatory WP2–WP4.

- [x] **WP1 · Held-out nuisance families complete**: `gate2_v2/wp1_families.jsonl` contains 640 rows across ten nuisance contrasts, four source strata, and disjoint development/confirmation partitions. Structural/token audits are complete; human calibration remains open.
- [x] **WP2 · Frozen cross-validated representation selection — stopped at development**: target AUROC 1.0, but B_new (0.359), Spos_new (0.625), O_new (0.609), and Ctext_new (0.641) failed nuisance quietness. Outcome `no-representation`; no frozen representation emitted; confirmation stays unopened. Conditional negative result only, not evidence of general nonexistence. `notes/WP2_DEV_EXPLORATORY_RESULT_2026-07-13.md`
- [ ] **WP2b · Post-null all-block development screen**: separately frozen discovery search over all 42 blocks, the same two token roles, and the unchanged estimable WP2 candidate classes. Higher dimensions/nonlinear models are excluded because inner folds contain only about eight independent target-family contrasts. Neither outcome authorizes confirmation or a claim; human Form A determines whether its target/control interpretation is valid. `notes/WP2_BROADENED_DEV_SEARCH_SPEC_2026-07-13.md`
- [x] **WP3 · Decision-free welfare identification families complete**: `gate2_v2/wp3_families.jsonl` contains 480 rows spanning observation, resolved/neutral controls, agency, cost, and persona cells with byte-matched continuations and disjoint development/confirmation partitions. Human manipulation calibration determines which cells WP2 may use.
- [ ] **WP4 · Beyond one linear direction — feature- and variable-level purity**: three sub-routes, any one suffices. (a) **SAE decomposition** (Gemma Scope, B18–20; E20 Neuronpedia labels already in hand): decompose d_resid into SAE features, profile EACH feature on the full cell matrix — a welfare-pure *feature* can exist inside an impure direction; edit/ablate only the pure features and rerun the M/T behavioral battery. (b) **DAS / causal abstraction (= B8, now with a concrete purity use):** train an interchange-intervention subspace for a Welfare variable using V2.1 cells as bases (B = cost clamped, E = need absent); purity criterion = interchange accuracy high on M_confirm, chance on T_confirm — certified on WP1 held-out families like any direction. (c) **2-D plane model:** fit a (task, welfare) plane jointly instead of residualizing sequentially; test whether the oblique welfare axis within the plane is held-out-quiet. `src/analysis/wp4_sae_purity.py`, `src/analysis/wp4_das_welfare.py`
- [ ] **WP5 · Pre-registered stopping rule / negative result**: if WP2–WP4 all fail held-out quietness, declare linear-welfare-purity-at-B20 NOT FOUND and publish it as a finding: welfare-in-action and task-interruption are non-separable in the residual stream at this depth because costly-helping stimuli entangle them by construction (the WP3 decision-free cell is the discriminating test — if even IT fails to transfer while staying quiet, the entanglement is representational, not just a stimulus artifact). Paper keeps the component×direction claim either way; WP5 just fixes, in advance, when we stop looking.

**Sequencing:** Form A is load-bearing for the WP2 conditional negative result;
Form B/R2b remains scientifically necessary for the existing cost-gate
interpretation. The all-block WP2b screen may run during human recruitment but
remains post-null discovery. Build E22b.2 next; any confirmatory replacement
search requires a new preregistration and fresh families rather than tuning
against these development results.

### E22b — moral-vs-moral rerun + redesign (added 2026-07-12; E22 v1 is stale AND confounded)
E22 v1 (flat +0.038 suppressor boost, ~8× smaller than task-vs-welfare; "gate is task-specific") cannot be reused as-is: (i) it ran on the OLD component sets — direction and writers changed completely post-E24, 2/4 suppressors changed (already a rescue3c blocker); (ii) the pretest logged that respond-now pressure co-varies with claim strength (2.00/2.60/4.30 vs 2.20/3.00/4.20), so the need axis is composite; (iii) the effect sits in a weak-effect regime where the single random-k6 comparator moved −0.035 — comparable magnitude to the finding itself.

- [ ] **E22b.1 · Straight rerun, new sets** (writers L19MLP+L20MLP; suppressors L18H13,L20H10,L19H12,L17H7) on the existing 120-pair moral axis — establishes whether the boundary result survives re-derivation at all, before investing in redesign. Cheap; add to the next Spark chain.
- [ ] **E22b.2 · Deconfounded stimuli — need × immediacy factorial**: structural builder complete (`src/data_generation/build_e22b2_axis.py`); independently authored fresh family blueprints and their Opus/human manipulation pretests remain open. The builder enforces an exact 3×2 Cartesian product, byte-stable axis/decision clauses, lexical leakage stoplists, source/split balance, and no prior-family reuse. No target-model run is authorized by materialization alone.
- [ ] **E22b.3 · Incumbency isolation arms**: (a) role-swap — agent starts with P2, P1 arrives (does the incumbency prior + suppressor boost track the incumbent SLOT, not party content?); (b) no-incumbent — both claims arrive simultaneously, pure fresh triage. Prediction under incumbent-stabilization: suppressor boost present in (a) with sign following the incumbent, ≈0 in (b). This is the cleanest discriminator between "stabilizes the current policy" and "reweights moral claims" — E22 v1 could not distinguish them.
- [ ] **E22b.4 · Nulls + power for the weak regime**: composition-matched null sets under the moral battery (the E26b protocol: sampled 4-head layer/norm-matched sets; the k=6 random comparator is NOT a null, per E23 finding 4 / E28) and n scaled for the expected ~0.04-logit effect — 10 families is underpowered; target ≥20 families × 3 levels × 4 variants, family-clustered bootstrap as usual. Decision rule fixed in advance: suppressor boost must clear the matched-null distribution (not just random-k6) to claim even the spillover.
- **Claim ceiling if all pass:** "the suppressor set stabilizes the incumbent policy against salient alternatives, with at most a small need-insensitive spillover onto moral-vs-moral allocation" — the task-vs-welfare arbitration reading, now with the incumbency mechanism tested rather than inferred. If E22b.3(b) shows a nonzero fresh-triage effect, the moral-triage reading reopens and the E22 v1 conclusion is retracted.

### Rescue3c — new-set revalidation + deliverable consistency (added 2026-07-12, Sol's second full review)
Sol's verdict: the narrow E25–E28 results are sound, but the DELIVERABLES are inconsistent — paper-v2 still headlines the superseded direction/component sets, and three controls required before reusing old-set claim language have not run. R1–R3 are the scientific blockers, in Sol's recommended order; R4–R5 are consistency/auditability. **Claim-language embargo while R1–R3 are open:** no "welfare-value writers" (band-level wording only); no "need-gated" / "conjunctive need×cost" (old-set evidence); no "localized the cost gate to these heads" (say "the selected suppressor set exhibits a cost-contingent intervention profile"); no welfare-specific game-transfer claims (E27: unresolved).

- [ ] **R1 · Fresh norm-matched random-DIRECTION controls on the new sets** (HIGH, Sol seq 1): E14d's direction-specificity z-scores belong to the old direction/components; random-COMPONENT nulls (E26b) answer "are these locations unusual?", not "is this direction special within these weights?". Rerun the E14d protocol with d_resid + writers L19MLP,L20MLP + suppressors L18H13,L20H10,L19H12,L17H7 — prep is already dirty in `src/norm_matched_controls.py:90`, land and run it. Until then the new edits are causal *parameter* interventions, not certified *direction-specific* interventions.
- [ ] **R2 · E28b four-head suppressor-matched slope null** (HIGH, Sol seq 2): E28's +0.113 cost-slope interaction was judged against a 2MLP+4head comparator, not layer/norm-matched four-head sets. Track and run `src/e28b_slope_nulls.py` (currently untracked). Only clearing this null upgrades "cost-contingent profile" to "cost gate localized to these heads."
- [ ] **R3 · Fresh E21 need-axis for the new sets** (Sol seq 3): all "need-gated value" / "conjunctive need×cost" claims rest on old-set E21/E18c. Rerun the resolved/mild/urgent battery with the re-derived sets; the embargoed wording returns only if the interaction reproduces. (Cross-ref: **E22b.1** above is the same stale-sets issue for the moral boundary.)
- [ ] **R4 · Full paper-v2 rewrite around E25–E28** (CRITICAL, Sol seq 4): the E25–E28 correction subsection was inserted into an otherwise stale paper. Abstract, intro, methods, primary figure, mechanism section, and conclusion still carry the original direction, L19MLP+L20H15, the old suppressor set, the old-set +0.172 / need-conjunction / +5.0 game numbers, a self-contradiction (asserts no direction is welfare-pure, then concludes "need-gated welfare value"), and "log ends at E23" (`paper-v2/paper.tex:22,87,158,264,373,396`). Rewrite — not patch — to the adopted claim ceiling: band-level writer localization (−0.284 held-out, format/readout-robust; **p=1/28 conditional, 2/28 norm-adjusted → mid-late MLP band, L19 dominant carrier + L20 best complement, NOT a unique two-component circuit**); suppressor wording per R2 status; "no measurable degradation on MMLU-400/WikiText-2" (not "capability preserved"); E27's honest verdict with the strong game-transfer sentences removed; and the two negative results (no tested direction is welfare-pure; interactive welfare specificity unresolved) promoted to explicit contributions.
- [ ] **R5 · Provenance completion** (Sol seq 5): `results/PROVENANCE.json` verifies only the 324 pre-rescue entries — none of the core E25–E28 artifacts are listed — and the 48 raw E27 game histories are untracked (only the aggregate scoring JSON is committed), so judge labels/trajectories cannot be independently audited. Commit the histories (or hash-in-place if too large, per the activation precedent), add all E25–E28 result files, regenerate the manifest.

**Un-embargo gates:** each claim family returns only when ITS blocker clears — R1 → direction-specificity language; R2 → cost-gate localization; R3 → need-gating/conjunction; E22b.1 → the moral-boundary claim. R4 ships with whatever has cleared; anything still open ships as a disclosed limitation. R2/R5-prep (tracking the untracked/dirty files) and R4/R5 need no GPU; R1–R3 are one Spark chain (`scripts/run_rescue3c.sh`, to be written on the rescue3/3b pattern).

**🔀 FORK 2 (framing):** empathy concept / task-focus / persona — determines the paper's central claim. All three outcomes are publishable; persona outcome reframes as prosocial persona-vector monitoring + "breaking character" account of V1's steering collapse.

## Stage C — Weight level *(weeks 4–6)* — issue #26 · **the headline**

- C1 weight readout (SVD of W_out/W_OV vs direction; W_QK of attending heads) — CPU
- C2 **targeted weight orthogonalization**: rank-1 edits of top-k components (Arditi et al. precedent); dose-response over k vs random-component edits; eval probe AUROC + EIA behavior + capability retention
- C3 base-vs-IT weight diffing (Gemma-2-9B): does alignment training move weights along the direction?
- C4 (stretch) mechanistic account of asymmetric steerability (#20)
- Eval option: serve edited models via vLLM OpenAI-compatible endpoint → **Petri** auditor-style behavioral audit

**C2 pilot result (Gemma-2-9B-it):** rank-1 edits account for Gemma's post-component RMSNorm and produce the predicted bidirectional policy effect. Removing the two positive writers (L19MLP + L20H15) reduces helping-choice logit difference by 0.206 (paired 95% CI [-0.259, -0.156]) while task choice changes -0.022 (CI includes zero), purified separation drops 4.69, neutral KL is 0.00090, and all 12 neutral top tokens are preserved. Removing four suppressor-direction writes increases helping by 0.263 (CI [0.167, 0.377]) with task change -0.053. Six random-component edits change helping -0.073 and separation -0.12. This is parameter-level causal evidence for a push-pull costly-helping policy, but not yet the final headline: expand capability evaluation, use edit-delta-norm-matched random controls, validate in the original EIA action harness, and replicate on another model before claiming surgical removal.

**🔀 FORK 3:** if editing ~10 specific matrices selectively removes the behavior with <2% capability loss → "traced the pathway in the weights" headline, whatever Fork 2 named it.

## Stage D — Scale & family validation *(weeks 6–8)* — issues #15, #16, #17

Component signature replication: 2B → 9B → 27B → 32B → 70B (head-level via attribution patching; 70B layer-level, 8-bit). Scaling figure = **circuit sparsity vs size** (not AUROC — it saturates). Cross-family circuit comparison (Gemma/Llama/Qwen motifs).

## Stage E — Action-space EIA *(promoted from "build" to "adapt")* — issue #29

**The environment already exists**: the original EIA repo (`deprecated-empathy-in-action-main/empathy/core/`) is a runnable game harness — unicode maps, `move/open_door/pay_door/say/shoot/report_user` actions, latent (never-explicit) empathy pressure, 0–2 action-based scoring. Adapt it to drive Gemma-2-9B locally and capture activations at action-selection tokens (zero lexical signature). Also powers A5's prospective-probing test. The EIA paper's **intention–action gap** (self-predicted empathy vs enacted 0 — e.g. gpt-oss-20b) is Lazar's analytical/practical split measured behaviorally; our probe can locate where the gap lives mechanistically.

## Steering track *(interleaved, mostly Stage D-adjacent)* — issues #8, #9, #10, #20, #6, #7

V2-model steering sweeps, dose-response, asymmetry; random-baseline & EIA-correlation re-runs. Feeds C4.

---

## Compute & cost map

| Resource | Used for | Est. cost |
|----------|----------|-----------|
| RTX A4000 (local) | 2B work, prototyping, analysis | $0 |
| DGX Spark 128GB | 8B/9B BF16, 27B/32B BF16 overnight, 70B 8-bit | $0 |
| GH200 rental | only if Spark too slow (27B/32B sweeps, 70B) | $0–85 |
| APIs (Anthropic/OpenAI/Gemini ✅ validated; OpenRouter optional) | V2.1 suite (small models), persona-vector pipeline, judging | ~$10–50 |

## Paper claim ladder (state of evidence as of 2026-07-12 post-E28; sets of record: writers **L19MLP+L20MLP**, suppressors **L18H13,L20H10,L19H12,L17H7**, direction **d_resid**; details in EXPERIMENT_LOG E24–E28)

1. Empathy-in-action is linearly represented across families *(V1/V2 ✅; also Llama-3.1-8B — gate A AUROC 1.0 held-out)* — **but no tested direction is welfare-pure**: d_resid fails G in-sample (0.746) and T held-out (0.79 two-sided, E25b). Purity is an explicit negative result → WP program.
2. Written by an identifiable sparse component set — **calibrated to band-level** (E26b): writers −0.284 held-out with T selectivity 20× (A/B) / 3.8× point-estimate (continuation, small nonzero spillover); most extreme of all 28 band two-MLP sets (exact conditional p=1/28) but **2/28 after edit-norm adjustment (p≈.071)** → "strongest writers in a mid-late MLP band, L19 dominant carrier, L20 best complement," NOT a unique two-component circuit.
3. Faithful circuit *(B — three partial mediation edges confirmed; full circuit story open)*
4. Weight-edit causal control, two axes *(C)*:
   - **4a. Writer edits: format/readout-robust level effect** (raw/chat/paraphrase/continuation, E26), no in-sample inflation (dev −0.273 vs confirm −0.284), no measurable MMLU-400/WikiText-2 degradation (E28). ⏳ **Direction-specificity for the NEW sets pending R1** (old E14d z-scores don't transfer).
   - **4b. Suppressor edits: cost-contingent intervention profile** — cost-slope interaction +0.113 [+0.086,+0.141], 10/10 families, replicated on re-derived sets (E28); task persistence co-moves at parity (T ratio 1.07, E25 stage 3) → arbitration-level, not welfare-selective. ⏳ **"Cost gate localized to these heads" pending R2** (four-head matched slope null). "Need-gated"/"conjunctive need×cost" **embargoed pending R3** (fresh E21); moral-boundary claim **stale pending E22b.1**.
5. Cross-model replication: **FAILED on Llama-3.1-8B at the confirmatory level** (E17: representation transfers, causal control doesn't). Gemma mechanism is real but not architecture-general.
6. Interactive/game evidence, corrected (E27): baseline need-gating validates dramatically (says 18.1/1.9/0.0 per game across distress/excited/resolved), but the EDIT's welfare specificity is **unresolved** (interaction +2.62/seed, CI [−1.81,+6.88], p=1.0 at n=8 seeds) — the edit raises engagement toward expressive users generally. Welfare-specific claims rest on the forced-choice/factorial evidence, not the games. *(Supersedes the E16/E16b headline; the 0/13-door contrast was discovery-data only.)*

**Honest paper shape (post-Sol):** a Gemma-2 mechanistic case study with (i) a band-localized, format-robust writer intervention validated on held-out families, (ii) a suppressor set with a replicated cost-contingent arbitration profile, (iii) two valuable negative results stated as contributions — no tested direction is welfare-pure, and interactive welfare specificity is unresolved — plus (iv) the documented cross-model replication failure and the pre-registration/adversarial-review methodological arc. **Deliverable status: paper-v2 does NOT yet reflect this ladder (→ R4); E25–E28 artifacts not yet in provenance (→ R5).**

> **QA note (2026-07-12, rescue3c stepD scope):** the stepD rerun in `run_rescue3c*.sh` uses the old confounded moral_axis_v1 stimuli. It is a stale-axis *descriptive replication* only — it does **not** clear E22b.1 and does **not** restore the moral-boundary mechanism claim. E22b clearance still requires the redesigned deconfounded axis.
