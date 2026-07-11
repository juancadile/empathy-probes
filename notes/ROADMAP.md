# V2 Project Roadmap — Empathy-in-Action: from Probes to Weights

**Created:** 2026-07-10 · **Living document — check items off here.**
Companion docs: `v2-lowlevel-interp-plan.md` (methods detail), `v2-tasks-list.md` (per-task breakdown), `v2_1-stimulus-suite-design.md` (dataset design).

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

**Decision gate:** none — 0.3 unblocks B5/#27 and #28 later; generate while GPUs do Stage A.

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

**🔀 FORK 2 (framing):** empathy concept / task-focus / persona — determines the paper's central claim. All three outcomes are publishable; persona outcome reframes as prosocial persona-vector monitoring + "breaking character" account of V1's steering collapse.

## Stage C — Weight level *(weeks 4–6)* — issue #26 · **the headline**

- C1 weight readout (SVD of W_out/W_OV vs direction; W_QK of attending heads) — CPU
- C2 **targeted weight orthogonalization**: rank-1 edits of top-k components (Arditi et al. precedent); dose-response over k vs random-component edits; eval probe AUROC + EIA behavior + capability retention
- C3 base-vs-IT weight diffing (Gemma-2-9B): does alignment training move weights along the direction?
- C4 (stretch) mechanistic account of asymmetric steerability (#20)
- Eval option: serve edited models via vLLM OpenAI-compatible endpoint → **Petri** auditor-style behavioral audit

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

## Paper claim ladder (target)

1. Empathy-in-action is linearly represented across families *(V1/V2 ✅)*
2. It is written by an identifiable sparse set of components *(A)*
3. Those components form a faithful circuit whose features show it is ⟨empathy | task-focus | persona⟩ *(B — Fork 2)*
4. Editing the specific weights that implement it selectively removes the behavior *(C — headline)*
5. The mechanism replicates across scale and family; alignment training modifies it at the parameter level *(C3+D)*
6. (demo) The direction governs *action selection*, not narrative style *(E)*
