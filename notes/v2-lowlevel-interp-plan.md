# V2 Plan: Lower-Level Interpretability (Components → Circuits → Weights)

**Created:** 2026-07-10
**Goal:** Upgrade the evidence standard from *"a probe detects the feature in activations"* (correlational) to *"we can trace the pathway in the weights that produces it"* (mechanistic/causal). This subsumes and extends Phase 4 of `v2-tasks-list.md`.

---

## The evidence ladder

Each rung is a strictly stronger claim than the one below it. V1 sits on rungs 1–2. V2 should reach rung 5.

| Rung | Claim | Method | Status |
|------|-------|--------|--------|
| 1 | The feature is linearly decodable from activations | Linear probes, AUROC | ✅ V1 (AUROC ~1.0) |
| 2 | Adding the direction changes behavior | Activation steering | ✅ V1 (asymmetric, fragile) |
| 3 | Specific components (heads, MLPs, layers) causally produce it | Activation/attribution patching, ablation | 🔲 V2 Phase A |
| 4 | The components form an identifiable circuit with known information flow | Path patching, sparse feature circuits, attribution graphs | 🔲 V2 Phase B |
| 5 | Specific weights produce it — editing them removes the behavior | Weight readout analysis, weight orthogonalization, base-vs-IT weight diffing | 🔲 V2 Phase C |

Key insight: **the probe direction is not just the V1 result — it's the tool that makes all of this cheap.** The projection of the residual stream onto the empathy direction is a scalar, differentiable metric. Every technique below uses it as the patching/attribution metric (alongside a behavioral logit-diff metric as a sanity check).

---

## Model matrix (work with the grain of existing infra)

Deep circuit work is only feasible where pretrained SAE/transcoder suites exist. Don't train SAEs from scratch — that was the expensive, optional item in the old plan and it's now unnecessary.

| Tier | Models | Depth | Why |
|------|--------|-------|-----|
| **1 — full mechanistic** | Gemma-2-2B, Gemma-2-9B (base + it), Llama-3.1-8B | Rungs 3–5 complete | Gemma Scope (SAEs + transcoders, all layers), Llama Scope. Fit in BF16 on any GPU ≥24GB. `circuit-tracer` supports Gemma-2-2B out of the box. |
| **2 — component-level validation** | Gemma-2-27B, Qwen-2.5-32B | Rung 3 (head/MLP granularity) | Fit on GH200 96GB in BF16. Gemma Scope has partial 27B coverage. Confirms Tier-1 findings aren't small-model artifacts. |
| **3 — coarse frontier validation** | Llama-3.1-70B | Rung 3 (layer granularity) + rung 5 weight-orthogonalization test | BF16 doesn't fit on one GH200 (140GB) — restrict to layer-level patching under 8-bit and the weight-edit test (edit BF16 weights on CPU, run quantized). |

**Framing for the paper:** "We fully trace the circuit at 2B–9B where SAE infrastructure exists, then show the same component signature (layer cluster, head motifs) replicates at 27B–70B." That's the standard frontier-interp move and reviewers accept it.

---

## Phase A — Component-level causal localization (rung 3)

*Which heads and MLPs write the empathy direction, and are they causally necessary?*

### A1. Direct feature attribution (DFA) — cheapest, do first
Decompose the probe projection at the readout layer into per-component contributions: every attention head's output (via its OV path) and every MLP's output are vectors added to the residual stream; dot each with the empathy direction. One hooked forward pass per example.
- **Output:** ranked list of "empathy-writing" components per model; concentration measure (is it 5 heads or 500?).
- **Compute:** trivial. Runs locally for Tier 1, hours on GH200 for Tier 2.

### A2. Activation patching at component granularity
On contrastive pairs (empathic ↔ non-empathic run of the same scenario), patch each head/MLP output from one run into the other. Two metrics: (a) probe projection, (b) behavioral logit-diff on empathic-vs-non-empathic continuation tokens.
- **Output:** causal effect per component; the "causal cluster" figure (old Fig 8) at head resolution instead of layer resolution.
- **Compute:** O(components × pairs) forward passes. Full sweep on Tier 1; noising direction only on Tier 2.

### A3. Attribution patching (gradient approximation) for Tier 2
AtP*-style linear approximation makes the full head×position sweep two forward + one backward pass per prompt. This is how 27B/32B get head-level maps without the O(n_components) cost.
- **Output:** head-level causal maps for Gemma-27B and Qwen-32B; cross-scale comparison with Tier 1.

### A4. Validation gate
DFA (who writes it) and patching (who's causally necessary) must agree on the top components. Disagreement is itself a finding (e.g., components that write the direction but are causally inert → the probe reads a correlate, which directly informs the task-focus confound).

---

## Phase B — Circuit level (rung 4)

*How do the components connect? What does the empathy circuit compute?*

### B1. Attribution graphs via transcoders (Gemma-2-2B)
Use the open-source `circuit-tracer` (Anthropic's attribution-graph method) on EIA scenario prompts. Produces feature-level graphs: which interpretable features feed the empathy-writing components.
- **Output:** attribution graphs for 3–5 scenarios; qualitative account ("emotion-of-user features → wellbeing-priority features → empathy-writing heads → output").

### B2. Sparse feature circuits (Gemma-2-9B, Marks et al. method)
Gemma Scope SAEs at every layer + attribution patching over SAE features → a sparse circuit of named features for the empathy-in-action behavior. 9B is the sweet spot: big enough to be taken seriously, fully covered by Gemma Scope.
- **Output:** the circuit diagram — the paper's centerpiece figure.

### B3. Path patching on the top components
For the top ~10 heads from Phase A: which upstream components feed their queries/keys/values, and which downstream components consume their output? Confirms the graph edges causally, not just via attribution.

### B4. Faithfulness check (causal-scrubbing-lite)
Ablate everything *outside* the identified circuit (mean-ablation) → behavior should survive. Ablate the circuit → behavior should die. Report completeness/faithfulness percentages. This is the difference between "we drew a plausible graph" and "the graph is the mechanism."

### B5. Confound resolution (directly addresses the V1 "task-focus" critique)
Run the identified circuit on control sets: (i) task-focus varies, empathy constant; (ii) empathy varies, task-focus constant; (iii) general sentiment/warmth without action. If the circuit fires for (i) but not (ii), V1's probe was a task-distraction detector — a publishable resolution either way. **This is the single highest-value scientific payoff of going lower-level: it settles what the probe actually detects.**

---

## Phase C — Weight level (rung 5, the new evidence standard)

*Trace the vía in the weights.*

### C1. Weight readout analysis (static, input-independent)
For the top causal components: does the empathy direction align with specific singular vectors of W_out (MLPs) / W_OV (heads)? Which neurons (rows) write it? For the attending heads, analyze W_QK: what token relationships do they compute, independent of any input? This grounds the circuit in the parameters themselves.
- **Compute:** free — pure linear algebra on downloaded weights.

### C2. Targeted weight orthogonalization — the headline causal test
Precedent: Arditi et al. removed refusal by projecting the refusal direction out of every weight matrix that writes to the residual stream. Do the *targeted* version:
1. Orthogonalize **only the top-k components from Phase A** w.r.t. the empathy direction (rank-1 edits to specific W_out matrices).
2. Measure: probe AUROC on the edited model, EIA behavioral scores, general capability (MMLU subset, perplexity) to show the edit is surgical.
3. Dose-response over k: how many components must be edited before behavior collapses? Compare against editing k *random* components of equal norm (the natural sequel to the random-direction control just added on this branch).

**If editing ~10 specific weight matrices selectively removes empathy-in-action while leaving capabilities intact, that is the "trace the pathway in the weights" result — a permanent, input-independent, parameter-level localization.**
- Run on all Tier-1 models; replicate the top-layer version on Llama-70B (edit on CPU, evaluate quantized).

### C3. Base vs instruct weight diffing
For Gemma-2-9B base vs it (weights are public for both):
- Compare DFA/causal maps base vs it: does alignment training create, strengthen, or merely reuse the empathy-writing components?
- Project the raw weight diff (W_it − W_base) per component onto the empathy direction: does fine-tuning literally move weights along it?
- **Payoff:** upgrades paper claim 6 ("alignment training modifies empathy geometry") from geometric observation to mechanism, and directly speaks to the FAccT angle.

### C4. Mechanistic account of asymmetric steerability (stretch)
V1's most striking finding — pro-empathy steering works, anti-empathy collapses the model — should now be explainable: when steering negatively, which components saturate or go out of distribution first? Does the collapse route through the identified circuit or through LayerNorm/generic pathways? Even a partial answer converts V1's most-criticized finding (N=1 Dolphin) into a mechanism.

---

## Phase D — Cross-model and scaling validation

- Replicate the component signature (layer cluster depth-fraction, head motif count, circuit sparsity) across Tier 1 → 2 → 3. The scaling figure becomes "circuit sparsity / localization vs model size" — much stronger than the old "AUROC vs size" (which saturates at 1.0 and shows nothing).
- Cross-family: do Gemma, Llama, Qwen converge on structurally similar circuits? (Upgrades the old "direction cosine similarity" analysis to circuit-level comparison.)

---

## Tooling

| Need | Tool |
|------|------|
| Hooks, patching, DFA on ≤9B | TransformerLens |
| Patching on 27B–70B | NNsight (or raw HF hooks) — TransformerLens doesn't scale there |
| SAEs/transcoders | Gemma Scope, Llama Scope via SAELens (no training needed) |
| Attribution graphs | `circuit-tracer` (open source, Gemma-2-2B supported) |
| Sparse feature circuits | Marks et al. reference implementation, adapted |
| Weight edits | Plain PyTorch on state dicts |

## Hardware

We now have local hardware that covers most of the plan; cloud rental becomes optional.

| Machine | Specs | Role |
|---------|-------|------|
| **RTX A4000 (local)** | 16GB VRAM | Interactive dev + all Gemma-2-2B work (DFA, patching, `circuit-tracer` attribution graphs), probe training, analysis, figures. Llama-3.1-8B / Gemma-2-9B in 8-bit for prototyping. |
| **DGX Spark** | GB10, 128GB unified memory (~273 GB/s bandwidth) | The workhorse. Fits Gemma-2-9B and Llama-3.1-8B in BF16 with room for SAEs; fits Gemma-2-27B (~54GB) and Qwen-2.5-32B (~64GB) in BF16; fits Llama-70B in 8-bit (~70GB). Bandwidth is modest, so it's slow per-token — but patching/DFA sweeps are batch forward-pass workloads, perfect for overnight jobs. |
| **GH200 (rental, $1.49/hr)** | 96GB HBM | Only where speed matters: large attribution-patching sweeps on 27B/32B if Spark overnight runs prove too slow, and any 70B work needing real throughput. |

Caveats for the Spark: it's aarch64 — verify ARM builds of TransformerLens/SAELens/NNsight early (all are pure-Python over PyTorch, which has ARM CUDA wheels for GB10, so this should be fine, but check in week 1). Prototype every experiment on the A4000 with Gemma-2-2B before scaling it on the Spark.

## Compute budget

| Item | Where | Cloud cost |
|------|-------|-----------|
| Phase A Tier 1 (2B, 8B, 9B×2) | A4000 + Spark | $0 |
| Phase A Tier 2 (27B, 32B attribution patching) | Spark overnight; GH200 fallback | $0–30 |
| Phase B (graphs, feature circuits, path patching, faithfulness) | A4000 (2B) + Spark (9B) | $0 |
| Phase C weight edits + eval sweeps (Tier 1) | Spark | $0 |
| Phase C 70B: layer patching (8-bit) + weight-edit eval | Spark (slow) or GH200 (~10h) | $0–15 |
| Phase D + reruns buffer | GH200 as needed | $0–40 |
| **Total** | | **~$0–85** (was ~$165 all-cloud) |

This *replaces* the old plan's "SAE training 20–40h" line — pretrained suites (Gemma Scope, Llama Scope) make it free.

## Priority order

1. **A1 + A2 on Gemma-2-9B** — one week, immediately tells you if the circuit is sparse (everything downstream depends on this).
2. **C2 targeted weight orthogonalization on 9B** — the headline result; do it as soon as A gives you the component list.
3. **B2 sparse feature circuits + B4 faithfulness** — the centerpiece figure.
4. **B5 confound controls** — settles empathy-vs-task-focus.
5. **C3 base-vs-IT diffing** — the FAccT/alignment angle.
6. A3 Tier-2 validation → D scaling → C4 asymmetry mechanism (stretch).

## Success criteria

- **Minimum:** component-level causal map on ≥2 models; top-k weight-orthogonalization selectively removes empathy behavior on ≥1 model with <2% capability degradation.
- **Target:** faithful sparse feature circuit on Gemma-2-9B (faithfulness >80%, completeness >80%); circuit signature replicates at 27B+; confound controls resolve what the probe detects.
- **Stretch:** cross-family circuit convergence; mechanistic account of steering asymmetry; base-vs-IT shows alignment training's parameter-level effect on the circuit.

## Risks

- **Circuit isn't sparse** (empathy is diffuse across hundreds of components). Mitigation: that's still a paper — "empathy-in-action is linearly decodable but not localizable" is an important negative result about the limits of circuit-style evidence for high-level social behaviors. The dose-response curve in C2 quantifies it.
- **Probe metric and behavioral metric disagree** during patching. Mitigation: report both throughout; the divergence is itself the detection-vs-control story V1 started.
- **Tier-3 (70B) infeasibility** on single GH200. Mitigation: 70B only gets layer-level patching + weight-edit eval; the deep claims live at 2B–9B by design.
