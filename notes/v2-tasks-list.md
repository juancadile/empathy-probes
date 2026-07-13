# V2 Empathy-in-Action Probes: Complete Task List for NeurIPS/FAccT

> **HISTORICAL TASK LIST — SUPERSEDED 2026-07-13.** Retained for issue and
> planning traceability only. Operational authority now lives in
> `SCIENTIFIC_PREREG_INDEX_2026-07-13.md`, `EXECUTION_ORDER_2026-07-13.md`,
> and the frozen gate/experiment preregistrations they index. The previously
> authoritative `v2-lowlevel-interp-plan.md` is itself historical; unchecked
> boxes and claim language below must not be treated as approved execution
> instructions.
>
> **Updated 2026-07-10 (historical):** Phase 4 was superseded at that time by the staged mechanistic plan in
> `v2-lowlevel-interp-plan.md` (components → circuits → weights). Key changes:
> we **no longer train our own SAEs** — we use pretrained suites (Gemma Scope, Llama Scope)
> — and the deep circuit/weight-level work runs on Gemma-2-2B/9B and Llama-3.1-8B (Tier 1),
> with component-level validation on 27B/32B and coarse validation on 70B.
> Compute: local DGX Spark (128GB unified) + RTX A4000 cover most phases; GH200 rental is
> now optional (~$0–85 instead of ~$165).

## Current Status (Nov 25, 2024)

### Dataset Generation: IN PROGRESS
| Model | Type | Pairs | Status | Family |
|-------|------|-------|--------|--------|
| Claude Sonnet | Closed | 500 | ✅ Done | Anthropic |
| Claude Haiku | Closed | 500 | ✅ Done | Anthropic |
| GPT-4o | Closed | 500 | ✅ Done | OpenAI |
| GPT-5.1 | Closed | 500 | ✅ Done | OpenAI |
| Gemini 2.5 Flash | Closed | 500 | ✅ Done | Google |
| Llama-70B-FP8 | Open | 500 | ✅ Done | Meta |
| Gemma-27B | Open | 500 | ✅ Done | Google |
| Qwen-32B | Open | 500 | ✅ Done | Alibaba |
| **Yi-1.5-34B** | Open | 500 | 🔲 TODO | 01.AI |
| **Mistral-Small-24B** | Open | 500 | 🔲 TODO | Mistral |
| **Total** | | **5,000** | | **5 open families** |

### Code Status
| Component | V1 Status | V2 Status | Notes |
|-----------|-----------|-----------|-------|
| Linear probes | Done (Phi-3, Dolphin, Qwen-7B) | **NEEDS ADAPTATION** | Adapt for Llama-70B, Gemma-27B, Qwen-32B |
| Steering | Done (Phi-3, Dolphin) | **NEEDS ADAPTATION** | Scale to larger models |
| Random baseline | Done | **NEEDS RE-RUN** | Run on v2 models |
| Lexical ablation | Done | **NEEDS RE-RUN** | Run on v2 models |
| Activation extraction | Done (small models) | **NEEDS ADAPTATION** | Handle 70B+ models, memory management |
| SAE training | NOT IMPLEMENTED | **NEW CODE NEEDED** | |
| Causal patching | NOT IMPLEMENTED | **NEW CODE NEEDED** | |

**Key Challenge:** V1 code used 4-bit quantization for small models. V2 needs:
- BF16 precision for interpretability (not FP8)
- Memory-efficient activation extraction for 70B models
- Possibly tensor parallelism or gradient checkpointing

### Open-Source Model Families (5 for Cross-Model Analysis)

| Model | Params | Family | HuggingFace ID | Architecture |
|-------|--------|--------|----------------|--------------|
| Llama-3.1-70B | 70B | Meta | `meta-llama/Llama-3.1-70B-Instruct` | LLaMA + GQA |
| Gemma-2-27B | 27B | Google | `google/gemma-2-27b-it` | Gemma + GQA |
| Qwen-2.5-32B | 32B | Alibaba | `Qwen/Qwen2.5-32B-Instruct` | Qwen + GQA |
| **Yi-1.5-34B** | 34B | 01.AI | `01-ai/Yi-1.5-34B-Chat` | LLaMA-like + GQA |
| **Mistral-Small-24B** | 24B | Mistral | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral + SWA + GQA |

**Why these 5 families?**
- Geographic diversity: USA (Meta, Google), China (Alibaba, 01.AI), France (Mistral)
- Architectural diversity: Standard attention vs Sliding Window Attention
- Size range: 24B - 70B (good for scaling analysis)
- All Apache 2.0 or permissive licenses

---

## Phase 1: Data Consolidation & Preparation

### Task 1.1: Build Final V2 Dataset
- [ ] Merge all generation_progress_*.jsonl files into unified dataset
- [ ] Create train/val/test splits (70/15/15)
- [ ] Ensure balanced distribution across scenarios and models
- [ ] Validate data quality (no truncations, proper formatting)

**Files to modify:** `build_final_dataset.py` or create `build_v2_dataset.py`

### Task 1.2: Create Activation Extraction Script for Open-Source Models
- [ ] Write `extract_activations_opensource.py` using HuggingFace transformers
- [ ] Extract mean-pooled activations for ALL layers
- [ ] Save as .pt or .npy files per model/layer
- [ ] Support for: Llama-70B, Gemma-27B, Qwen-32B
- [ ] Option for token-level activations (for SAEs)

**New file:** `src/extract_activations_opensource.py`

**Storage estimate:**
- Mean-pooled: ~1.3GB per model (~4GB total)
- Token-level: ~50-100GB per model (if needed for SAEs)

---

## Phase 2: Core Interpretability Experiments

### Task 2.1: Adapt Linear Probes for V2 Models
**Existing code:** `probe_extraction.py`, `probe_extraction_cross_model.py` (worked on Phi-3, Dolphin, Qwen-7B)

**Adaptation needed:**
- [ ] Update MODEL_CONFIGS to add Llama-70B, Gemma-27B, Qwen-32B
- [ ] Remove 4-bit quantization, use BF16 for accurate activations
- [ ] Add memory management for 70B model (gradient checkpointing or layer-by-layer extraction)
- [ ] Train probes on ALL layers (not just selected)
- [ ] Compute AUROC per layer for each model
- [ ] Generate Figure: AUROC by layer for all 3 models

**Files to modify:** `probe_extraction_cross_model.py`
**Output:** `results/probes/{model}_empathy_direction_layer_{N}.npy`

### Task 2.2: Cross-Model Probe Transfer
- [ ] Train probe on Model A → test on Model B activations
- [ ] Create 3x3 transfer matrix (Llama ↔ Gemma ↔ Qwen)
- [ ] Also test: Feed closed-source text through open-source model → probe
- [ ] Generate Figure: Cross-model transfer heatmap

**New file:** `src/probe_transfer_analysis.py`

### Task 2.3: Re-run Random Baseline for V2 Models
**Existing code:** `random_baseline_proper.py` (worked on V1 models)

**Adaptation needed:**
- [ ] Update to use V2 model activations
- [ ] Run random direction baseline for all 3 open-source models
- [ ] Verify AUROC ≈ 0.5 for random directions
- [ ] Statistical significance testing

**Files to modify:** `random_baseline_proper.py`

### Task 2.4: Re-run EIA Behavioral Correlation for V2 Models
**Existing code:** `eia_evaluator.py` (worked on V1 models)

**Adaptation needed:**
- [ ] Update to use V2 model activations and probes
- [ ] Compute projection scores for all pairs
- [ ] Correlate with EIA rubric scores (0/1/2)
- [ ] Target: Pearson r > 0.4
- [ ] Generate Figure: Projection vs EIA score scatter

**Files to modify:** `eia_evaluator.py`

---

## Phase 3: Steering Experiments

### Task 3.1: Adapt Steering for V2 Models
**Existing code:** `steering.py`, `steering_comprehensive.py`, `steering_cross_model.py` (worked on Phi-3, Dolphin)

**Adaptation needed:**
- [ ] Update MODEL_CONFIGS to add Llama-70B, Gemma-27B, Qwen-32B
- [ ] Handle larger models (memory management, possibly vLLM for generation)
- [ ] Alpha sweep: α ∈ {-20, -10, -5, -3, 0, 3, 5, 10, 20}
- [ ] 5 samples per (α, layer, scenario) combination
- [ ] Test on top-3 AUROC layers per model
- [ ] All 5 scenarios

**Files to modify:** `steering_comprehensive.py`, `steering_cross_model.py`
**Output:** `results/steering/{model}_comprehensive.json`

### Task 3.2: Steering Analysis & Visualization
- [ ] Compute success rate per α value
- [ ] Identify asymmetry (positive vs negative steering)
- [ ] Detection-steering gap analysis
- [ ] Generate Figures:
  - Dose-response curves
  - Steering success by layer
  - Asymmetry visualization

**Files to modify:** `analyze_steering_results.py`, `visualize_steering.py`

### Task 3.3: Cross-Model Steering Transfer
- [ ] Use Llama's empathy direction → steer Gemma/Qwen
- [ ] Measure if steering transfers across architectures
- [ ] This is a novel contribution if it works (or doesn't!)

**New file:** `src/steering_transfer.py`

---

## Phase 4: Advanced Mechanistic Interpretability

> **Superseded by `v2-lowlevel-interp-plan.md`** — the tasks below are kept for issue
> traceability but the authoritative scope, ordering, and model assignments live there.

### Task 4.0: Direct Feature Attribution (NEW — do first, cheapest)
- [ ] Decompose probe projection into per-component contributions (each head's OV output
      and each MLP output dotted with the empathy direction)
- [ ] One hooked forward pass per example; rank "empathy-writing" components
- [ ] Measure circuit concentration (5 heads or 500?)

**New file:** `src/direct_feature_attribution.py`
**Maps to:** Plan Phase A1. Runs on A4000 (2B) / Spark (9B).

### Task 4.1: Implement Activation Patching
- [ ] Write `activation_patching.py`
- [ ] Patch at **component granularity** (heads/MLPs) on Tier-1 models, layer granularity on 70B
- [ ] Metrics: probe projection AND behavioral logit-diff
- [ ] Identify "causal cluster" — now at head resolution
- [ ] Attribution patching (AtP*-style gradient approximation) for 27B/32B sweeps

**New file:** `src/activation_patching.py`
**Maps to:** Plan Phases A2–A3.

### Task 4.2: Implement Attention Head / MLP Ablation
- [ ] Write `component_ablation.py`
- [ ] Mean-ablate specific attention heads or MLPs
- [ ] Measure impact on empathy probe projection + behavior
- [ ] Validation gate: DFA (who writes it) and ablation (who's necessary) must agree

**New file:** `src/component_ablation.py`
**Maps to:** Plan Phase A2/A4.

### Task 4.3: SAE Feature Circuits via PRETRAINED SAEs (was: SAE training — ❌ no longer training our own)
- [ ] ~~Train SAEs from scratch~~ → **Use Gemma Scope (2B/9B, partial 27B) and Llama Scope (8B) via SAELens**
- [ ] Sparse feature circuits (Marks et al. method) on Gemma-2-9B
- [ ] Attribution graphs via `circuit-tracer` on Gemma-2-2B
- [ ] Feature ablation: remove circuit features → measure probe + behavior change
- [ ] Faithfulness/completeness check (ablate outside circuit → behavior survives; ablate circuit → dies)
- [ ] Confound controls: task-focus-varies vs empathy-varies scenario sets

**Maps to:** Plan Phases B1, B2, B4, B5. Saves the old 20–40h SAE-training compute line entirely.

### Task 4.4: Path Patching / Causal Mediation (Advanced)
- [ ] Path patching on top ~10 heads from Task 4.0/4.1: upstream Q/K/V feeds, downstream consumers
- [ ] Confirm attribution-graph edges causally

**New file:** `src/path_patching.py`
**Maps to:** Plan Phase B3.

### Task 4.5: Weight-Level Localization (NEW — the headline)
- [ ] Weight readout: align empathy direction with singular vectors of W_out / W_OV of top components
- [ ] **Targeted weight orthogonalization**: rank-1 edit top-k components' weights (Arditi et al. precedent);
      dose-response over k vs random-component edits (sequel to `random_direction_control.py`)
- [ ] Eval edited models: probe AUROC, EIA behavioral scores, capability retention (MMLU subset, perplexity)
- [ ] Base-vs-instruct weight diffing on Gemma-2-9B (does alignment training move weights along the direction?)

**New files:** `src/weight_readout.py`, `src/weight_orthogonalization.py`
**Maps to:** Plan Phases C1–C3.

---

## Phase 5: Scaling Analysis

### Task 5.1: Add Smaller Models for Scaling Curve
- [ ] Run probes on Gemma-2-2B and Gemma-2-9B
- [ ] Extract activations, train probes, compute AUROC
- [ ] Creates scaling curve: 2B → 9B → 27B → 32B → 70B

**Files to modify:** `probe_extraction_cross_model.py`

### Task 5.2: Scaling Law Analysis
- [ ] Plot AUROC vs model size (log scale)
- [ ] Compute linear separability as function of parameters
- [ ] Generate Figure: Scaling law of empathy representability

**New file:** `src/analyze_scaling.py`

### Task 5.3: Cross-Family Geometric Analysis
- [ ] Compute cosine similarity of empathy directions across models
- [ ] Measure if empathy direction is "universal" or family-specific
- [ ] Generate Figure: Cross-family direction similarity heatmap

**New file:** `src/geometric_analysis.py`

---

## Phase 6: Paper Figures & Results

### Figure List for NeurIPS/FAccT

| Figure | Description | Status |
|--------|-------------|--------|
| Fig 1 | AUROC by layer (all 3 open-source models) | To generate |
| Fig 2 | Random baseline distribution | Exists, update for v2 |
| Fig 3 | EIA correlation scatter | Exists, update for v2 |
| Fig 4 | Cross-model transfer heatmap | New |
| Fig 5 | Steering dose-response curves | To generate |
| Fig 6 | Detection-steering gap visualization | Exists, update |
| Fig 7 | Scaling law: AUROC vs model size | New |
| Fig 8 | Causal layer cluster (activation patching) | New |
| Fig 9 | Cross-family empathy direction similarity | New |
| Fig 10 | SAE feature analysis (if implemented) | New (optional) |

### Task 6.1: Regenerate All Figures
- [ ] Update `generate_figures.py` for v2 data
- [ ] Ensure publication quality (300 DPI, proper fonts)
- [ ] NeurIPS format compliance

---

## Compute Requirements

### Local hardware (primary — new as of July 2026)

| Machine | Specs | Covers |
|---------|-------|--------|
| **RTX A4000** | 16GB VRAM | Gemma-2-2B everything (DFA, patching, circuit-tracer), probes, prototyping, analysis, figures |
| **DGX Spark** | GB10, 128GB unified memory | Gemma-2-9B / Llama-8B in BF16; Gemma-27B / Qwen-32B in BF16; Llama-70B in 8-bit. Slow per-token — run sweeps overnight. Verify ARM (aarch64) builds of TransformerLens/SAELens/NNsight in week 1. |

### GH200 96GB rental (~$1.49/hr) — now optional

| Phase | Cloud hours | Notes |
|-------|-------------|-------|
| 27B/32B attribution-patching sweeps | 0–20h | Only if Spark overnight runs are too slow |
| 70B throughput work | 0–10h | Layer patching + weight-edit eval otherwise on Spark 8-bit |
| ~~SAE training~~ | **0h** | ❌ Cut — using pretrained Gemma Scope / Llama Scope |
| Buffer / reruns | 0–25h | |
| **Total** | **0–55h ≈ $0–85** | Was 66–109h ≈ $100–165 |

### CPU-only tasks
- Data consolidation, weight readout analysis (pure linear algebra on state dicts)
- Result analysis, figure generation, paper writing

---

## Priority Order (Recommended)

### Must-Have for NeurIPS (Priority 1)
1. [x] Task 1.1: Build final v2 dataset
2. [x] Task 1.2: Activation extraction script
3. [x] Task 2.1: Linear probes on open-source models
4. [x] Task 2.2: Cross-model probe transfer
5. [ ] Task 4.0 + 4.1: DFA + component-level patching on Gemma-2-9B (Plan Phase A)
6. [ ] Task 4.5: Targeted weight orthogonalization (Plan Phase C2 — the headline)
7. [ ] Task 4.3: Sparse feature circuits via pretrained SAEs + faithfulness (Plan Phases B2/B4)

### High Impact (Priority 2)
8. [ ] Task 4.3 confound controls: task-focus vs empathy (Plan Phase B5)
9. [ ] Task 4.5 base-vs-IT weight diffing (Plan Phase C3)
10. [ ] Task 3.1: Comprehensive steering on V2 models
11. [ ] Task 5.2: Scaling figure — now "circuit sparsity vs size", not AUROC (which saturates)
12. [ ] Task 5.3: Cross-family analysis — upgraded to circuit-level comparison

### Optional but Impressive (Priority 3)
13. [ ] Task 4.4: Path patching
14. [ ] Task 3.3: Cross-model steering transfer
15. [ ] Mechanistic account of steering asymmetry (Plan Phase C4)

---

## Key Claims for Paper

After completing these tasks, the paper will support:

1. **"Empathy-in-action is linearly represented in frontier LLMs"**
   - Evidence: AUROC > 0.9 on Llama-70B, Gemma-27B, Qwen-32B

2. **"The representation scales with model size"**
   - Evidence: Scaling curve from 2B to 70B

3. **"The direction is partially universal across model families"**
   - Evidence: Cross-model transfer and geometric analysis

4. **"Detection does not imply controllability"**
   - Evidence: Steering experiments showing asymmetry/failure modes

5. **"Specific layers causally mediate empathic behavior"**
   - Evidence: Activation patching results

6. **"Alignment training modifies empathy geometry"** (for FAccT angle)
   - Evidence: Base vs instruct model comparison

---

## Quick Start Commands

```bash
# 1. SSH into compute instance
ssh -i ssh/juan.pem ubuntu@<INSTANCE_IP>

# 2. Extract activations (run first)
python src/extract_activations_opensource.py --model llama-70b --all-layers

# 3. Train probes
python src/probe_extraction_cross_model.py --model llama-70b

# 4. Run steering
python src/steering_comprehensive.py --model llama-70b --layers 15,20,25

# 5. Activation patching
python src/activation_patching.py --model llama-70b
```

---

## Notes

- **FP8 vs BF16**: Use BF16 for all interpretability experiments (not FP8). 8-bit acceptable only for 70B coarse validation.
- **Storage**: store all activations liberally
- **Checkpointing**: Save intermediate results frequently
- **Compute**: default to local (A4000 → prototype, DGX Spark → sweeps); rent GH200 only when throughput-bound
