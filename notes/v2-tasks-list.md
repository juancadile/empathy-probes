# V2 Empathy-in-Action Probes: Complete Task List for NeurIPS/FAccT

## Current Status (Nov 25, 2024)

### Dataset Generation: COMPLETE
| Model | Type | Pairs | Status |
|-------|------|-------|--------|
| Claude Sonnet | Closed | 500 | Done |
| Claude Haiku | Closed | 500 | Done |
| GPT-4o | Closed | 500 | Done |
| GPT-5.1 | Closed | 500 | Done |
| Gemini 2.5 Flash | Closed | 500 | Done |
| Llama-70B-FP8 | Open | 500 | Done |
| Gemma-27B | Open | 500 | Done |
| Qwen-32B | Open | 500 | Done |
| **Total** | | **4,000** | |

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

### Task 4.1: Implement Activation Patching
- [ ] Write `activation_patching.py`
- [ ] Patch layer-by-layer: replace empathic → non-empathic activations
- [ ] Measure behavioral change (does output become less empathic?)
- [ ] Identify "causal cluster" of layers

**New file:** `src/activation_patching.py`

### Task 4.2: Implement Attention Head / MLP Ablation
- [ ] Write `component_ablation.py`
- [ ] Zero out specific attention heads or MLP layers
- [ ] Measure impact on empathy probe projection
- [ ] Identify critical components

**New file:** `src/component_ablation.py`

### Task 4.3: Implement SAE Training (Optional but High Impact)
- [ ] Write `train_sae.py` using e.g. SAELens or custom implementation
- [ ] Train on selected layers (top AUROC layers)
- [ ] Bottleneck dims: 16, 32, 64
- [ ] Identify monosemantic empathy features
- [ ] Feature ablation: remove SAE features → measure probe change

**New file:** `src/train_sae.py`
**Dependencies:** SAELens or implement TopK SAE

**Compute note:** Requires full precision (BF16), not FP8

### Task 4.4: Path Patching / Causal Mediation (Advanced)
- [ ] Implement path patching through attention → MLP paths
- [ ] Compute indirect effects of empathy direction
- [ ] Identify specific circuits for empathy representation

**New file:** `src/path_patching.py`

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

### GH200 96GB Instance (~$1.49/hr)

| Phase | Estimated Hours | Tasks |
|-------|-----------------|-------|
| Activation extraction | 8-12h | Extract all layers for 3 models |
| Probe training | 4-6h | Train probes, compute AUROC |
| Steering experiments | 20-30h | Full α sweep, 5 samples each |
| Activation patching | 10-15h | Layer-by-layer patching |
| SAE training | 20-40h | 2-4 SAEs per model (optional) |
| Scaling models | 4-6h | Gemma-2B, Gemma-9B |
| **Total** | **66-109h** | **~$100-165** |

### Local Machine Tasks (No GPU needed)
- Data consolidation
- Result analysis
- Figure generation
- Paper writing

---

## Priority Order (Recommended)

### Must-Have for NeurIPS (Priority 1)
1. [ ] Task 1.1: Build final v2 dataset
2. [ ] Task 1.2: Activation extraction script
3. [ ] Task 2.1: Linear probes on open-source models
4. [ ] Task 2.2: Cross-model probe transfer
5. [ ] Task 3.1: Comprehensive steering
6. [ ] Task 5.2: Scaling law figure

### High Impact (Priority 2)
7. [ ] Task 4.1: Activation patching (causal cluster)
8. [ ] Task 4.2: Component ablation
9. [ ] Task 5.3: Cross-family geometric analysis

### Optional but Impressive (Priority 3)
10. [ ] Task 4.3: SAE training
11. [ ] Task 4.4: Path patching
12. [ ] Task 3.3: Cross-model steering transfer

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

- **FP8 vs BF16**: Use BF16 for all interpretability experiments (not FP8)
- **Storage**: 100TB available, store all activations liberally
- **Checkpointing**: Save intermediate results frequently
- **Instance**: Keep H100 running until all experiments complete
