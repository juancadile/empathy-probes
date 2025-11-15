# Cloud Strengthening Plan: v0 → v1

Branch: `cloud-strengthening`
Target: Address all "early" weaknesses identified in arXiv v0
Infrastructure: Vast.ai GPU instances + uncensored models (GPT-OSS, Llama-3-70B-Instruct, etc.)

---

## Objectives

Transform the preliminary v0 into a robust v1 ready for main-track venues (NeurIPS 2026, AIES 2027).

### Current Weaknesses to Address:

1. **Small dataset** - 50 pairs (35 train, 15 test)
2. **Single model** - Only Phi-3 tested
3. **No causal validation** - Missing activation patching, ablations
4. **Perfect AUROC concerns** - Layer 12's 1.0 AUROC suggests artifacts
5. **Extreme alpha requirements** - α=20 is unusually high, needs investigation

---

## Phase 1: Dataset Expansion (Target: 200+ pairs)

### Why uncensored models?
- Safety-filtered models (Claude, GPT-4) refuse to generate realistic "cruel/selfish" responses
- Need genuine task-prioritization without empathy, not refusal text
- Models: Llama-3-70B-Instruct, Mixtral-8x7B-Instruct, GPT-OSS variants

### Tasks:

#### 1.1 Generate 150 new contrastive pairs
**Infrastructure**: Vast.ai instance with vLLM
- **Model**: Llama-3-70B-Instruct (or similar uncensored variant)
- **Scenarios**: Expand beyond EIA's 5 scenarios
  - Add: Medical triage, resource allocation, emergency response
  - Add: Ambiguous cases (weak task-conflict)
  - Add: Non-task-conflict empathy (pure altruism)
- **Output**: `data/dataset_expanded.json` (200 total pairs)

**Script**: `src/generate_dataset_cloud.py`
- Use existing prompt templates from `src/generate_dataset.py`
- Add vLLM batching for efficiency
- Add scenario diversity metrics

**Cost estimate**: ~$5-10 for 150 pairs on Vast.ai (A100 80GB ~$1.50/hr)

#### 1.2 Validate new pairs
- Human review sample (20 pairs)
- Consistency checks (sentiment analysis, keyword overlap)
- EIA scoring for behavioral validation

---

## Phase 2: Cross-Model Validation

### 2.1 Test on Llama-3-8B and Gemma-2-9B
**Infrastructure**: Local M1 Pro + Vast.ai for larger models

**Tasks**:
- Extract probes from Llama-3-8B using expanded dataset
- Extract probes from Gemma-2-9B
- Compare AUROC across models
- Test if Phi-3 probes transfer to Llama/Gemma

**Expected finding**: If AUROC is perfect on all models → confirms artifacts hypothesis

**Script**: `src/probe_extraction_cross_model.py`

### 2.2 Test on Llama-3-70B (cloud)
**Infrastructure**: Vast.ai A100 instance

- Extract probes from larger model
- Compare layer-depth patterns
- Test steering effectiveness at scale

**Cost estimate**: ~$10-15 (3-4 hours on A100)

---

## Phase 3: Causal Validation

### 3.1 Activation Patching
**Reference**: Geiger et al. (2023) - Causal Abstraction

**Tasks**:
- Implement activation patching for probe directions
- Test interchange intervention:
  - Replace layer 12 activations from empathic text → task-focused text
  - Measure behavior change
- Compare patching vs steering (additive intervention)

**Expected finding**: If patching fails but steering works → suggests probe captures auxiliary features, not core mechanism

**Script**: `src/activation_patching.py`

### 3.2 Lexical Ablations
**Tasks**:
- Remove empathy keywords from text (help, care, support, etc.)
- Re-run probe projections
- Measure AUROC drop

**Expected finding**: If AUROC drops significantly → confirms lexical shortcut hypothesis

**Script**: `src/lexical_ablation.py`

### 3.3 Causal Mediation Analysis
**Reference**: Vig et al. (2020) - Causal Mediation

**Tasks**:
- Compute indirect effect (IE) of probe direction on output
- Compare IE across layers
- Test if layer 12 has higher causal influence

**Script**: `src/causal_mediation.py`

---

## Phase 4: Steering Robustness

### 4.1 Alpha Sweep Analysis
**Tasks**:
- Test finer-grained alpha values: [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
- Measure success rate curve
- Identify "elbow point" where steering begins

**Expected finding**: Clarify if α=20 is genuinely required or if lower values work with more samples

**Script**: `src/steering_alpha_sweep.py`

### 4.2 Task-Free Empathy Scenarios
**Tasks**:
- Generate scenarios with NO task conflict
  - Pure altruism: "You see someone drop their wallet. What do you do?"
  - No competing objective
- Test if steering still works

**Expected finding**: If steering fails without task-conflict → confirms task-attenuation hypothesis

**Script**: `src/generate_task_free_scenarios.py`

### 4.3 Negative Steering Analysis
**Tasks**:
- Extend negative steering experiments
- Verify it increases task-focus WITHOUT cruelty
- Measure sentiment shift

**Script**: Already done in v0, but add more samples

---

## Phase 5: Artifact Investigation

### 5.1 Random Baseline Refinement
**Tasks**:
- Generate 1000 random directions (instead of 100)
- Test if any random direction achieves AUROC > 0.95
- Compute p-value for empathy probe's AUROC

**Script**: `src/random_baseline_large_scale.py`

### 5.2 Probe Direction Analysis
**Tasks**:
- Visualize probe direction in activation space (PCA/t-SNE)
- Compare to task-related directions (e.g., "task completion" probe)
- Test cosine similarity between empathy and task probes

**Expected finding**: If empathy probe ≈ negative task probe → confirms task-attenuation

**Script**: `src/probe_visualization.py`

---

## Infrastructure Setup

### Vast.ai Configuration
```bash
# Search for instances
vastai search offers 'reliability > 0.95 num_gpus=1 gpu_name=A100 disk_space > 100'

# Launch instance
vastai create instance <INSTANCE_ID> \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --disk 100 \
  --jupyter

# SSH and setup
ssh -p <PORT> root@<IP>
git clone https://github.com/juancadile/empathy-probes.git
cd empathy-probes
pip install -r requirements.txt
pip install vllm transformers[torch]
```

### Model Access
- **Llama-3-70B-Instruct**: Hugging Face (ungated, no safety tuning)
- **Mixtral-8x7B-Instruct**: Hugging Face
- **GPT-OSS**: Check if available on Hugging Face

### Cost Budget
- Dataset generation: $10
- Cross-model validation: $15
- Large-scale experiments: $25
- **Total**: ~$50 for full strengthening

---

## Timeline

### Week 1 (Nov 18-24): Dataset Expansion
- Generate 150 new pairs
- Validate and integrate
- **Deliverable**: `data/dataset_expanded.json`

### Week 2 (Nov 25-Dec 1): Cross-Model Validation
- Test on Llama-3-8B, Gemma-2-9B (local)
- Test on Llama-3-70B (cloud)
- **Deliverable**: Cross-model AUROC comparison

### Week 3 (Dec 2-8): Causal Validation
- Implement activation patching
- Run lexical ablations
- **Deliverable**: Causal validation results

### Week 4 (Dec 9-15): Steering Robustness
- Alpha sweep
- Task-free scenarios
- **Deliverable**: Updated steering analysis

### Week 5 (Dec 16-22): Analysis & Write-up
- Artifact investigation
- Update paper to v1
- **Deliverable**: v1 draft for workshop submissions

---

## Success Metrics

### Minimum for v1:
- ✅ Dataset: 200+ pairs
- ✅ Models: Phi-3, Llama-3, Gemma-2 tested
- ✅ Causal validation: At least activation patching + lexical ablation
- ✅ Artifact investigation: Confirm/disconfirm lexical shortcuts

### Ideal for v1:
- ✅ All of the above
- ✅ Task-free scenarios tested
- ✅ Causal mediation analysis complete
- ✅ Cross-model steering transfer tested

---

## Next Steps

1. Set up Vast.ai account and test instance launch
2. Install vLLM and test Llama-3-70B inference
3. Write `src/generate_dataset_cloud.py`
4. Start Phase 1: Dataset expansion
