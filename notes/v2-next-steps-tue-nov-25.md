# V2 Next Steps - Tuesday Nov 25, 2024

## Current Status

### Dataset Generation Complete (or in progress)
- **Qwen-32B**: 500/500 pairs - COMPLETE, downloaded locally
- **Gemma-27B**: 500/500 pairs - COMPLETE, downloaded locally
- **Llama-70B-FP8**: ~20/500 pairs - IN PROGRESS (~2.4 hours remaining)

### Total V2 Dataset
- 2,500 closed-source pairs (Claude Sonnet-4, Haiku, GPT-4o, GPT-5.1, Gemini 2.5 Flash)
- 1,500 open-source pairs (Qwen-32B, Gemma-27B, Llama-70B)
- **4,000 total contrastive pairs**

---

## Key Decisions Made

### FP8 Quantization for Llama-70B
- Used `neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8` (~70GB) instead of full BF16 (~140GB)
- **Reasoning**: FP8 is fine for text generation; full precision better for interpretability experiments

**Plan**:
- Use FP8 quantized for dataset generation (current task)
- Use full precision BF16 for probing, causal scrubbing, SAE work (future experiments with more compute)

---

## Activation Storage Strategy

### Original Approach (what we did)
- Generated text completions only
- Did NOT store activations during generation

### Better Approach for Future (with 100TB storage available)
Extract ALL layer activations upfront, then:
1. Run AUROC/steering sweeps across all layers
2. Keep only useful layers, delete rest

### Storage Estimates (trivial with 100TB)
For Llama-70B (80 layers):
- Mean-pooled activation per sample: 8192 dims × 2 bytes = 16KB
- Per layer, 500 pairs × 2: ~16MB
- All 80 layers: ~1.3GB per model
- Token-level (not mean-pooled): ~50-100GB per model

### Next Steps for Activations
After Llama-70B generation completes:
1. Run a **second pass** to extract activations from all layers
2. Store everything (storage is cheap)
3. Run layer sweeps to find optimal layers for probing/steering
4. Delete unused layer data if needed

---

## Instance Details

- **H100 80GB**: 209.20.159.239
- **SSH key**: `/ssh/juan.pem`
- **HuggingFace token**: configured on instance

---

## Immediate TODOs

1. [ ] Wait for Llama-70B generation to complete (~2.4 hours)
2. [ ] Download Llama-70B data locally
3. [ ] Create activation extraction script that saves ALL layers
4. [ ] Run activation extraction for all 3 open-source models
5. [ ] Verify dataset integrity (500 pairs per model)

---

## Future Experiments (from v2-plan-5.1.md)

### Tier 1 - Frontier Models
- Llama-3.1-70B (full precision for interpretability)
- Gemma-2-27B (base + instruct comparison)
- GPT-OSS-20B or similar

### Key Experiments
- Linear probes across all layers
- SAE training on selected layers
- Steering experiments (alpha sweeps)
- Causal patching / path patching
- Scaling law analysis (2B -> 9B -> 27B -> 70B)
- Cross-family representational divergence

### Compute Requirements
- SAE training: Use full precision (not FP8)
- Causal patching: Use full precision
- Probing: FP8 may be acceptable but full precision preferred
