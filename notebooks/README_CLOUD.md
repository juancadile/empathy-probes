# Cloud Experiments: Cross-Model Validation

## Overview

This folder contains scripts for running empathy probe experiments across multiple models on cloud infrastructure (Colab/vast.ai).

## Models (Current Phase)

1. **Qwen2.5-7B-Instruct** - Latest strong open model (Qwen Team, 2024)
2. **Dolphin-Llama-3.1-8B** - Uncensored Llama variant (no safety guardrails)

**Future models** (pending current results):
- GPT-oss-20b (OpenAI OSS, 20B params)
- Abliterated GPT-oss-20b (safety layers removed)

## Quick Start

### Option A: Google Colab (Free)

1. Open `cloud_experiments.ipynb` in Colab
2. Set Runtime → Change runtime type → T4 GPU
3. Run all cells (~2-3 hours total)

### Option B: Vast.ai (Faster, ~$5)

```bash
# 1. Find instance
vastai search offers 'reliability > 0.95 num_gpus=1 gpu_name=A100 disk_space > 100'

# 2. Launch instance
vastai create instance <INSTANCE_ID> \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --disk 100

# 3. SSH into instance
ssh -p <PORT> root@<IP>

# 4. Run setup script
git clone https://github.com/YOUR_USERNAME/empathy-action-probes.git
cd empathy-action-probes
bash scripts/setup_cloud_experiment.sh

# 5. Run experiments
python src/probe_extraction_cross_model.py --models all
```

## Expected Results

Based on Phi-3 baseline:
- **Detection**: AUROC 0.85-1.00 across models
- **Best layers**: Middle layers (12-16) likely optimal
- **Cross-architecture generalization**: Should see consistent empathy representations

## Output Files

All results saved to `results/cross_model_validation/`:
- `{model}_results.json` - Metrics per layer
- `{model}_layer{X}_probe.npy` - Probe direction vectors
- `all_models_results.json` - Combined summary

## Time Estimates

| Model | Colab (T4) | Vast.ai (A100) | Cost (vast.ai) |
|-------|------------|----------------|----------------|
| Qwen2.5-7B | ~40 min | ~15 min | ~$0.40 |
| Dolphin-Llama-3.1-8B | ~45 min | ~18 min | ~$0.50 |
| **Total (2 models)** | ~1.5 hours | ~30 min | ~$0.90 |

## Research Questions

1. **Do empathy probes generalize across architectures?**
   - Compare AUROC: Phi-3 vs Qwen vs Llama vs GPT-oss

2. **Do safety guardrails affect empathy encoding?**
   - Compare Dolphin (uncensored) vs Llama-3.1-Instruct

3. **Does model size matter?**
   - 7B (Qwen) vs 8B (Llama) vs 20B (GPT-oss)

## Future Extensions

- **Probe transfer**: Test if Phi-3 probe works on other models
- **Steering experiments**: Run steering tests on best-performing models
- **Dataset expansion**: Generate 150+ new pairs using these models
- **Guardrails study**: Compare base vs instruct vs uncensored variants

## Troubleshooting

**Out of memory?**
- Reduce batch size in script
- Use 8-bit quantization (add `load_in_8bit=True` to model loading)

**Model download failing?**
- Check Hugging Face access token for gated models
- Try different mirror: `export HF_ENDPOINT=https://hf-mirror.com`

**Colab disconnecting?**
- Run cells individually instead of "Run all"
- Use Colab Pro for longer runtimes

## Contact

See main README for project details.
