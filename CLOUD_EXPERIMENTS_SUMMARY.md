# Cloud Experiments: What We've Set Up

## Created Files

### 1. Scripts
- **`scripts/setup_cloud_experiment.sh`** - Automated setup for vast.ai/Colab
- **`src/probe_extraction_cross_model.py`** - Main experiment script (3 models)

### 2. Notebooks
- **`notebooks/cloud_experiments.ipynb`** - Google Colab notebook (click & run)
- **`notebooks/README_CLOUD.md`** - Detailed instructions

### 3. Documentation
- **`paper/paper.tex`** - Added "Safety guardrails effect" to Future Work (line 254)

---

## Models to Test (Current Phase)

| Model | Size | Type | Purpose |
|-------|------|------|---------|
| **Qwen2.5-7B-Instruct** | 7B | Safety-trained | Latest strong open model |
| **Dolphin-Llama-3.1-8B** | 8B | Uncensored | No safety guardrails |

**Future models** (based on current results):
- **GPT-oss-20b** (20B, OpenAI OSS)
- **Abliterated GPT-oss-20b** (safety layers removed)
- Comparison: vanilla vs abliterated to study safety training effects

---

## Quick Start Options

### Option 1: Google Colab (Easiest, Free)
1. Open `notebooks/cloud_experiments.ipynb` in Colab
2. Set GPU to T4 (Runtime → Change runtime type)
3. Click "Run all" (~2.5 hours)
4. Download results zip

### Option 2: Vast.ai (Faster, ~$2)
```bash
# See notebooks/README_CLOUD.md for full instructions
bash scripts/setup_cloud_experiment.sh
python src/probe_extraction_cross_model.py --models all
```

### Option 3: Run Locally (if you have GPU)
```bash
cd /Users/juancadile/Documents/samuel-anthropic/empathy-action-probes
python src/probe_extraction_cross_model.py --models all
```

---

## What Gets Tested

For each model:
1. **Extract probes** from layers {8, 12, 16, 20, 24}
2. **Validate on test set**: AUROC, accuracy, separation
3. **Save probe directions** as `.npy` files
4. **Generate comparison table** across all models

---

## Expected Outcomes

### Success Scenario
- All models achieve AUROC > 0.85 → Empathy is model-agnostic
- Middle layers (12-16) perform best → Consistent with Phi-3
- Dolphin (uncensored) shows similar performance → Empathy ≠ safety training

### Interesting Findings
- GPT-oss-20b outperforms smaller models → Size helps
- Dolphin differs significantly → Safety training affects empathy encoding
- Different layers peak → Architecture-dependent representations

### Failure Scenario
- AUROC < 0.75 on other models → Phi-3's perfect score was artifact
- No consistent pattern → Empathy not linearly represented

---

## Timeline

1. **Now**: Run experiments (2-3 hours Colab or 1 hour vast.ai)
2. **After results**: Analyze cross-model performance
3. **Next**:
   - If successful → Run steering experiments on best models
   - If mixed → Investigate why some models work better
   - Add results to paper as cross-model validation

---

## Cost

| Platform | Time (2 models) | Cost |
|----------|-----------------|------|
| **Google Colab (free tier)** | ~1.5 hours | $0 |
| **Vast.ai (A100)** | ~30 min | ~$1 |
| **Local (if you have GPU)** | ~1.5 hours | $0 |

---

## Next Steps After Running

1. **Analyze results**:
   ```python
   # See notebooks/cloud_experiments.ipynb, cell "View Results"
   ```

2. **Update paper** with cross-model validation section

3. **Run steering tests** on best-performing models

4. **Compare guardrails effect**:
   - Dolphin (uncensored) vs Llama-3.1-Instruct (safety-trained)
   - Base vs Instruct variants

---

## Files You Need to Update

Before running on GitHub/Colab, update:
- `scripts/setup_cloud_experiment.sh` line 29: Replace `YOUR_USERNAME` with your GitHub username
- `notebooks/cloud_experiments.ipynb` cell 2: Replace `YOUR_USERNAME` with your GitHub username

Or just run locally first to test!

---

## Questions?

See `notebooks/README_CLOUD.md` for detailed troubleshooting and instructions.
