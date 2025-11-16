# Ready to Run: Cross-Model Experiments

## ✅ What's Ready

### Current Phase: 2 Models
1. **Qwen2.5-7B-Instruct** (safety-trained)
2. **Dolphin-Llama-3.1-8B** (uncensored, no guardrails)

### Future Phase (noted in paper)
3. **GPT-oss-20b** (OpenAI OSS, 20B)
4. **Abliterated GPT-oss-20b** (safety layers removed)

---

## 🚀 How to Run

### Option 1: Local (if you have GPU)

```bash
cd /Users/juancadile/Documents/samuel-anthropic/empathy-action-probes

# Quick test with one model first (~40 min)
bash scripts/test_single_model.sh
# → Choose 1 for Qwen2.5-7B

# If test succeeds, run both models (~1.5 hours)
python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
```

### Option 2: Google Colab (Free, Easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Follow `notebooks/COLAB_INSTRUCTIONS.md` step-by-step
3. Total time: ~1.5 hours on free T4 GPU

### Option 3: Vast.ai (Fastest, ~$1)

```bash
# See notebooks/README_CLOUD.md for full instructions
bash scripts/setup_cloud_experiment.sh
python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
```

---

## 📊 What You'll Get

Results in `results/cross_model_validation/`:
- Probe performance (AUROC, accuracy) for each model × each layer
- Comparison table: Does Qwen vs Dolphin differ?
- Probe direction vectors (`.npy` files) for steering experiments

---

## 🎯 Research Questions

1. **Cross-architecture generalization**: Do probes work on Qwen (different from Phi-3)?
2. **Safety guardrails effect**: Does uncensored Dolphin differ from safety-trained Qwen?
3. **Layer consistency**: Do middle layers (12-16) still perform best?

---

## 📝 What's Updated

### Code
- ✅ `src/probe_extraction_cross_model.py` - Runs on 2 models
- ✅ `scripts/test_single_model.sh` - Quick local test
- ✅ `scripts/setup_cloud_experiment.sh` - Cloud setup

### Documentation
- ✅ `notebooks/COLAB_INSTRUCTIONS.md` - Step-by-step Colab guide
- ✅ `notebooks/README_CLOUD.md` - Detailed cloud instructions
- ✅ `CLOUD_EXPERIMENTS_SUMMARY.md` - Overview
- ✅ `READY_TO_RUN.md` (this file)

### Paper
- ✅ Added "Safety guardrails effect" to Future Work
- ✅ Mentioned GPT-oss-20b and abliterated variants
- ✅ Updated cross-model validation section

---

## ⏱️ Time & Cost

| Platform | Time | Cost |
|----------|------|------|
| **Local (with GPU)** | ~1.5 hours | $0 |
| **Colab (free T4)** | ~1.5 hours | $0 |
| **Vast.ai (A100)** | ~30 min | ~$1 |

---

## 🔥 Recommended: Start with Local Test

```bash
# Takes ~40 min, tests everything works
cd /Users/juancadile/Documents/samuel-anthropic/empathy-action-probes
bash scripts/test_single_model.sh
```

Choose option 1 (Qwen2.5-7B). If this succeeds, you know everything works and can run both models or move to Colab.

---

## 📈 Next Steps After Results

1. **Analyze**: Do models show similar AUROC? Different layers?
2. **Compare**: Safety-trained (Qwen) vs uncensored (Dolphin)
3. **Update paper**: Add cross-model validation results
4. **Decide**: Based on results, run GPT-oss experiments or not

---

## 💡 Key Insight

By testing Dolphin (uncensored) vs Qwen (safety-trained), you'll see if empathy probes:
- **Option A**: Capture genuine empathy (both models similar)
- **Option B**: Capture safety-trained patterns (models differ significantly)

This directly addresses the "Safety guardrails effect" future work you added to the paper!

---

## Questions?

See `notebooks/COLAB_INSTRUCTIONS.md` or `notebooks/README_CLOUD.md`
