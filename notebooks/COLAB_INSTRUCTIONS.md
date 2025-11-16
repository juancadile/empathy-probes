# Running Cross-Model Experiments on Google Colab

## Setup

### 1. Create New Colab Notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Set GPU Runtime

- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **T4 GPU** (or better)
- Click **Save**

### 3. Run Setup Cells

```python
# Cell 1: Check GPU
!nvidia-smi
```

```python
# Cell 2: Clone repository
!git clone https://github.com/YOUR_USERNAME/empathy-action-probes.git
%cd empathy-action-probes
```

```python
# Cell 3: Install dependencies
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn torch
```

### 4. Run Experiments

```python
# Cell 4: Run both models (~1.5 hours)
!python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
```

**OR** run individually:

```python
# Option A: Qwen2.5-7B only (~40 min)
!python src/probe_extraction_cross_model.py --models qwen2.5-7b
```

```python
# Option B: Dolphin-Llama-3.1-8B only (~45 min)
!python src/probe_extraction_cross_model.py --models dolphin-llama-3.1-8b
```

### 5. View Results

```python
# Cell 5: Load and display results
import json
import pandas as pd

with open('results/cross_model_validation/all_models_results.json', 'r') as f:
    results = json.load(f)

# Create comparison table
rows = []
for model_key, model_results in results.items():
    for layer, metrics in model_results['layers'].items():
        rows.append({
            'Model': model_key,
            'Layer': layer,
            'AUROC': f"{metrics['auroc']:.3f}",
            'Accuracy': f"{metrics['accuracy']:.1%}",
            'Separation': f"{metrics['separation']:.2f}"
        })

df = pd.DataFrame(rows)
print("\n=== Cross-Model Probe Performance ===\n")
print(df.to_string(index=False))

# Find best layer per model
print("\n=== Best Layer Per Model ===\n")
best_layers = df.loc[df.groupby('Model')['AUROC'].idxmax()]
print(best_layers[['Model', 'Layer', 'AUROC', 'Accuracy']].to_string(index=False))
```

### 6. Download Results

```python
# Cell 6: Zip and download
!zip -r cross_model_results.zip results/cross_model_validation/

from google.colab import files
files.download('cross_model_results.zip')
```

---

## Models

**Current phase:**
1. **Qwen2.5-7B-Instruct** (7B, safety-trained)
2. **Dolphin-Llama-3.1-8B** (8B, uncensored)

**Future** (based on results):
- GPT-oss-20b (20B)
- Abliterated GPT-oss-20b

---

## Expected Time

| Model | Time on T4 GPU |
|-------|----------------|
| Qwen2.5-7B | ~40 minutes |
| Dolphin-Llama-3.1-8B | ~45 minutes |
| **Total** | ~1.5 hours |

---

## Troubleshooting

**Out of memory?**
- Use smaller batch size (edit `probe_extraction_cross_model.py`)
- Or run models individually instead of together

**Colab disconnecting?**
- Run cells one at a time instead of all at once
- Use Colab Pro for longer runtimes

**Model download slow?**
- This is normal - models are 7-8GB each
- Downloads are cached, so re-runs are faster

---

## What You'll Get

Results saved to `results/cross_model_validation/`:
- `qwen2.5-7b_results.json` - Metrics for Qwen
- `dolphin-llama-3.1-8b_results.json` - Metrics for Dolphin
- `all_models_results.json` - Combined summary
- `{model}_layer{X}_probe.npy` - Probe direction vectors

---

## Next Steps After Running

1. Compare AUROC across models
2. Analyze: Does uncensored (Dolphin) differ from safety-trained (Qwen)?
3. If results look good → Run steering experiments
4. If results are mixed → Investigate why

---

## Questions?

See `notebooks/README_CLOUD.md` for more details.
