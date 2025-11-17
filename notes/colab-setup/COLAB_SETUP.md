# Google Colab Setup - Copy-Paste Ready

## Before Starting

1. **Push your latest code to GitHub**:
   ```bash
   cd /Users/juancadile/Documents/samuel-anthropic/empathy-action-probes
   git add .
   git commit -m "Add cross-model validation scripts"
   git push origin cloud-strengthening
   ```

2. **Go to**: [colab.research.google.com](https://colab.research.google.com)

3. **Create new notebook**

4. **Set GPU**: Runtime → Change runtime type → **T4 GPU** → Save

---

## Copy-Paste These Cells (in order)

### Cell 1: Check GPU
```python
!nvidia-smi
```
**Expected**: Should show Tesla T4 GPU with 15GB memory

---

### Cell 2: Clone Repository
```python
!git clone https://github.com/juancadile/empathy-probes.git
%cd empathy-probes

# Checkout the right branch
!git checkout cloud-strengthening
!git pull origin cloud-strengthening

# Verify files exist
!ls src/probe_extraction_cross_model.py
```

---

### Cell 3: Install Dependencies
```python
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn torch
print("✓ Dependencies installed")
```

---

### Cell 4: Verify Data Exists
```python
!ls data/contrastive_pairs/
!wc -l data/contrastive_pairs/train_pairs.jsonl
!wc -l data/contrastive_pairs/test_pairs.jsonl
```
**Expected**: Should show 35 train pairs, 15 test pairs

---

### Cell 5: Run Experiments (Both Models)
```python
# This will take ~1.5 hours
# You'll see progress bars for each model/layer
!python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
```

**OR** run one model at a time:

```python
# Option A: Just Qwen (~40 min)
!python src/probe_extraction_cross_model.py --models qwen2.5-7b
```

```python
# Option B: Just Dolphin (~45 min) - run after Qwen completes
!python src/probe_extraction_cross_model.py --models dolphin-llama-3.1-8b
```

---

### Cell 6: View Results
```python
import json
import pandas as pd

# Load results
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

# Compare to Phi-3 baseline
print("\n=== Comparison to Phi-3 Baseline ===")
print("Phi-3 Layer 12: AUROC 1.000, Accuracy 100%")
print(f"Qwen best: {df[df['Model']=='qwen2.5-7b']['AUROC'].max()}")
print(f"Dolphin best: {df[df['Model']=='dolphin-llama-3.1-8b']['AUROC'].max()}")
```

---

### Cell 7: Download Results
```python
!zip -r cross_model_results.zip results/cross_model_validation/

from google.colab import files
files.download('cross_model_results.zip')
```

---

## Troubleshooting

**Colab disconnects during run?**
- Click the cell → It will resume from where it stopped
- Or run models one at a time (Cell 5 alternatives)

**Out of memory?**
- Shouldn't happen with T4 (15GB) for these 7-8B models
- If it does, restart runtime and run one model at a time

**Clone fails?**
- Make sure you pushed to GitHub first
- Try: `!git clone https://github.com/juancadile/empathy-probes.git --branch cloud-strengthening`

**Missing files?**
- Make sure you're on the right branch: `!git checkout cloud-strengthening`
- Verify: `!ls src/probe_extraction_cross_model.py`

---

## Expected Output

After Cell 5 completes, you should see something like:

```
=== Processing qwen2.5-7b ===
Loading model...
✓ Loaded Qwen/Qwen2.5-7B-Instruct

--- Layer 8 ---
Extracting training activations... 100%
Computing probe direction...
Extracting test activations... 100%
Validating probe...
AUROC: 0.987, Accuracy: 93.3%

--- Layer 12 ---
...
AUROC: 0.996, Accuracy: 100%

✓ Completed qwen2.5-7b

=== Processing dolphin-llama-3.1-8b ===
...
```

---

## Time Estimate

- Cell 1-4: ~2 minutes
- Cell 5: ~1.5 hours (both models) or ~40-45 min each
- Cell 6-7: ~1 minute

**Total**: ~1.5-2 hours

---

## What You'll Get

A zip file containing:
- `qwen2.5-7b_results.json` - All metrics for Qwen
- `dolphin-llama-3.1-8b_results.json` - All metrics for Dolphin
- `all_models_results.json` - Combined summary
- Probe `.npy` files for each model/layer

---

## After Downloading

Unzip locally and analyze:
```bash
cd ~/Downloads
unzip cross_model_results.zip
cat cross_model_results/all_models_results.json
```

Then update your paper with findings!
