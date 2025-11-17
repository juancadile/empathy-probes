# Colab Quickstart - Cross-Model Steering Experiments

## What You'll Get
Test whether steering works on Qwen2.5-7B and Dolphin-Llama-3.1-8B, comparing to Phi-3 results.

**Key questions:**
- Does uncensored Dolphin steer better or worse than safety-trained models?
- Is the detection-steering gap (perfect detection, 30-40% steering) universal?

---

## Step 1: Open Google Colab
Go to: https://colab.research.google.com/

**GPU Setup:** Runtime → Change runtime type → T4 GPU (free tier works!)

---

## Step 2: Setup Environment

Copy-paste this into the first cell:

```python
# Clone repo and checkout branch
!git clone https://github.com/juancadile/empathy-probes.git
%cd empathy-probes
!git checkout cloud-strengthening

# Install dependencies
!pip install -q transformers torch accelerate scikit-learn
```

Run it (Shift+Enter). Takes ~1 minute.

---

## Step 3: Run Steering Experiments

### Quick Test (recommended first):
```python
# Test just one scenario on both models (~15 minutes)
!python src/steering_cross_model.py \
    --models all \
    --scenarios food_delivery \
    --alphas 0.0 10.0 \
    --samples 3
```

### Full Experiment:
```python
# All 3 scenarios, 4 alpha values, 5 samples (~60 minutes)
!python src/steering_cross_model.py \
    --models all \
    --scenarios food_delivery the_listener the_protector \
    --alphas 0.0 5.0 10.0 20.0 \
    --samples 5
```

---

## Step 4: View Results

```python
import json

# Load results
with open('results/cross_model_steering/all_models_steering.json', 'r') as f:
    results = json.load(f)

# Print summary
for model_key, data in results.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {model_key}")
    print(f"{'='*80}\n")

    for exp in data['experiments']:
        print(f"Scenario: {exp['scenario']}")
        for cond in exp['conditions']:
            print(f"\n  Alpha {cond['alpha']}:")
            for i, sample in enumerate(cond['samples'][:2], 1):  # Show first 2
                print(f"    {i}. {sample[:150]}...")
        print("\n" + "-"*80)
```

---

## Step 5: Download Results

```python
# Create zip and download
!zip -r steering_results.zip results/cross_model_steering/
from google.colab import files
files.download('steering_results.zip')
```

---

## Expected Output

You'll get 3 JSON files:
- `qwen2.5-7b_steering.json` - Qwen steering results
- `dolphin-llama-3.1-8b_steering.json` - Dolphin steering results
- `all_models_steering.json` - Combined

Each contains:
- Baseline (alpha=0.0) generations
- Steered generations at different strengths (5.0, 10.0, 20.0)
- 5 samples per condition for statistical analysis

---

## What to Look For

**Hypothesis 1: Dolphin steers better (less safety resistance)**
- Check if Dolphin shows more empathic behavior at alpha=10.0 vs Qwen

**Hypothesis 2: Dolphin steers worse (weaker task-sacrifice encoding)**
- Check if Dolphin shows LESS empathy change vs safety-trained models

**Hypothesis 3: Detection-steering gap is universal**
- Both models show perfect detection (AUROC 0.996-1.0)
- Do both also show similar low steering success (30-40%)?

---

## Troubleshooting

**Out of memory?**
- Use T4 GPU (not CPU)
- Try one model at a time: `--models qwen2.5-7b`

**Slow?**
- Use quick test (1 scenario, 2 alphas, 3 samples)
- Or upgrade to Colab Pro for A100 GPU

**Can't find files?**
- Make sure you're in the `empathy-probes` directory
- Run `%pwd` to check current directory

---

## Next Steps After Running

1. Send me the `steering_results.zip`
2. We'll analyze success rates across models
3. Update paper with cross-model steering findings
4. Determine if the detection-steering gap is architecture-specific or universal
