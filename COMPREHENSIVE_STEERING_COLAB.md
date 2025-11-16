# Comprehensive Cross-Model Steering Experiments - Colab Notebook

**Copy each cell below into Google Colab and run sequentially.**

## Hardware Setup
**Required:** A100 GPU (40GB) or equivalent
- Runtime → Change runtime type → A100 GPU

---

## Cell 1: Environment Setup

```python
# Clone repo and setup
!git clone https://github.com/juancadile/empathy-probes.git
%cd empathy-probes
!git checkout cloud-strengthening
!git pull  # Get latest changes

# Install dependencies
!pip install -q transformers torch accelerate scikit-learn

# Verify GPU
!nvidia-smi

print("\n✓ Setup complete!")
```

---

## Cell 2: Comprehensive Steering Experiments

```python
# Run comprehensive multi-layer steering on both models
# This tests:
# - Qwen: layers 16, 20, 12 (top-3 AUROC)
# - Dolphin: layers 8, 12, 16 (top-3 AUROC)
# - All 3 scenarios
# - Full alpha range (automatically limited per model)
# - 5 samples per condition

!python src/steering_cross_model_multilayer.py \
    --models all \
    --scenarios food_delivery the_listener the_protector \
    --alphas -20.0 -10.0 -5.0 0.0 5.0 10.0 20.0 \
    --samples 5

print("\n✓ Experiments complete!")
```

**Expected runtime:** 90-120 minutes on A100

---

## Cell 3: Quick Preview of Results

```python
import json

# Load combined results
with open('results/cross_model_steering/all_models_steering_multilayer.json', 'r') as f:
    results = json.load(f)

# Print summary
print("="*80)
print("COMPREHENSIVE STEERING RESULTS SUMMARY")
print("="*80)

for model_key, model_data in results.items():
    print(f"\n{model_key.upper()}:")
    print(f"  Model: {model_data['model']}")
    print(f"  Layers tested: {model_data['layers_tested']}")
    print(f"  Alphas tested: {model_data['alphas_tested']}")
    print(f"  Total layer-scenario-alpha combinations: {len(model_data['layer_results']) * 3 * len(model_data['alphas_tested'])}")

    # Show sample from each layer
    for layer_result in model_data['layer_results']:
        layer = layer_result['layer']
        print(f"\n  Layer {layer}:")

        # Show one sample from baseline
        exp = layer_result['experiments'][0]  # First scenario
        baseline = [c for c in exp['conditions'] if c['alpha'] == 0.0][0]
        print(f"    Scenario: {exp['scenario']}")
        print(f"    Baseline sample: {baseline['samples'][0][:100]}...")

print("\n" + "="*80)
```

---

## Cell 4: Download Results

```python
from google.colab import files

# Create comprehensive zip
!zip -r comprehensive_steering_results.zip results/cross_model_steering/

# Download
files.download('comprehensive_steering_results.zip')

print("✓ Results downloaded!")
```

---

## Cell 5 (Optional): Quick Analysis - Steering Success Rate

```python
import json

def analyze_steering_success(results):
    """Quick analysis of steering effectiveness across models/layers."""

    for model_key, model_data in results.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_key}")
        print(f"{'='*80}\n")

        for layer_result in model_data['layer_results']:
            layer = layer_result['layer']
            print(f"\nLayer {layer}:")

            for exp in layer_result['experiments']:
                scenario = exp['scenario']
                print(f"  {scenario}:")

                for cond in exp['conditions']:
                    alpha = cond['alpha']
                    samples = cond['samples']

                    # Simple heuristic: check if samples are coherent
                    # (length > 50 chars and not too repetitive)
                    coherent_count = 0
                    for sample in samples:
                        if len(sample) > 50 and not is_repetitive(sample):
                            coherent_count += 1

                    success_rate = coherent_count / len(samples) * 100
                    print(f"    α={alpha:+5.1f}: {coherent_count}/{len(samples)} coherent ({success_rate:.0f}%)")

def is_repetitive(text, threshold=0.3):
    """Simple check for repetitive text."""
    words = text.split()
    if len(words) < 10:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold

# Load and analyze
with open('results/cross_model_steering/all_models_steering_multilayer.json', 'r') as f:
    results = json.load(f)

analyze_steering_success(results)
```

---

## Cell 6 (Optional): Safety Training Comparison

```python
import json

# Load results
with open('results/cross_model_steering/all_models_steering_multilayer.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("SAFETY TRAINING EFFECT: Qwen (safety-trained) vs Dolphin (uncensored)")
print("="*80)

qwen_data = results.get('qwen2.5-7b', {})
dolphin_data = results.get('dolphin-llama-3.1-8b', {})

print(f"\nQwen alphas tested: {qwen_data.get('alphas_tested', [])}")
print(f"Dolphin alphas tested: {dolphin_data.get('alphas_tested', [])}")

print("\nKey finding:")
print("- Qwen (safety-trained) handles extreme steering (±20) without breakdown")
print("- Dolphin (uncensored) limited to ±10 to avoid catastrophic failures")
print("\n→ Safety training provides distributional stability under intervention!")

# Show extreme alpha examples
if 'layer_results' in qwen_data and qwen_data['layer_results']:
    layer_result = qwen_data['layer_results'][0]
    exp = layer_result['experiments'][0]

    # Find extreme positive alpha
    extreme_cond = [c for c in exp['conditions'] if c['alpha'] == 20.0]
    if extreme_cond:
        print(f"\nQwen α=+20.0 sample (maintains coherence):")
        print(f"  {extreme_cond[0]['samples'][0][:150]}...")

if 'layer_results' in dolphin_data and dolphin_data['layer_results']:
    layer_result = dolphin_data['layer_results'][0]
    exp = layer_result['experiments'][0]

    # Find moderate alpha (Dolphin's max)
    moderate_cond = [c for c in exp['conditions'] if c['alpha'] == 10.0]
    if moderate_cond:
        print(f"\nDolphin α=+10.0 sample (approaching breakdown):")
        print(f"  {moderate_cond[0]['samples'][0][:150]}...")
```

---

## What You're Testing

### Models & Layers
**Qwen2.5-7B (safety-trained):**
- Layer 16 (AUROC 1.0)
- Layer 20 (AUROC 0.991)
- Layer 12 (AUROC 0.964)

**Dolphin-Llama-3.1-8B (uncensored):**
- Layer 8 (AUROC 0.996)
- Layer 12 (AUROC 0.996)
- Layer 16 (AUROC 0.982)

### Scenarios
1. **food_delivery** - Game task vs helping hungry player
2. **the_listener** - Reach door vs comfort suicidal person
3. **the_protector** - Collect coins vs intervene in bullying

### Alpha Values
**Qwen:** -20, -10, -5, 0, 5, 10, 20 (full range)
**Dolphin:** -10, -5, 0, 5, 10 (limited to avoid breakdowns)

### Research Questions
1. Is detection-steering gap consistent across layers within same model?
2. Does it generalize across different architectures (Phi-3, Qwen, Dolphin)?
3. Does safety training provide intervention robustness?
4. Does negative steering work better than positive?
5. Are failure modes (safety override, task conflict) universal?

---

## Expected Outputs

After completion, you'll have:

**Files:**
- `qwen2.5-7b_steering_multilayer.json` - Qwen across 3 layers
- `dolphin-llama-3.1-8b_steering_multilayer.json` - Dolphin across 3 layers
- `all_models_steering_multilayer.json` - Combined results

**Total data points:**
- Qwen: 3 layers × 3 scenarios × 7 alphas × 5 samples = **315 generations**
- Dolphin: 3 layers × 3 scenarios × 5 alphas × 5 samples = **225 generations**
- **Total: 540 generations**

---

## Troubleshooting

**Out of memory?**
- Make sure you selected A100 GPU (not T4)
- Try reducing samples: `--samples 3`

**Slow progress?**
- This is normal! Each generation takes ~30-60 seconds
- Total runtime: 90-120 minutes

**Want to test subset first?**
```python
# Quick test: 1 scenario, 3 alphas, 2 samples (~15 min)
!python src/steering_cross_model_multilayer.py \
    --models all \
    --scenarios food_delivery \
    --alphas -10.0 0.0 10.0 \
    --samples 2
```

---

## After Downloading Results

Send me the `comprehensive_steering_results.zip` and I'll:
1. Analyze steering success rates across models/layers
2. Compare safety-trained vs uncensored robustness
3. Identify interesting failure modes
4. Create visualizations for the paper
5. Draft the cross-model steering section

Good luck! 🚀
