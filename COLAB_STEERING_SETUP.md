# Cross-Model Steering Experiments - Colab Setup

This guide shows how to run steering experiments on Qwen2.5-7B and Dolphin-Llama-3.1-8B using Google Colab.

## Research Questions

1. **Architecture dependence**: Does steering success vary by model architecture?
2. **Safety training effect**: Does uncensored Dolphin show better/worse steering than safety-trained models?
3. **Detection-steering gap**: Is the 30-40% steering success rate consistent across models?

## Colab Setup

### 1. Environment Setup

```python
# Clone repo and install dependencies
!git clone https://github.com/juancadile/empathy-probes.git
%cd empathy-probes
!git checkout cloud-strengthening
!pip install -q transformers torch accelerate scikit-learn
```

### 2. Run Steering Experiments

#### Option A: Test Both Models (Recommended)
```python
!python src/steering_cross_model.py \
    --models all \
    --scenarios food_delivery the_listener the_protector \
    --alphas 0.0 5.0 10.0 20.0 \
    --samples 5
```

#### Option B: Test Individual Models
```python
# Qwen only
!python src/steering_cross_model.py --models qwen2.5-7b

# Dolphin only
!python src/steering_cross_model.py --models dolphin-llama-3.1-8b
```

#### Option C: Quick Test (1 scenario, fewer samples)
```python
!python src/steering_cross_model.py \
    --models all \
    --scenarios food_delivery \
    --alphas 0.0 10.0 \
    --samples 3
```

### 3. Download Results

```python
# Zip and download
!zip -r steering_results.zip results/cross_model_steering/
from google.colab import files
files.download('steering_results.zip')
```

### 4. View Results in Colab

```python
import json

# Load results
with open('results/cross_model_steering/all_models_steering.json', 'r') as f:
    results = json.load(f)

# Print summary
for model_key, model_results in results.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_key}")
    print(f"{'='*80}")

    for exp in model_results['experiments']:
        print(f"\nScenario: {exp['scenario']} - {exp['title']}")
        for cond in exp['conditions']:
            alpha = cond['alpha']
            samples = cond['samples']
            print(f"\n  Alpha={alpha}:")
            for i, sample in enumerate(samples, 1):
                print(f"    {i}. {sample[:100]}...")
```

## Expected Runtime

- **Qwen2.5-7B**: ~20-30 minutes (3 scenarios × 4 alphas × 5 samples)
- **Dolphin-Llama-3.1-8B**: ~25-35 minutes (similar)
- **Total (both models)**: ~50-70 minutes

## GPU Requirements

- **Minimum**: T4 (15GB VRAM) - should work but might be tight
- **Recommended**: A100 (40GB) or similar for faster runtime

## Results Files

After running, you'll have:
- `results/cross_model_steering/qwen2.5-7b_steering.json`
- `results/cross_model_steering/dolphin-llama-3.1-8b_steering.json`
- `results/cross_model_steering/all_models_steering.json` (combined)

## Hypothesis Testing

After experiments, we can analyze:

1. **Steering success rate by model**:
   - Does Dolphin (uncensored) have higher success? (hypothesis: yes, less safety resistance)
   - Or lower success? (hypothesis: yes, doesn't encode task-sacrifice as strongly)

2. **Alpha sensitivity**:
   - Do all models show similar alpha curves?
   - Is there a "sweet spot" alpha value?

3. **Scenario consistency**:
   - Do some scenarios steer better across all models?
   - Does task conflict affect all models equally?

## Next Steps After Running

1. Commit results to repo
2. Compare with Phi-3 steering results
3. Update paper with cross-model steering findings
4. Decide if detection-steering gap is architecture-agnostic or model-specific
