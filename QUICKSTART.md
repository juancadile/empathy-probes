# Quick Start Guide

Get started with empathy probe extraction in 5 steps.

## Prerequisites

- Python 3.9+
- 12GB+ RAM (16GB recommended for M1 Pro)
- API keys for Claude and/or GPT-4 (for dataset generation)

## Setup (5 minutes)

```bash
# 1. Navigate to project directory
cd empathy-action-probes

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys (for dataset generation)
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Run Full Pipeline (~1-2 hours)

### Option A: Full Pipeline (Recommended)

```bash
python run_full_pipeline.py
```

This will:
1. Generate 90 contrastive pairs using Claude + GPT-4
2. Extract empathy probes from Gemma-2-9B across 5 layers
3. Validate probes on test set (compute AUROC)
4. Predict EIA behavioral scores
5. Run steering experiments

### Option B: Skip Dataset Generation (Use Pre-Generated Data)

If you already have data or want to test with existing dataset:

```bash
python run_full_pipeline.py --skip-generation
```

### Option C: Skip Steering (Faster)

If you want results quickly without steering experiments:

```bash
python run_full_pipeline.py --skip-steering
```

## Run Individual Steps

```bash
# Step 1: Generate dataset only (~30-45 min)
python src/generate_dataset.py

# Step 2: Extract probes only (~30-40 min)
python src/probe_extraction.py

# Step 3: Predict EIA scores (~10 min)
python src/eia_evaluator.py

# Step 4: Run steering experiments (~15 min)
python src/steering.py
```

## Check Results

Results are saved to `results/` directory:

```bash
# View validation results
cat results/validation_auroc.json

# View EIA prediction results
cat results/eia_correlation.json

# View steering examples
cat results/steering_examples.json
```

## Analyze Results (Interactive)

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_analyze_results.ipynb
```

This will generate visualizations:
- Layer-wise performance plots
- EIA correlation scatter plots
- Probe vector analysis
- Steering comparison tables

## Expected Results

**Validation (Target: >75% AUROC)**
- Best layer: 16 (middle layers typically work best)
- AUROC: 0.75-0.85 (target: >0.75)

**EIA Prediction (Target: r > 0.4)**
- Pearson correlation: 0.4-0.6
- Binary accuracy: 75-85%

**Steering (Target: 1-2 successful examples)**
- At least 1-2 scenarios show clear behavioral change
- Higher alpha values (5.0-10.0) show stronger effects

## Troubleshooting

### Memory Issues

If you get OOM errors on M1 Pro:

```python
# Edit src/probe_extraction.py
# Change batch_size from 1 to 1 (already at minimum)
# Or reduce layers to [16, 20] only
```

### API Rate Limits

If dataset generation hits rate limits:

```python
# Edit src/generate_dataset.py
# Reduce RUNS_PER_MODEL_PER_SCENARIO from 3 to 2
```

### Model Loading Issues

If Gemma-2-9B doesn't load:

```python
# Try smaller model in src/probe_extraction.py:
model_name = "google/gemma-2-2b-it"  # Faster, less capable
```

## Next Steps

1. **Analyze results**: Open `notebooks/01_analyze_results.ipynb`
2. **Modify parameters**: Edit `src/probe_extraction.py` to test different layers
3. **Add scenarios**: Add more EIA scenarios to `data/eia_scenarios/scenarios.json`
4. **Scale up**: Generate 500+ pairs for more robust results
5. **Try other models**: Test on Llama, Phi, etc.

## Questions?

See `README.md` for full documentation or open an issue.
