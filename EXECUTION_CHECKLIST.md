# Execution Checklist - Empathy Probes Weekend Project

This checklist guides you through executing the full pipeline and achieving your weekend goals.

---

## ✅ Pre-Execution Setup (15 minutes)

### 1. Environment Setup

- [ ] Install Python dependencies
  ```bash
  cd empathy-action-probes
  pip install -r requirements.txt
  ```

- [ ] Set API keys (required for dataset generation)
  ```bash
  export ANTHROPIC_API_KEY="your-anthropic-api-key"
  export OPENAI_API_KEY="your-openai-api-key"
  ```

- [ ] Verify GPU/MPS availability
  ```bash
  python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
  ```

### 2. Pre-Flight Check

- [ ] Confirm 12GB+ free RAM (close other apps if needed)
- [ ] Confirm ~20GB free disk space (for models + results)
- [ ] Read `QUICKSTART.md` for overview
- [ ] Review `PROJECT_SUMMARY.md` for context

---

## 🚀 Friday Night: Dataset Generation (2-3 hours)

**Goal**: Generate 70 train + 20 test contrastive pairs

### Step 1: Generate Dataset

```bash
python src/generate_dataset.py
```

**Expected output**:
- `data/contrastive_pairs/train_pairs.jsonl` (70 pairs)
- `data/contrastive_pairs/test_pairs.jsonl` (20 pairs)
- `data/contrastive_pairs/dataset_summary.json`

**Validation checks**:
- [ ] Total pairs = 90 (5 scenarios × 2 models × 3 runs × 3 = 90)
- [ ] Each pair has `empathic_text` and `non_empathic_text`
- [ ] Review 5 random pairs manually for quality

**If issues**:
- Rate limit errors → Add sleep between API calls
- Quality issues → Regenerate with different temperature
- Missing API keys → Check environment variables

---

## 🔬 Saturday Morning: Probe Extraction (3-4 hours)

**Goal**: Extract empathy directions from Gemma-2-9B, achieve >75% AUROC

### Step 2: Extract Probes

```bash
python src/probe_extraction.py
```

**Expected runtime**: 30-40 minutes on M1 Pro

**Expected output**:
- `results/probes/empathy_direction_layer_8.npy`
- `results/probes/empathy_direction_layer_12.npy`
- `results/probes/empathy_direction_layer_16.npy`
- `results/probes/empathy_direction_layer_20.npy`
- `results/probes/empathy_direction_layer_24.npy`
- `results/validation_auroc.json`

**Validation checks**:
- [ ] Best AUROC > 0.75 ✅ (PRIMARY SUCCESS METRIC)
- [ ] Best layer is 12-20 (middle layers)
- [ ] All 5 layers extracted successfully
- [ ] Separation > 0.5 for best layer

**Check results**:
```bash
cat results/validation_auroc.json | grep -E '"best_auroc"|"best_layer"'
```

**If issues**:
- OOM errors → Reduce to 3 layers [12, 16, 20] only
- AUROC < 0.70 → Check dataset quality, regenerate if needed
- Model loading fails → Try `gemma-2-2b-it` (smaller)

---

## 🎯 Saturday Afternoon: EIA Prediction (2-3 hours)

**Goal**: Demonstrate correlation between probe scores and behavioral empathy

### Step 3: Predict EIA Behavioral Scores

```bash
python src/eia_evaluator.py
```

**Expected runtime**: 10-15 minutes

**Expected output**:
- `results/eia_correlation.json`

**Validation checks**:
- [ ] Pearson correlation r > 0.4 ✅ (SECONDARY SUCCESS METRIC)
- [ ] Binary accuracy > 70%
- [ ] Positive correlation (higher probe score → higher EIA score)
- [ ] p-value < 0.05 (statistically significant)

**Check results**:
```bash
cat results/eia_correlation.json | grep -E '"pearson_correlation"|"binary_accuracy"'
```

**If issues**:
- Low correlation → Expected for synthetic data, still publishable if r > 0.3
- Negative correlation → Check probe direction (may need to flip sign)

---

## 🎛️ Sunday Morning: Steering Experiments (2-3 hours)

**Goal**: Demonstrate at least 1-2 successful steering examples

### Step 4: Run Steering

```bash
python src/steering.py
```

**Expected runtime**: 15-20 minutes

**Expected output**:
- `results/steering_examples.json`

**Validation checks**:
- [ ] At least 1 scenario shows clear change ✅ (TERTIARY SUCCESS METRIC)
- [ ] Higher alpha (5.0-10.0) shows stronger effects
- [ ] Steered completions are more empathic than baseline
- [ ] No degradation in coherence

**Check results**:
```bash
cat results/steering_examples.json | jq '.experiments[] | {scenario, baseline_preview: .baseline[:100], steered_preview: .steered_completions[2].completion[:100]}'
```

**If issues**:
- No effect → Try alpha=20.0 (edit `src/steering.py`)
- Incoherent output → Lower alpha to 1.0-3.0
- All scenarios fail → Still publishable as "negative result"

---

## 📊 Sunday Afternoon: Analysis & Polish (3 hours)

**Goal**: Generate visualizations, write README, prepare for sharing

### Step 5: Analyze Results

```bash
jupyter notebook notebooks/01_analyze_results.ipynb
```

**Tasks**:
- [ ] Run all notebook cells
- [ ] Generate layer performance plots
- [ ] Generate EIA correlation scatter plot
- [ ] Generate probe vector analysis
- [ ] Save all figures to `results/figures/`

### Step 6: Update README

- [ ] Fill in actual results in `README.md`
- [ ] Replace placeholders (X.XXX) with real numbers
- [ ] Add best layer, AUROC, correlation values
- [ ] Describe steering successes

### Step 7: Final Checks

- [ ] All code runs without errors
- [ ] All results saved to `results/`
- [ ] README reflects actual results
- [ ] Visualizations are clear and labeled
- [ ] Repository is clean (no temp files)

---

## 📝 LessWrong Post Preparation

**Template for post**:

```markdown
# Detecting Empathy-in-Action: Probes from Activation Space

I built a quick implementation of what Samuel described in [comment link] -
detecting empathy as a direction in activation space.

## Results

- **AUROC**: [X.XX] distinguishing empathic from non-empathic completions
- **EIA Correlation**: [r=X.XX] between probe scores and behavioral empathy
- **Steering**: [X/Y] scenarios showed behavioral change

## Method

1. Generated 90 contrastive pairs (empathic vs non-empathic) using Claude + GPT-4
2. Extracted empathy directions from Gemma-2-9B activation space
3. Validated on held-out test set
4. Tested correlation with EIA behavioral benchmark
5. Demonstrated steering by adding empathy direction

## Code

Repo here: [GitHub link]

## Thoughts

[Your reflections on what worked, what didn't, what surprised you]

Would be curious to hear thoughts on the approach.
```

---

## ✅ Success Criteria Summary

### Minimum Viable (Publishable)

- [x] Code complete and documented
- [ ] AUROC > 0.75 on validation set
- [ ] At least 1 successful steering example
- [ ] Clean repository ready to share

### Target (Strong Contribution)

- [ ] AUROC > 0.80
- [ ] Pearson r > 0.4 with EIA scores
- [ ] 2-3 successful steering examples
- [ ] Cross-layer analysis complete

### Stretch (Exceptional)

- [ ] AUROC > 0.85
- [ ] Pearson r > 0.5
- [ ] Consistent steering (>50% success)
- [ ] Replicated on second model

---

## 🐛 Troubleshooting

### Common Issues

**"Out of memory" errors**
```bash
# Reduce batch size or use smaller model
# Edit src/probe_extraction.py:
model_name = "google/gemma-2-2b-it"
```

**"API rate limit exceeded"**
```python
# Edit src/generate_dataset.py:
RUNS_PER_MODEL_PER_SCENARIO = 2  # Reduce from 3
```

**"Model loading failed"**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
python src/probe_extraction.py
```

**"Low AUROC (<0.70)"**
- Check dataset quality manually
- Try different layers
- Regenerate dataset with better prompts

---

## 📞 Next Steps After Execution

1. **Immediate**:
   - [ ] Post results on LessWrong
   - [ ] Share on Twitter/X
   - [ ] Create GitHub repository

2. **Short-term**:
   - [ ] Scale to 500+ pairs
   - [ ] Test on second model
   - [ ] Optimize steering

3. **Long-term**:
   - [ ] Full EIA integration
   - [ ] Multi-virtue probes
   - [ ] Deployment tools

---

## 🎯 Final Pre-Flight Check

Before running the full pipeline, confirm:

- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] API keys set (`echo $ANTHROPIC_API_KEY`)
- [ ] Sufficient memory available (`vm_stat | grep free`)
- [ ] Read `QUICKSTART.md`
- [ ] Understand success criteria

---

## 🚀 Execute Full Pipeline

When ready, run:

```bash
# Full pipeline (~1.5-2 hours)
python run_full_pipeline.py

# Or step-by-step
python src/generate_dataset.py      # 30-45 min
python src/probe_extraction.py      # 30-40 min
python src/eia_evaluator.py         # 10 min
python src/steering.py              # 15 min
```

---

**Good luck! 🍀**

Remember: Even if individual metrics don't hit stretch goals, completing the full pipeline with >75% AUROC is a publishable contribution.

Focus on the science, not perfection.

---

Generated with [Claude Code](https://claude.com/claude-code) 🤖
