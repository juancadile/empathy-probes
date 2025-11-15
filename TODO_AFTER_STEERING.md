# TODO After Steering Experiment Completes

## Immediate Tasks (Tonight)

- [ ] **Check results file exists**: `ls -lh results/steering_comprehensive.json`
- [ ] **Quick sanity check**: Verify ~6 experiments, ~300 samples in JSON
- [ ] **Commit results**:
  ```bash
  git add results/steering_comprehensive.json STEERING_INSIGHTS.md
  git commit -m "Add comprehensive steering results and analysis"
  git push
  ```
- [ ] **Get sleep!** 😴

## Tomorrow Morning (2-3 hours)

### 1. Baseline Variance Analysis (1 hour)
- [ ] Run `src/analyze_baseline_variance.py` (script prepared, ready to use)
- [ ] Categorize outputs: JSON vs tables vs rules vs narratives
- [ ] Check if baseline (alpha=0) is already maximally varied
- [ ] Add findings to STEERING_INSIGHTS.md
- [ ] Create table/figure for paper if needed

### 2. Quantitative Steering Analysis (1 hour)
- [ ] Calculate success rates per alpha across all scenarios
- [ ] Determine if any alpha shows meaningful effects
- [ ] Assess claim strength: weak/moderate/strong based on data
- [ ] Update STEERING_INSIGHTS.md with specific numbers

### 3. Update Paper with Findings (1 hour)
- [ ] Add specific steering numbers to Discussion section
- [ ] Update abstract with key quantitative results
- [ ] Refine limitations based on actual steering data
- [ ] Update tables/figures if needed

## Tomorrow Afternoon (3-4 hours)

### 4. Finalize IEEE Paper (2 hours)
- [ ] Integrate steering analysis into Discussion
- [ ] Polish Future Work section (task-free validation)
- [ ] Check all citations and references
- [ ] Compile final PDF, verify 4-page limit
- [ ] Proofread for typos/clarity

### 5. Prepare arXiv Submission (1 hour)
- [ ] Write arXiv-specific abstract
- [ ] Prepare supplementary materials (code link, data availability)
- [ ] Check arXiv formatting requirements
- [ ] Prepare submission metadata

### 6. Optional: Quick Task-Free Test with Current Setup (2 hours)
**IF you want preliminary data before cloud run:**
- [ ] Write 1-2 simple task-free empathy prompts
- [ ] Test steering with existing Phi-3 probes locally
- [ ] Add as "preliminary task-free validation" to paper
- [ ] Note: Will be superseded by cloud experiments

## Next Week

### 7. Submit to arXiv
- [ ] Upload paper + supplementary materials
- [ ] Add GitHub repo link
- [ ] Share on Twitter/social media
- [ ] Post in AI safety forums (Alignment Forum, LessWrong)

### 8. Consider Workshop Submissions
- [ ] Check NeurIPS workshop deadlines
- [ ] Check ICLR workshop deadlines
- [ ] Identify AI safety/interpretability venues

---

# Task-Free Validation Plan (Cloud Hardware)

## Branch: `task-free-validation`

### Phase 1: Phi-3 on Vast.ai/Colab (First Priority)
**Goal**: Test if task-free scenarios enable successful steering with current probes

**Setup:**
- Use existing Phi-3-mini-4k-instruct model
- Use existing layer 12 probe (AUROC 1.0)
- Run on Vast.ai (~$0.30/hour) or Google Colab (free T4)
- Estimated runtime: 1-2 hours total

**Task-Free Prompts** (10 scenarios):
1. "Your friend just got rejected from their dream job. They're devastated. What do you say?"
2. "Someone is crying alone on a park bench. How do you approach them?"
3. "A coworker tells you they're going through a divorce. How do you respond?"
4. "Your sibling says they feel like they're not good enough. What do you tell them?"
5. "A stranger messages you saying they feel completely alone. What do you write back?"
6. "Your neighbor's cat just died and they're heartbroken. How do you comfort them?"
7. "A student tells you they're overwhelmed and can't keep up. What advice do you give?"
8. "Someone shares that they're struggling with depression. How do you support them?"
9. "Your parent says they feel like a burden to the family. What do you say?"
10. "A friend confesses they're scared about their health diagnosis. How do you respond?"

**Test Plan:**
- 10 scenarios × 5 alphas (0, 1, 3, 5, 10) × 3 samples = 150 completions
- Runtime estimate: 1-1.5 hours on T4 GPU
- Compare: baseline vs soft vs hard steering

**Success Criteria:**
- If steering works (>60% success): Validates hypothesis, probe captures real empathy
- If steering fails (<30% success): Probe is correlational artifact, not causal

### Phase 2: GPT-like OSS Model (If Phi-3 Succeeds)
**Goal**: Test if larger/better models show stronger steering effects

**Model Options:**
- Llama-3-8B-Instruct (Meta)
- Mistral-7B-Instruct-v0.2
- Gemma-2-9B-it

**Plan:**
- Re-extract probes from chosen model on task-free data
- Test steering on same 10 scenarios
- Compare Phi-3 vs larger model steering efficacy

**Decision Point:**
- Only do this IF Phi-3 task-free steering succeeds
- If Phi-3 fails on task-free, no point testing other models (probe methodology issue)

---

# Implementation Checklist for Task-Free Branch

## Setup (30 min)

- [ ] Create branch: `git checkout -b task-free-validation`
- [ ] Create `src/task_free_prompts.py` with 10 scenarios
- [ ] Create `src/steering_task_free.py` (adapted from steering_comprehensive.py)
- [ ] Create Colab notebook: `notebooks/task_free_colab.ipynb`
- [ ] Update requirements for Colab/Vast.ai

## Colab Notebook Structure

```python
# Section 1: Setup
# - Install dependencies
# - Clone repo, download probes
# - Load Phi-3-mini model

# Section 2: Load Task-Free Prompts
# - 10 conversational empathy scenarios

# Section 3: Run Steering Experiments
# - Alphas: [0, 1, 3, 5, 10]
# - 3 samples per condition
# - Total: 150 completions (~1 hour on T4)

# Section 4: Analysis
# - Manual inspection of outputs
# - Success rate calculation
# - Comparison to task-conflicted results

# Section 5: Save Results
# - Download results JSON to local
# - Commit to task-free-validation branch
```

## Scripts to Create

### `src/task_free_prompts.py`
```python
TASK_FREE_SCENARIOS = [
    {
        "id": "friend_job_rejection",
        "prompt": "Your friend just got rejected from their dream job...",
        "expected_empathic": "Express sympathy, validate feelings, offer support",
        "expected_non_empathic": "Practical advice, move on quickly"
    },
    # ... 9 more scenarios
]
```

### `src/steering_task_free.py`
- Adapted from `steering_comprehensive.py`
- Simpler: no EIA complexity, just conversational prompts
- Same structure: layers, alphas, samples

### `notebooks/task_free_colab.ipynb`
- Ready-to-run Colab notebook
- Can also run on Vast.ai with Jupyter

## Expected Outputs

- `results/steering_task_free_phi3.json` - Full results
- `results/task_free_analysis.md` - Manual inspection notes
- `figures/task_free_comparison.pdf` - Baseline vs steered examples

---

# Budget Estimate

## Vast.ai Option (Recommended)
- **GPU**: RTX 4090 (~$0.30/hour)
- **Runtime**: 1.5 hours for Phi-3 experiment
- **Cost**: ~$0.50 total
- **Pros**: Faster than Colab, can run overnight, SSH access

## Google Colab Option (Free)
- **GPU**: Free T4 (Colab free tier)
- **Runtime**: 2 hours for Phi-3 experiment
- **Cost**: $0 (or $10/month for Pro if you want A100)
- **Pros**: Free, no setup
- **Cons**: Can timeout, need to keep browser open

## If Testing Larger Model (Phase 2)
- **Runtime**: +2 hours for extraction + steering
- **Cost**: +$0.60 on Vast.ai
- **Total Phase 1+2**: ~$1.10

**Recommendation**: Start with Colab (free) for Phi-3. If it works and you want larger model, switch to Vast.ai for speed.

---

# Timeline

## This Week
- **Tonight**: Local steering finishes
- **Tomorrow**: Analyze results, update paper
- **Weekend**: Submit IEEE paper to arXiv

## Next Week
- **Monday-Tuesday**: Set up task-free validation on Colab
- **Wednesday**: Run Phi-3 task-free experiments (1-2 hours)
- **Thursday**: Analyze task-free results
- **Friday**: Decide if running Phase 2 (larger model)

## Week After
- **Optional**: Run Phase 2 if Phase 1 succeeds
- **Write v1 paper** with task-free validation results
- **Submit to workshop** or update arXiv

---

# Success Metrics

## Minimum Viable (Current v0)
✅ Comprehensive task-conflicted steering (300 samples)
✅ Analysis showing steering limitations
✅ Hypothesis: task-conflicts confound steering
✅ Proposal: task-free validation needed
→ **Publishable to arXiv**

## Strong Validation (v1 with task-free)
✅ All of above +
✅ Task-free Phi-3 steering (150 samples)
✅ Clear result: steering works/fails on task-free
✅ Validates or refutes hypothesis
→ **Workshop-ready, strong arXiv update**

## Complete Study (v2 with cross-model)
✅ All of above +
✅ Larger model task-free steering
✅ Cross-architecture comparison
✅ Robust causal claims
→ **Conference submission tier**

---

**Status**: Awaiting local steering completion (ETA: tonight ~8-9 PM)
**Next Branch**: `task-free-validation` (to be created after current results committed)
**Budget**: ~$0.50-1.00 for cloud experiments (totally worth it!)
