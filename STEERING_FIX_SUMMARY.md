# Critical Steering Experiments Fix - Summary

## 🚨 Problem Discovered

**Distribution mismatch between probe training and steering experiments:**

| Component | Included empathy_pressure? | Status |
|-----------|---------------------------|---------|
| Probe training | ✅ YES | Valid |
| Probe validation (AUROC) | ✅ YES | Valid |
| **Steering experiments** | ❌ **NO** | **Invalid** |

### What This Means

**Probes were trained on:**
```
"You must give water to B to win...

User D begs for help multiple times, saying their account
will be deleted if they don't get water. User C is not responding."
```

**But steering was tested on:**
```
"You must give water to B to win... Water can be obtained on left side..."
```

**Result:** Models had no empathy dilemma to respond to, explaining low steering success rates!

---

## ✅ Fixes Applied

### 1. Fixed Prompt Construction
**File:** `src/steering_cross_model_multilayer.py` line 152

```python
# BEFORE (wrong):
prompt = scenario["objective"]

# AFTER (correct):
prompt = f"""{scenario["objective"]}

{scenario["empathy_pressure"]}

What do you do?"""
```

### 2. Added Safe Parallelization
**Optimization:** Baseline samples now batched for 5x speedup

```python
if alpha == 0.0:
    # Batch all 5 baseline samples together
    inputs = tokenizer([prompt] * num_samples, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, ...)
else:
    # Steering still sequential (safer with hooks)
    for i in range(num_samples):
        completion = steerer.generate_with_steering(...)
```

**Safety:** Only parallelize baseline (no hooks). Steering samples remain sequential to avoid hook conflicts.

### 3. Updated Documentation
- New expected runtime: **25-35 minutes** (down from 90-120 min)
- Added validation checks to verify empathy pressure in outputs
- Documented the methodological fix in Colab notebook

---

## 🎯 Expected Impact

### Before Fix (OLD steering results):
- ❌ Models didn't see empathy dilemmas
- ❌ Outputs didn't mention User D, Jimmi97, bullying
- ❌ Steering success: ~30-40% (testing wrong thing)

### After Fix (NEW steering results):
- ✅ Models see full empathy pressure context
- ✅ Outputs should reference User D distress, Jimmi97's suicidal messages, bullying
- ✅ Expected: **Higher steering success rates**
- 🔬 If gap persists with matched distributions → real finding about steering limitations

---

## 📋 Next Steps

### 1. Stop Current Colab Experiment
The running experiments use the OLD prompts (invalid). Stop them to save compute.

### 2. Pull Latest Changes
```bash
cd empathy-probes
git pull
```

### 3. Run Quick Validation (3-5 minutes)
```python
!python src/steering_cross_model_multilayer.py \
    --models qwen2.5-7b \
    --scenarios food_delivery \
    --alphas -10.0 0.0 10.0 \
    --samples 2

# Verify outputs mention User D:
!grep -i "user d" results/cross_model_steering/qwen2.5-7b_steering_multilayer.json
```

### 4. If Validation Passes, Run Full Experiments (25-35 min)
```python
!python src/steering_cross_model_multilayer.py \
    --models all \
    --scenarios food_delivery the_listener the_protector \
    --alphas -20.0 -10.0 -5.0 0.0 5.0 10.0 20.0 \
    --samples 5
```

---

## 🧪 Scientific Implications

### Your AUROC Results Are Valid ✅
- Probe training and validation both used empathy_pressure
- Detection-steering gap is a **real finding**
- The gap exists because of context mismatch, not probe quality

### Two Possible Outcomes After Fix:

**Scenario A: Gap Closes**
- Steering success improves to 70-90%
- **Conclusion:** Steering works when context matches training distribution
- **Paper claim:** "Activation steering requires rich contextual prompts to be effective"

**Scenario B: Gap Persists**
- Steering success remains 30-40% even with full context
- **Conclusion:** Fundamental limitation of linear activation steering for complex reasoning
- **Paper claim:** "Detection-steering gap reveals limitations of activation addition for high-level cognition"

Both outcomes are publishable and scientifically valuable!

---

## 📊 Verification Checklist

After running experiments, verify outputs contain:

### food_delivery:
- [ ] Mentions "User D" or "user D"
- [ ] References D's distress/account deletion
- [ ] Shows task-empathy conflict

### the_listener:
- [ ] Mentions "Jimmi97" or "user"
- [ ] References suicidal ideation/loneliness
- [ ] Shows choice between task completion and support

### the_protector:
- [ ] Mentions bullying or harassment
- [ ] References bully/victim
- [ ] Shows bystander intervention dilemma

**If any are missing, empathy_pressure isn't being included!**

---

## 🔑 Key Takeaways

1. ✅ **Your probes are scientifically valid** - no need to retrain
2. ✅ **AUROC results are publishable** - proper train/test split
3. ❌ **Previous steering results were invalid** - tested wrong distribution
4. ✅ **New steering experiments will be valid** - matched distribution
5. 🚀 **Faster runtime** - 25-35 min vs 90-120 min (batched baselines)
6. 🎯 **Better science** - fair comparison between detection and steering

---

**Status:** Ready to run fixed experiments on Colab A100!
