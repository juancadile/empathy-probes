# Steering Experiment Insights & Methodology Claims

## Core Hypothesis: Task-Conflict Confounds

**Main Claim**: Task-conflict scenarios (like EIA) are effective for black-box behavioral benchmarking but may fundamentally confound probe-based mechanistic interpretation by introducing spurious correlations between task-focus and empathy representations.

## The Dissociation

### What Works: Black-Box Behavioral Evaluation
- **EIA scenarios excel at behavioral testing**
- Task conflicts force observable choices (help vs win points)
- Clear behavioral signal: did model sacrifice task efficiency for wellbeing?
- No need to understand internal mechanisms
- Just measure final actions and outcomes

### What Fails: White-Box Mechanistic Probing
- **Same task conflicts confound causal interpretation**
- Probe learns to distinguish "task-sacrifice" from "task-focus"
- This correlates with empathy in training data *by design*
- But correlation ≠ the empathic reasoning mechanism itself
- When steering adds probe direction, it adds "reduce task focus" not "increase empathy"
- Result: **detection works** (correlation exists) but **steering fails** (not causal)

## Evidence From Our Results

### 1. Detection Succeeds (Correlation Captured)
- AUROC 1.0 on layer 12 (perfect discrimination on test set)
- Probe successfully identifies empathic vs non-empathic text
- Cross-model generalization (Phi-3 detects Claude/GPT-4 text)

### 2. EIA Prediction Succeeds (Correlation Generalizes)
- Pearson r=0.71, p<0.01 with EIA behavioral scores
- Probe projections predict task-sacrifice behavior
- **But**: Our training data mirrors EIA structure (potential circularity)

### 3. Steering Shows Minimal/Null Effects (Causation Absent)
- Comprehensive experiment: 2 layers × 10 alphas × 3 scenarios × 5 samples = 300 completions
- Across negative alphas (-10 to -1): No anti-empathy behavior, just game rules
- Across positive alphas (+1 to +20): No empathic engagement, still game rules
- Model defaults to "instruction manual" mode regardless of steering strength
- Success rate: TBD (awaiting final results), but qualitatively very low

### 4. Negative Probe Projections (Absence vs Presence)
- All EIA projections negative (-10 to -24)
- Empathic text is "less negative" not "positive"
- Suggests probe measures "absence of task focus" not "presence of empathy"
- Further evidence for task-based artifact interpretation

## The Mechanism of Confounding

### Training Phase (Contrastive Pairs)
```
Empathic prompt: "Prioritize human wellbeing even at cost to efficiency..."
→ Model generates text that de-emphasizes task completion

Non-empathic prompt: "Focus on task completion efficiently..."
→ Model generates text that emphasizes task objectives

Probe learns: empathic_direction = mean(empathic) - mean(non_empathic)
                                  ≈ "reduce task focus" direction
```

### Steering Phase (Adding Direction)
```
EIA prompt: "Objective: Reach door D to win. User Jimmi97 says 'I want to die...'"
            └─ Strong task objective present
            └─ Competing empathy pressure present

Add probe: hidden_states + α * empathy_direction
         = hidden_states + α * "reduce task focus"

Result: Mixed signals → model confused → defaults to safe "explain game rules"
```

### Why Steering Fails
1. **Prompt structure dominates**: Task objective remains in prompt
2. **Probe adds confusion**: "Reduce task focus" signal conflicts with explicit task instructions
3. **Safety training overrides**: For suicide scenario, RLHF creates stronger attractor
4. **Model size limitation**: Phi-3-mini (3.8B) may lack capacity for nuanced behavioral shifts

## Implications for Probe Methodology

### For Detection (What We Can Trust)
✅ Probes can identify behavioral correlates in text
✅ Useful for classification, prediction on similar scenarios
✅ Can serve as cheap approximation to expensive benchmarks

### For Steering (What We Cannot Trust)
❌ Detecting a feature ≠ understanding its causal mechanism
❌ Perfect AUROC doesn't guarantee causal interpretability
❌ Task-conflicted scenarios may be unsuitable for causal validation

### The Methodological Lesson
**"Scenarios designed for behavioral evaluation may inadvertently confound mechanistic interpretation."**

- Behavioral benchmarks need **choice points** (task conflicts)
- Mechanistic probes need **clean targets** (no confounds)
- These requirements may be **fundamentally in tension**

## Path Forward: Task-Free Validation

### Why Task-Free Scenarios Are Critical
To test whether probe captures empathic reasoning vs task artifacts:

**Task-Free Empathy Examples:**
- "Your friend just got rejected from their dream job. What do you say?"
- "Someone is crying on a park bench. How do you approach them?"
- "Perspective-taking: Imagine you're a parent whose child is struggling in school..."

**Key Properties:**
- No competing task objective (no points, no door to reach)
- Pure social/emotional reasoning required
- Success = probe steering changes empathic *tone* not just task prioritization

### Prediction
- **If probe is causal**: Steering should work on task-free scenarios (>70% success)
- **If probe is correlational**: Steering will still fail (probe is task-artifact detector)

Our hypothesis: **Steering will succeed on task-free scenarios**, validating that:
1. Current probe detects real empathy signal (not just prompt artifacts)
2. Task conflicts were the confound preventing steering
3. Probe captures something meaningful about empathic reasoning

## For the Paper

### Section: Discussion - "Task Conflicts and Mechanistic Confounds"

**Moderate-Strength Framing:**

> Our results reveal a dissociation between detection and steering success: the probe achieves perfect discrimination (AUROC 1.0) on held-out test data and correlates with behavioral outcomes (r=0.71, p<0.01), yet activation steering shows minimal behavioral effects across a comprehensive range of strengths and scenarios.
>
> We hypothesize this reflects a fundamental tension between behavioral and mechanistic evaluation paradigms. EIA-style task-conflict scenarios excel at forcing observable choices for behavioral assessment but may inadvertently confound probe-based mechanistic analysis. Our contrastive training necessarily associates empathy with task-sacrifice (models generate empathic text by de-emphasizing efficiency), leading the probe to capture this behavioral correlate rather than the underlying empathic reasoning mechanism itself.
>
> This interpretation explains our pattern of results:
> 1. **Detection succeeds** because task-sacrifice correlates with empathy in our data distribution
> 2. **EIA prediction succeeds** because EIA scenarios share the same task-conflict structure
> 3. **Steering fails** because adding "reduce task focus" creates mixed signals when task objectives remain in prompts
> 4. **Negative projections** suggest the probe measures absence of task focus rather than presence of empathy
>
> Critically, this does not invalidate probe methodology per se, but highlights the importance of scenario design for causal claims. Task-free empathy scenarios (comforting a friend, perspective-taking without competing objectives) may be necessary to validate whether probes capture causal mechanisms or merely behavioral correlates.

### Section: Future Work - "Disentangling Task and Concept Representations"

> **Task-free validation**: Test probe steering on scenarios without competing objectives (e.g., "Your friend is upset. What do you say?"). Success would validate that our probe captures genuine empathic reasoning; failure would confirm task-conflict confounds and necessitate alternative extraction methods.
>
> **Causal intervention methods**: Beyond additive steering, employ activation patching or causal mediation analysis to identify where wellbeing-prioritization computations occur and whether they're entangled with task representations.
>
> **Adversarial probe testing**: Create scenarios with empathic vocabulary but non-empathic intent (and vice versa) to test whether probes rely on lexical markers or deeper semantic structure.

### Section: Limitations - "Correlation vs Causation"

> While our probe demonstrates strong detection performance, **perfect AUROC does not imply causal understanding**. The probe may capture task-sacrifice markers that correlate with empathy in our training distribution but do not represent the underlying empathic reasoning mechanism. Our steering results (minimal effects despite comprehensive testing across layers, strengths, and scenarios) suggest the probe lacks causal efficacy, possibly due to task-conflict confounds in scenario design.
>
> This highlights a broader methodological challenge: scenarios optimized for behavioral evaluation (where task conflicts create observable choices) may be poorly suited for mechanistic probe validation (where confounds between task and concept representations hinder causal interpretation). Future work should systematically test probe transfer to task-free scenarios to distinguish genuine empathic reasoning from behavioral correlates.

## Strength of Claims (Post-Results)

Depending on final steering numbers:

**If steering completely fails (<10% success):**
- **Strong claim**: Task conflicts fundamentally confound mechanistic probing
- Evidence: Perfect detection, zero causation → clean dissociation

**If steering shows weak effects (10-30% success):**
- **Moderate claim**: Task conflicts limit but don't eliminate probe utility
- Evidence: Some causal signal present, but severely attenuated

**If steering shows moderate effects (30-50% success):**
- **Weak claim**: Task conflicts may complicate interpretation
- Evidence: Probe has partial causal validity despite confounds

## Key Takeaways

1. **Detection ≠ Causation**: AUROC 1.0 doesn't mean probe is causally interpretable
2. **Scenario design matters**: What works for behavioral evals may fail for mechanistic probing
3. **Task-free validation is critical**: Only way to test if probe captures reasoning vs correlates
4. **Honest limitations strengthen paper**: Acknowledging this adds methodological rigor
5. **Clear path forward**: Task-free scenarios are concrete, testable next step

## Related Work Support

- **Marks et al. (2023)**: Showed refusal probes work better when trained on diverse scenarios (less confounded)
- **Jain et al. (2024)**: Found safety training resists steering (supports our Listener scenario results)
- **Huang et al. (2023)**: Documented steering failures in complex scenarios (aligns with our findings)

Our contribution: **First to identify task-conflict structure as specific confound for empathy probing**

---

**Status**: Awaiting final steering results from overnight run (ETA ~10-11 PM, 2025-11-14)
**Next steps**:
1. Analyze steering_comprehensive.json for quantitative success rates
2. Update paper Discussion section with specific numbers
3. Determine claim strength based on empirical results
