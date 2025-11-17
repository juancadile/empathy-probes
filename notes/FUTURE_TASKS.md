# Future Tasks for Empathy-Action Probes Project

Based on reviewer feedback, organized by priority.

---

## High Priority (Must Do Before Workshop Submission)

### 1. Conceptual Framing for Empathy Operationalization
- [ ] Add short section motivating the choice of "willingness to sacrifice task efficiency to help"
- [ ] Differentiate between affective, cognitive, and behavioral empathy
- [ ] Discuss whether EIA captures "moral reasoning" vs "task prioritization"
- [ ] Address finding that "empathy direction" may measure "lack of task focus" (all-negative projections)
- [ ] Clarify whether probes capture virtue vs task conflict framing

**Location**: Introduction or new "Conceptual Framework" section

### 2. Add Alignment Theory Significance Section
Create new subsection: **"Implications for Alignment & Safety Training"**

Explicitly connect findings to:
- [ ] Value lock-in claims about RLHF
- [ ] Whether safety tuning resists "removal of human values"
- [ ] Representation pathways for moral concepts
- [ ] Debate over whether empathy is a "superposition" feature

**Location**: Discussion section

### 3. Causal Limitations Acknowledgment
- [ ] Add 1 paragraph explicitly stating causal limitations of steering experiments
- [ ] Discuss: Does adding the direction *cause* empathy or just perturb generation?
- [ ] Address potential confounding with other features in residual stream injection
- [ ] Speculative discussion of causal mediation or pathways (if feasible)

**Location**: Limitations or Discussion section

### 4. Paper Length and Structure Cleanup
- [ ] Remove repetitive restatements of AUROC results (currently 3-4 times)
- [ ] Consolidate steering examples into one comprehensive table
- [ ] Move minor plots to appendix
- [ ] Tighten prose throughout
- [ ] Ensure consistent terminology ("non-empathic" vs "task-focused" vs "anti-empathy")

**Target**: Reduce length by 15-20% while maintaining substance

### 5. Discussion Section Overhaul
- [ ] Add tighter Discussion that explains *why these results matter*
- [ ] Add paragraph on theoretical meaning of asymmetric steerability
- [ ] Strengthen "Convergent Concepts, Divergent Geometry" connection to implications

---

## Medium Priority (Would Strengthen Significantly)

### 6. Cross-Model Alignment Experiments
- [ ] Try CCA (Canonical Correlation Analysis) to align representation spaces
- [ ] Try Orthogonal Procrustes alignment
- [ ] Test whether aligned probes transfer better
- [ ] Check if probe direction is stable within same model across different seeds

**Expected outcome**: Better theoretical explanation for cross-model probe failure

### 7. Expand Statistical Rigor
- [ ] Add variance for random vector baseline for Qwen and Dolphin (currently only Phi-3)
- [ ] Include confidence intervals on all key metrics
- [ ] Report statistical significance tests where appropriate

### 8. Human Examples from EIA
- [ ] Add representative EIA scenario examples to appendix for clarity
- [ ] Show what empathic vs non-empathic responses look like
- [ ] Help readers understand the operationalization concretely

---

## Lower Priority (Nice to Have)

### 9. Additional Uncensored Model Testing
Test whether Dolphin's asymmetry generalizes to other uncensored models:
- [ ] Mistral-uncensored
- [ ] Hermes variants
- [ ] OpenOrca-uncensored
- [ ] WizardLM-uncensored

**Goal**: Determine if asymmetric steerability is Dolphin-specific or general property of uncensored models

### 10. Figure Quality Improvements
- [ ] Simplify visually busy figures
- [ ] Ensure main figures (dose-response, AUROC by layer, cross-model) are publication-ready
- [ ] Consider combining related subfigures
- [ ] Add clearer captions with take-home messages

---

## Publication Strategy

### Immediate Actions
- [x] Post current version to arXiv as v0
- [ ] Implement high-priority revisions
- [ ] Prepare workshop-length version (4-6 pages + appendix)

### Target Venues (in order of fit)

**Most Likely to Accept:**
1. NeurIPS RepEng Workshop (Representation Engineering)
2. ICLR Mechanistic Interpretability Workshop
3. FAccT/AIES workshops on value alignment or moral reasoning in LLMs

**Possible:**
- COLM interpretability tracks
- ML Safety Workshop
- Anthropic/DeepMind internal reading groups

**Future (after strengthening):**
- Main-track NeurIPS/ICLR
- Main-track FAccT or AIES

---

## Theoretical Questions to Address

### Cross-Model Probe Failure
- [ ] Can CCA/Procrustes actually align the spaces?
- [ ] Have we tested aligning activation subspaces?
- [ ] Is probe direction stable within model across seeds?
- [ ] What does this say about representation learning more broadly?

### Asymmetric Steerability
- [ ] Why does Dolphin catastrophically fail on anti-empathy steering?
- [ ] What does this reveal about safety tuning mechanisms?
- [ ] Connection to "unlearning" and value removal literature?
- [ ] Implications for robustness of alignment interventions?

### Linear Separability
- [ ] Why is empathy linearly separable at mid-layers specifically?
- [ ] What computational function do mid-layers serve for this concept?
- [ ] How does this relate to other moral concepts (honesty, harm, fairness)?

---

## Long-Term Research Directions

### Extensions
1. **Multi-concept probes**: Can we detect bundles of moral concepts?
2. **Causal interventions**: Use activation patching to test causal claims
3. **Scaling laws**: How does probe quality change with model size?
4. **Training dynamics**: When does linear separability emerge during training?
5. **Human-AI alignment metrics**: Use EIA as testbed for alignment evaluation

### Applications
1. **Alignment auditing**: Use probes to detect value changes post-training
2. **Steerable AI assistants**: Apply to real-world empathy modulation
3. **Safety benchmarking**: Compare safety-trained vs base models systematically

---

## Notes from Review

**Key Strengths to Preserve:**
- Multi-model comparison with ablations
- Clear empirical results (AUROC ~1.0, r=.71)
- "Convergent semantics, divergent geometry" framing
- Honest acknowledgment of limitations

**Reviewer's Core Assessment:**
> "Empathy is linearly separable in all three models, but the geometry is model-specific, and steerability reveals asymmetric failure modes."

This is the contribution. Keep it front and center.

**Main Weaknesses to Address:**
1. Underdeveloped philosophical motivation
2. Unclear alignment/safety significance
3. Paper too long for contribution size
4. Need clearer causal story

---

## Timeline Suggestion

**Week 1-2**: High-priority revisions (conceptual framing, discussion, length)
**Week 3**: Medium-priority experiments (CCA, stats, examples)
**Week 4**: Workshop version preparation + submission
**Ongoing**: arXiv v1 with updates

---

## Questions for Juan/Samuel

1. Which venue should we target first?
2. Priority: faster workshop submission vs more experiments?
3. Should we focus on empirical depth or conceptual sharpening?
4. Interest in pursuing additional uncensored models?
5. Timeline constraints for submission?

---

*Last updated: 2025-11-17*
*Based on: Reviewer feedback document*
