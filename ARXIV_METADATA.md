# arXiv Submission Metadata

## Title
Detecting vs Steering Empathy in LLMs: Cross-Model Probes Reveal Asymmetric Manipulation Patterns

## Authors
Juan P. Cadile
- Department of Philosophy, University of Rochester
- Email: jcadile@ur.rochester.edu

## Abstract

We investigate empathy as a linear direction in LLM activation space, testing both detection and manipulation across three models: Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).

**Detection:** Near-perfect within-model performance at optimal layers (AUROC 0.996–1.00). Critically, uncensored Dolphin matches safety-trained models, demonstrating that empathy encoding emerges independent of safety training. Phi-3 probes correlate strongly with behavioral scores (r=0.71, p<0.01). However, cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific geometric implementations despite convergent detection.

**Steering reveals model-specific patterns:** Safety-trained Qwen achieves 65.3% steering success with bidirectional control and complete coherence at extreme interventions (α=±20). Uncensored Dolphin shows 94.4% success for pro-empathy steering (positive alphas only) but catastrophically fails at anti-empathy steering: outputs degenerate into empty strings and code-like artifacts. Phi-3 (3.8B, smallest model) achieves 61.7% success with coherence maintenance similar to Qwen, requiring extreme alphas (α=20) for consistent steering.

**Key insight:** The detection-steering gap manifests differently across models. Qwen (7B, safety-trained) and Phi-3 (3.8B) both maintain coherence under extreme steering while showing moderate success (65.3% and 61.7% respectively). Dolphin (8B, uncensored) shows higher steerability (94.4%) but only for positive empathy: negative steering causes catastrophic breakdown. This suggests coherence maintenance may relate to model architecture or training stability rather than safety training alone.

Our results suggest that safety training may affect the quality of steerability rather than preventing it entirely, though this finding requires validation across more model pairs to challenge assumptions about value lock-in through RLHF.

## Categories
- Primary: cs.CL (Computation and Language)
- Secondary: cs.AI (Artificial Intelligence)
- Secondary: cs.LG (Machine Learning)

## Keywords
- Large Language Models
- Activation Steering
- Representation Engineering
- Empathy Detection
- AI Safety
- Interpretability
- Linear Representation Hypothesis
- RLHF
- Safety Training

## Comments
14 pages, 9 figures, 2 tables. Code and data available at https://github.com/juancadile/empathy-action-probes

## Submission Checklist
- [ ] Paper PDF (paper.pdf)
- [ ] Source files (.tex, .bib, figures/)
- [ ] README with code/data availability
- [ ] Figures in publication-ready format
- [ ] References properly formatted
- [ ] Author information complete
- [ ] Abstract under 1920 characters
