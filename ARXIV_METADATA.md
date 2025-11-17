# arXiv Submission Metadata

## Title
Detecting vs Steering Empathy in LLMs: Cross-Model Probes Reveal Asymmetric Manipulation Patterns

## Authors
Juan P. Cadile
- Department of Philosophy, University of Rochester
- Email: jcadile@ur.rochester.edu

## Abstract (for arXiv submission - plain text, no markdown)

We investigate empathy as a linear direction in LLM activation space, testing detection and manipulation across Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).

Detection: Near-perfect within-model performance at optimal layers (AUROC 0.996-1.00). Uncensored Dolphin matches safety-trained models, demonstrating empathy encoding emerges independent of safety training. Phi-3 probes correlate with behavioral scores (r=0.71, p<0.01). However, cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific implementations.

Steering reveals model-specific patterns: Qwen achieves 65.3% success with bidirectional control and coherence at extreme interventions (alpha=+/-20). Dolphin shows 94.4% success for pro-empathy steering but catastrophically fails at anti-empathy: outputs degenerate into empty strings and code artifacts. Phi-3 achieves 61.7% success with coherence maintenance similar to Qwen.

The detection-steering gap manifests differently across models. Qwen and Phi-3 maintain coherence under extreme steering while showing moderate success. Dolphin shows higher steerability but only for positive empathy: negative steering causes catastrophic breakdown. This suggests coherence maintenance may relate to model architecture or training stability rather than safety training alone, though this requires validation across more model pairs.

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
