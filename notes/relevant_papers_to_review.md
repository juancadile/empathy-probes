# Relevant Papers to Review

## Papers for Literature Review

1. **Zhi et al. (2024)** - CSAI Conference
   https://dblp1.uni-trier.de/rec/conf/csai/Zhi24.html

2. **ACL 2025 Findings Paper**
   https://aclanthology.org/2025.findings-acl.1188.pdf

3. **arXiv Paper (May 2024)**
   https://arxiv.org/html/2405.17345v1

4. **arXiv Paper (October 2024)**
   https://arxiv.org/abs/2510.04484

## Critical for V2 Interpretation (added 2026-07-10)

5. **Persona Vectors: Monitoring and Controlling Character Traits in Language Models** (Chen et al., Anthropic, 2025)
   https://arxiv.org/abs/2507.21509
   - Extracts trait directions from natural-language trait descriptions via contrastive prompts — **methodologically almost identical to our extraction pipeline**
   - Raises the central question: is our direction "empathy" or "the model playing an empathetic character"?
   - Their automated pipeline lets us extract an explicit "empathetic assistant" persona vector to compare against ours (cosine sim, cross-steering)

6. **Emotion Concepts and their Function in a Large Language Model** (Sofroniew, Kauvar, Saunders et al., 2026)
   https://arxiv.org/abs/2604.07729
   - Claude Sonnet 4.5 has emotion-concept representations that generalize across contexts and causally influence behavior (incl. misalignment rates) — "functional emotions" without implied subjective experience
   - Gives us the discrimination criteria: concept representations should generalize across contexts and track *content*; persona directions should track *who the model is being*

## Already Reviewed

- **Scoring Empathy with Large Language Models** (December 2024)
  https://arxiv.org/html/2412.20264v1
  - Focus: Interpretable empathy scoring via explicit subfactors
  - Complementary to our mechanistic approach (output measurement vs internal causality)
