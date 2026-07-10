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

7. **Activation Oracles: Generalist LatentQA models** (Karvonen, Chua, Dumas, ..., Evans, Marks, 2025)
   https://arxiv.org/abs/2512.15674 · code+weights: https://github.com/adamkarvonen/activation_oracles
   - LLMs trained to accept activations as input and answer arbitrary natural-language questions about them; generalize far OOD, recover fine-tuned-in propensities never seen in training
   - **Pretrained oracles available for Gemma-2 and Llama-3 families — our exact Tier 1 models**; inference runs on a T4 (→ our A4000)
   - Use as a third discrimination instrument (Stage B): ask the oracle directly whether activations reflect wellbeing-prioritization vs task-focus vs a caring persona; interrogate steered activations ("what changed?"); verify weight-edited models at the representation level (Stage C)
   - Caveat: still activation-level and correlational — a richer *readout*, not causal evidence; can confabulate, so validate on cells with known labels first

8. **Causal Abstractions of Neural Networks** (Geiger, Lu, Icard, Potts, NeurIPS 2021)
   https://proceedings.neurips.cc/paper_files/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf
   - THE theoretical foundation for our evidence ladder: align neural representations with variables of a high-level causal model, verify via **interchange interventions**; a network "realizes" the causal model if intervened network and intervened model agree counterfactually
   - §2 constructs a network where probes read out information perfectly yet the representation is **causally inert** — the formal version of the V1 critique; cite in the paper's intro
   - Concrete upgrade for Stage B: define a high-level causal model of empathy-in-action (PerceiveNeed → WeighCost → Decide) and test alignment with interchange intervention accuracy (IIA); successors DAS / Boundless DAS (ran on Alpaca-7B → feasible at our Tier 1 scale) + `pyvene` library
   - Turns "layer 12 is causal" into "the model realizes a causal abstraction in which a wellbeing variable mediates action selection" — a much better-defined claim

## Already Reviewed

- **Scoring Empathy with Large Language Models** (December 2024)
  https://arxiv.org/html/2412.20264v1
  - Focus: Interpretable empathy scoring via explicit subfactors
  - Complementary to our mechanistic approach (output measurement vs internal causality)
