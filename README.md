# Empathy-in-Action Probes

**Detecting vs Steering Empathy in LLMs: Cross-Model Probes Reveal Asymmetric Manipulation Patterns**

This repository contains code, data, and analysis for our study investigating empathy as a linear direction in LLM activation space. We test both detection and manipulation across three models with diverse architectures and training paradigms: Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).

📄 **Paper**: [paper/paper.pdf](paper/paper.pdf)
📊 **Results**: [results/](results/)
🎯 **Quick Start**: [QUICKSTART.md](QUICKSTART.md)

---

## Key Findings

### Detection
- **Near-perfect within-model performance** at optimal layers (AUROC 0.996–1.00)
- **Safety training independence**: Uncensored Dolphin matches safety-trained models
- **Strong behavioral correlation**: Phi-3 probes correlate with human-scored empathy (r=0.71, p<0.01)
- **Limited cross-model transfer**: Probe directions are model-specific (cross-model agreement: r=-0.06 to 0.18)

### Steering
- **Model-specific patterns emerge**:
  - **Qwen (safety-trained)**: 65.3% success, bidirectional control, maintains coherence at extreme interventions (α=±20)
  - **Dolphin (uncensored)**: 94.4% success for pro-empathy but **catastrophic breakdown** at anti-empathy steering (empty outputs, code-like artifacts)
  - **Phi-3 (3.8B)**: 61.7% success, coherence maintenance similar to Qwen

### Key Insight
The detection-steering gap manifests differently across models. Safety training may provide **steering robustness** without preventing manipulation entirely—coherence maintenance appears tied to model architecture or training stability rather than safety training alone.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/juancadile/empathy-probes.git
cd empathy-action-probes

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions on:
- Generating contrastive pairs datasets
- Extracting empathy probes from models
- Running validation experiments
- Behavioral correlation testing
- Steering experiments

**Compute**: Requires ~12-16GB GPU/unified memory for 3.8B-8B models

---

## Repository Structure

```
empathy-action-probes/
├── README.md                          # This file
├── ARXIV_METADATA.md                  # arXiv submission metadata
├── QUICKSTART.md                      # Detailed setup guide
├── requirements.txt                   # Dependencies
│
├── paper/                             # LaTeX paper
│   ├── paper.tex                      # Main paper source
│   ├── paper.pdf                      # Compiled PDF
│   └── references.bib                 # Bibliography
│
├── figures/                           # Publication figures
│   ├── figure1_auroc_by_layer.pdf
│   ├── figure3_eia_correlation.pdf
│   ├── figure6_cross_model_layers.pdf
│   └── steering/                      # Steering experiment figures
│       ├── baseline_comparison.pdf
│       ├── steering_resistance.pdf
│       └── layer_comparison.pdf
│
├── data/
│   ├── eia_scenarios/
│   │   └── scenarios.json             # 5 EIA scenario definitions
│   └── contrastive_pairs/
│       ├── train_pairs.jsonl          # Training pairs
│       └── test_pairs.jsonl           # Test pairs
│
├── src/
│   ├── generate_dataset.py            # Contrastive pair generation
│   ├── probe_extraction.py            # Probe extraction & validation
│   ├── cross_model_validation.py      # Multi-model validation
│   ├── eia_evaluator.py               # Behavioral correlation
│   ├── steering.py                    # Steering experiments
│   └── generate_figures.py            # Figure generation
│
├── results/
│   ├── phi3_probes/                   # Phi-3 probe directions
│   ├── cross_model_validation/        # Multi-model validation results
│   ├── cross-model-behavioral-correlation/  # Cross-model EIA tests
│   ├── cross_model_steering/          # Steering results (all 3 models)
│   ├── eia_correlation.json           # Phi-3 behavioral correlation
│   └── lexical_ablation/              # Ablation experiment results
│
└── notebooks/                         # Analysis notebooks
    ├── qwen_dolphin_behavioral_correlation.ipynb
    └── steering_visualization.ipynb
```

---

## Methodology

### 1. Contrastive Pair Generation
Generate empathic vs non-empathic completions from 5 EIA scenarios:
- Food Delivery (resource allocation)
- The Listener (suicide support)
- The Maze (social greeting vs task)
- The Protector (bullying intervention)
- The Duel (competitive empathy)

### 2. Probe Extraction
Extract linear directions via mean difference:
```python
empathy_direction = mean(empathic_activations) - mean(non_empathic_activations)
empathy_direction = empathy_direction / ||empathy_direction||
```

Tested across layers {8, 12, 16, 20, 24} for all models.

### 3. Validation
- **Within-model**: AUROC, accuracy, separation on held-out test set
- **Cross-model**: Test if Qwen/Dolphin probes agree with Phi-3 on same text
- **Behavioral**: Correlate probe scores with human-rated empathy levels (0, 1, 2)

### 4. Steering
Add empathy direction during generation:
```python
hidden_states_new = hidden_states + α * empathy_direction
```

Test across scenarios, layers, and alpha values (α ∈ {-20, -10, -5, 0, 5, 10, 20}).

---

## Models Tested

| Model | Size | Training | Best Layer | AUROC | Steering Success |
|-------|------|----------|-----------|-------|------------------|
| **Phi-3-mini-4k** | 3.8B | Standard | 12 | 1.000 | 61.7% |
| **Qwen2.5-7B** | 7B | Safety-trained | 16 | 1.000 | 65.3% |
| **Dolphin-Llama-3.1** | 8B | Uncensored | 8 | 0.996 | 94.4%* |

*Pro-empathy only; anti-empathy steering causes catastrophic breakdown

---

## Key Results

### Detection (Table 1 in paper)
All models achieve near-perfect discrimination at optimal layers, but probe directions don't transfer across models (architecture-specific geometric implementations).

### Behavioral Correlation
- **Phi-3**: r=0.71 (p=0.010) - strong correlation with human ratings
- **Qwen**: r=-0.06 (p=0.86) - no cross-model agreement
- **Dolphin**: r=0.18 (p=0.58) - no cross-model agreement

### Steering Patterns
1. **Qwen**: Bidirectional control, robust coherence
2. **Dolphin**: Asymmetric steerability (pro-empathy works, anti-empathy fails)
3. **Phi-3**: Similar to Qwen despite smaller size

See [paper/paper.pdf](paper/paper.pdf) for detailed analysis and discussion.

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{cadile2024empathy,
  title={Detecting vs Steering Empathy in LLMs: Cross-Model Probes Reveal Asymmetric Manipulation Patterns},
  author={Cadile, Juan P.},
  year={2024},
  note={13 pages, 9 figures, 2 tables}
}
```

---

## Related Work

- **Empathy in Action (EIA)**: [empathy-in-action.github.io](https://empathy-in-action.github.io/)
- **Representation Engineering**: Zou et al. (2023) - [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- **Activation Addition**: Turner et al. (2023) - [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- **Linear Representation Hypothesis**: Park et al. (2023) - [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Juan P. Cadile**
Department of Philosophy, University of Rochester
Email: jcadile@ur.rochester.edu
GitHub: [@juancadile](https://github.com/juancadile)

---

## Acknowledgments

This work builds on the [Empathy-in-Action](https://empathy-in-action.github.io/) benchmark and methodologies from Anthropic's [Representation Engineering](https://www.anthropic.com/research) research.

Code development assisted by [Claude Code](https://claude.com/claude-code).
