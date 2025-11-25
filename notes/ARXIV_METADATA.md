# arXiv Submission Metadata

## Title
Detecting and Steering LLMs' Empathy in Action

## Authors
Juan P. Cadile
- Department of Philosophy, University of Rochester
- Email: jcadile@ur.rochester.edu

## Abstract (for arXiv submission - plain text, no LaTeX)

We investigate empathy-in-action---the willingness to sacrifice task efficiency to address human needs---as a linear direction in LLM activation space. Using contrastive prompts grounded in the Empathy-in-Action (EIA) benchmark, we test detection and steering across Phi-3-mini-4k (3.8B), Qwen2.5-7B (safety-trained), and Dolphin-Llama-3.1-8B (uncensored).

Detection: All models show AUROC 0.996-1.00 at optimal layers. Uncensored Dolphin matches safety-trained models, demonstrating empathy encoding emerges independent of safety training. Phi-3 probes correlate strongly with EIA behavioral scores (r=0.71, p<0.01). Cross-model probe agreement is limited (Qwen: r=-0.06, Dolphin: r=0.18), revealing architecture-specific implementations despite convergent detection.

Steering: Qwen achieves 65.3% success with bidirectional control and coherence at extreme interventions. Phi-3 shows 61.7% success with similar coherence. Dolphin exhibits asymmetric steerability: 94.4% success for pro-empathy steering but catastrophic breakdown for anti-empathy (empty outputs, code artifacts).

Implications: The detection-steering gap varies by model. Qwen and Phi-3 maintain bidirectional coherence; Dolphin shows robustness only for empathy enhancement. Safety training may affect steering robustness rather than preventing manipulation, though validation across more models is needed.

**Character count: 1292 / 1920 max**

## Categories
- Primary: cs.CL (Computation and Language)
- Secondary: cs.AI (Artificial Intelligence)
- Secondary: cs.LG (Machine Learning)

## ACM Classification
I.2.7 (Natural Language Processing)

## Comments Field
14 pages, 9 figures

## Report Number
[Leave blank]

## Journal Reference
[Leave blank]

## External DOI
[Leave blank]

## MSC Class
[Leave blank]

## Are you an author?
Yes

## Keywords
- Large Language Models
- Activation Steering
- Representation Engineering
- Empathy Detection
- AI Safety
- Interpretability
- Linear Representation Hypothesis
- Behavioral Alignment
- Cross-Model Probes

## Submission Files
Final tarball: `arxiv_submission_clean.tar.gz` (153KB)

Contents:
- paper.tex (with updated title)
- paper.bbl (compiled bibliography)
- references.bib
- neurips_2024.sty
- 9 figure PDFs (flat structure, no subdirectories)

## Submission Checklist
- [x] Paper source (.tex, .bib, .sty)
- [x] All figures in publication-ready format (PDFs)
- [x] Clean tarball without macOS metadata (COPYFILE_DISABLE=1)
- [x] Flat file structure (no subdirectories)
- [x] No compiled PDF (arXiv generates it)
- [x] Abstract under 1920 characters (1292 chars)
- [x] Title updated to reflect actual findings
- [x] Author information complete
- [x] References properly formatted

## Notes
- Title changed from "Cross-Model Probes for Behavioral Alignment" to "Detecting and Steering LLMs' Empathy in Action" to accurately reflect findings that probes DON'T transfer across models
- Used COPYFILE_DISABLE=1 to prevent macOS resource fork files (._*)
- ArXiv automatically removes compiled PDF and generates fresh one
- All LaTeX formatting removed from abstract (no $, \textbf{}, etc.)
