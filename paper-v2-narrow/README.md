# Narrow V2 Paper

Standalone manuscript package for:

> **Causal Control Under Representational Ambiguity:** Costly-Helping Choices
> in a Language Model

The manuscript does not rely on any earlier paper or version narrative. Its
claim ceiling is defined in `CLAIM_LEDGER.md`.

## Build

```bash
make
```

This regenerates every prototype figure from committed JSON artifacts and then
builds `paper.pdf` with `latexmk`.

## Files

- `paper.tex`: standalone two-column manuscript plus appendix.
- `paper.pdf`: current compiled draft.
- `references.bib`: working bibliography.
- `CLAIM_LEDGER.md`: claim-to-artifact binding and forbidden upgrades.
- `FIGURE_PLAN.md`: main and appendix visual plan.
- `scripts/build_figures.py`: deterministic prototype figure generator.
- `figures/`: generated PDF and PNG figures.

## Current claim ceiling

The paper supports causal control of a matched costly-helping choice assay by a
rank-one edit to a selected mid-layer MLP band in Gemma-2-9B-it. It does not
claim a construct-pure welfare representation, complete circuit, general
capability preservation, enacted-behavior control, or cross-model universality.

