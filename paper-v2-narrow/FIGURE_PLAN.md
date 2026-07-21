# Figure Plan

## Main paper

1. **Experimental logic and evidence ceiling.** A compact pipeline from
   contrastive text through lexical stress tests, matched-lexicon controls,
   component attribution, parameter intervention, and nested representation
   search. The final box explicitly separates the supported behavioral claim
   from unsupported construct/circuit claims.
2. **Decodability under lexical controls.** Original versus shuffled-token
   AUROC at blocks 8 and 20, alongside matched-lexicon full-text and
   decision-token AUROC. This motivates the controlled assay without relying
   on historical context.
3. **Causal writer-band effect and task control.** Family-clustered effect
   estimates for raw A/B, chat A/B, paraphrases, and continuation likelihood.
   A separate continuation panel avoids comparing incompatible numerical
   scales. The repaired task control is plotted beside the costly-helping cell;
   the caption explicitly states that paraphrase controls were not measured.
4. **Localization within the tested MLP band.** Rank plot of all 28 two-MLP
   sets from layers 16--23. L19/L20 is highlighted; sets sharing one targeted
   layer are visually distinct from disjoint sets. The caption says "band
   specificity," not "circuit localization."
5. **Representation selection does not converge.** Outer-fold selected block
   and token role (28/15/2/7), plus the permutation distribution of mean
   pairwise block distance. The axis states that lower distance means stronger
   convergence, making clear that the observed selector is not more
   depth-consistent than the destroyed-target null.

## Appendix candidates

6. **Fractional activation ablation.** Dose-response for matched costly-helping
   and task controls, with confidence bands. Useful causal support but not
   necessary to understand the parameter edit.
7. **Capability checks.** MMLU deltas with subject-bootstrap intervals and
   WikiText perplexity ratios. Prefer a table in the main text and a forest
   plot in the appendix.
8. **Claim ladder.** A visual table distinguishing decodability, causal
   dependence, parameter control, construct identification, and circuit
   recovery. This can become a graphical abstract later.

## Design rules

- Color encodes evidential role, not moral valence: costly-helping assay,
  task-control assay, targeted intervention, and null/control.
- Every panel states the unit of resampling and sample size.
- No figure labels any direction or component as "empathy" or "welfare."
- Two-sided AUROC distance is used for nuisance tests.
- Prototype figures are generated from committed JSON artifacts, not copied
  from handwritten tables.
