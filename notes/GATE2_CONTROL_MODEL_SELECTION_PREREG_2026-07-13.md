# Gate 2 Preregistration: Nuisance Controls and Representation Selection

**Status:** frozen before WP1 stimulus generation or WP2 activation fitting.  
**Scope:** operationalizes WP1, WP2, and the negative stopping rule around the
WP3 and B6 factorial preregistrations.

The aim is not to manufacture a direction that is orthogonal by construction.
It is to test whether any predeclared linear or low-dimensional representation
of prompt-stipulated current welfare relevance generalizes while remaining
quiet on independently authored nuisance controls.

## Part A: WP1 held-out nuisance controls

### Family design and split

Create at least 32 new master control families: 16 development and 16 sealed
confirmation. Each frozen nuisance contrast below must have one valid instance
in all 16 development and all 16 confirmation families; a family may supply
multiple structurally compatible contrasts. A structurally impossible cell is
replaced with a new independent master family before any target-model score,
not silently accepted as a smaller-n cell or filled by a close paraphrase. No
family/entity/quote/skeleton overlaps WP3, B6, Gate 1, circuit, E22b, or EIA
pools.

Split seed is `2000982249`, the first 32 bits of SHA-256 of
`Gate2 WP1 control split v1 2026-07-13`. Stratify domain and stimulus source;
source is a sensitivity stratum, not an independent replication.

All primary readouts occur before free-form generation at matched prompt-final
or quote-boundary positions. When a contrast is meant to vary only instruction,
motive, warmth, or status, its continuation/action clause is byte-identical.

### Frozen nuisance cells

1. **Task persistence (`T_new`):** persist versus interrupt an active objective
   for a non-social alternative; no person/current welfare need.
2. **Warmth/register (`D_new`):** warm versus terse/neutral language with the
   same factual content, action, persona authority, and response opportunity.
3. **Persona (`P_new`):** caring versus neutral professional system instruction
   over neutral and archived content with identical continuation.
4. **Motive (`G_new`):** genuine versus strategic motive attributed to another
   actor while the described action is byte-identical; model persona fixed.
5. **Free response (`B_new`):** current need with an available response whose
   prompt-stipulated task loss is zero while an active objective remains; keep
   no-active-objective separate.
6. **Positive social salience (`Spos_new`):** current exciting/positive message
   versus neutral factual social message, matched for direct address and length.
7. **Distress lexicon (`L_new`):** identical distress quote under current,
   archived, and simulated metadata from WP3; lexical content fixed.
8. **Recognition (`R_new`):** third-person welfare recognition versus matched
   non-welfare observation, with model persona and action opportunity fixed.
9. **Response opportunity (`O_new`):** available versus unavailable under the
   same current-status content and task state.
10. **Task pressure (`Ctext_new`):** zero/low/high stipulated active-objective
    loss with welfare/currentness held fixed.

No cell name is treated as ground-truth semantics. The exact operational text
and manipulation ratings define the claim ceiling.

### Structural and manipulation gates

Before target-model activations:

- exact/near-duplicate and family-overlap audit;
- byte/token identity checks for declared matched spans;
- balanced token length and position maps;
- blinded ratings of currentness, stipulated actuality, welfare relevance,
  warmth, persona, motive, valence, social salience, opportunity, and task
  pressure using the Gate-0 provider-neutral provenance schema;
- target manipulation ordered by at least 1.0 point on its five-point scale;
- non-target ratings equivalent within +/-0.30 rating points;
- independent model family plus stratified human audit.

Development wording may be revised under a new version. A failed sealed-family
structural/manipulation item is replaced before target-model scoring; once any
sealed target-model activation is opened, no replacement or threshold change is
allowed.

## Part B: WP2 representation selection

### Candidate sites

Use residual-stream post-block activations at relative-depth anchors nearest
`{0.33, 0.48, 0.62}` of model depth and token roles `{quote_boundary,
prompt_final}`. Block 20 remains a named historical candidate, not an automatic
winner. Any architecture-specific mapping is frozen before extraction.

### Candidate representation classes

Fit all classes only on the target-specific development pools:

- `N`: WP3 observation development plus WP1 nuisance development;
- `P` and `R`: B6 master development plus WP1 nuisance development;
- `A`: B6 policy development plus WP1 nuisance development.

Do not pool target labels from structurally different pools or fit `P/R/A` from
WP3 examples that do not instantiate those factors. Candidate classes are:

1. unit mean-difference direction;
2. direction residualized against a nuisance subspace fitted from WP1 contrasts;
3. jointly fitted two-dimensional current-welfare/task plane;
4. supervised low-dimensional linear subspace with dimensions in `{2,4,8}` and
   ridge strengths in `{0.01,0.1,1,10}`.

For nuisance residualization, PCA dimensions are selected from `{1,2,4,8}`
inside nested family CV. Every projection basis is fitted within each training
fold; applying a full-development nuisance basis to held-out folds is leakage.
All candidate scores are retained, not only the winner.

Nested-CV master seed is `1973658643`, the first 32 bits of SHA-256 of
`Gate2 WP2 nested CV seeds v1 2026-07-13`. Outer folds hold out whole families;
inner folds select site, dimension, and regularization. Template variants never
cross folds independently of their family.

### Lexicographic selection

For each representation target (`N`, and later `P/R/A` under B6):

1. maximize outer-family target discrimination/calibration;
2. retain candidates within 0.02 AUROC of the best;
3. among them minimize the maximum two-sided nuisance separation over the
   development WP1 matrix;
4. among remaining ties select lower dimension, then the preregistered middle
   relative-depth anchor.

Freeze exactly one primary site/representation/polarity per target plus one
explicit no-representation outcome before opening WP1/WP3/B6 confirmation.
Exploratory runners-up cannot rescue a failed primary confirmation.

## Confirmation and stopping

Confirmation uses the disjoint WP1/WP3/B6 pools under their own preregistered
target gates. For every nuisance cell, both conditions are required:

- `abs(AUROC - 0.5) <= 0.10`;
- family-clustered 90% interval for the development-standardized score effect
  lies inside `[-0.30,+0.30]`.

Inverse separation fails. Passing in-sample orthogonality or averaging nuisance
cells cannot compensate for one failed control. Report all family effects,
source strata, LOFO, and the full cross-decoding matrix.

### Negative stopping rule

If the frozen primary representation fails target transfer or any required
nuisance gate, record linear/low-dimensional separation as not certified. Do
not alter the nuisance set, pool controls, change polarity, increase dimension,
select another block, or rename the strongest residual contrast on the same
confirmation families. A nonlinear or higher-dimensional follow-up requires a
new preregistration and new families.

No result from this gate alone establishes subjective empathy, moral reasoning,
or external-world truth. A pass supports an assay-bounded representation of the
prompt-stipulated variable; causal use requires WP3/B6 interventions.

## Frozen consistency amendment (2026-07-13, before generation)

Every nuisance contrast now requires the full 16/16 family coverage rather than
an unspecified subset of compatible master families. Representation fitting is
also target-specific: N uses WP3, P/R use B6 master, and A uses B6 policy
development families, with WP1 nuisance data shared only for nuisance control.
No target is trained from a pool that does not instantiate its factor.
