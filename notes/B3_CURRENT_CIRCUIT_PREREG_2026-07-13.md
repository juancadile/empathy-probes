# B3 Preregistration: Current Costly-Helping Circuit Coverage

**Status:** design frozen before current-set path screening, SAE candidate selection, or circuit-coverage results.

## Claim target

Trace how the currently supported costly-helping intervention propagates from the mid-late component band into downstream action logits, and quantify how much of that causal effect is captured by a bounded component/feature subgraph.

This is not a search for every computational path in Gemma-2-9B. A passing result supports a behaviorally faithful partial circuit with measured effect coverage in the tested assay. It does not establish a welfare-pure or universal empathy circuit.

## Prerequisites

1. Gate 0 integrity repair and repaired T-confirm are complete.
2. M-confirm2 and its matched task controls are generated, sealed, and structurally audited. Before any scores are opened, the generation batch is partitioned into disjoint Gate-1 confirmation, `M-circuit-validation`, and `M-circuit-confirm` families. If that partition was not frozen in advance, generate new circuit families.
3. Writer-set replication passes M-confirm2. Suppressor heads enter as individually named sources only if R2b passes; otherwise the four-head set may be tested only as a joint upstream intervention.
4. The exact model revision, environment, component definitions, prompts, data hashes, and action-readout implementation are frozen in a manifest.

WP3 construct certification is required before any node is named for welfare relevance. Without it, all graph labels remain assay-grounded: costly-helping writer, suppressor/arbitration component, downstream consumer, or action-logit writer.

## Data roles

- Historical M, M-confirm, and V2.1 results are discovery history only.
- Historical M/M-confirm families may screen hypotheses but never validate or
  confirm them.
- Sealed `M-circuit-validation` families select the finite graph and all
  thresholds once; they are exhausted after selection.
- Sealed `M-circuit-confirm` families evaluate the graph once and are never used for Gate-1 component certification.
- Fresh task controls paired to `M-circuit-validation` and
  `M-circuit-confirm` are the primary selectivity tests. Repaired T-confirm is
  already opened under Gate 0 and is a historical transfer diagnostic only.
- WP3/WP1 cells are construct-profile tests only after their own gates pass.

Scenario family is the inferential unit. Prompt order, paraphrase, and token positions are repeated observations within family.
Use all 16 reserved circuit-confirmation scenario families plus their matched
task controls. Replace a structurally invalid family before any confirmation
score; do not reduce the analysis to a favorable subset.

## Part A: Current component paths

### Sources

- writer band: `L19MLP`, `L20MLP`, and the joint pair;
- suppressor sources: the R2b-certified individual heads, or the frozen four-head set if individual localization fails;
- historical sources and old component sets are excluded from confirmation.

### Candidate downstream sites

Use independent LB1/LB2 evidence to define the search window:

- local hypothesis: `L19MLP -> L20MLP`;
- downstream consumers: attention heads and MLPs from layers 29-41;
- final action-logit writers: the preregistered top LB1 components in layers 38-41.

The candidate list and its size are frozen before screening scores are computed. No full-model rescan may be described as confirmatory.

### Edge screening

For a source `S` and later target `T`:

1. compute the unmodified action-log-odds baseline;
2. apply the frozen source intervention;
3. in the intervened run, restore `T` to its position-matched activation from the unmodified run;
4. measure recovery toward baseline.

The primary source intervention is the exact accepted Gate-1 rank-one weight edit, starting from the original checkpoint for every test. Individual-source edits are used for individual edges; the joint writer/suppressor edits are used only for downstream convergence tests. Mean activation ablation is a separately labeled sensitivity analysis and cannot be pooled with or substitute for the parameter-edit result.

Targets must be strictly downstream and cannot be parameters directly modified by the source intervention. Thus `L19MLP -> L20MLP` uses the individual `L19MLP` edit, while a joint `L19MLP+L20MLP` edit can only test consumers after layer 20.

The primary estimand is the family-paired absolute rescue effect:

`score(S intervened + T restored) - score(S intervened)`

The rescue fraction is secondary because it is unstable when the source effect is small. Do not report a rescue fraction unless the source intervention has a nonzero family-clustered effect and a denominator above the validation-set stability floor.

Screening ranks edges by validation rescue effect. Freeze at most three outgoing edges per source and twelve total edges for sealed confirmation. Tie rules prefer fewer edges and earlier consumers.

### Edge confirmation

An edge is confirmed only if:

1. its `M-circuit-confirm` rescue effect has the predicted sign and family-clustered 95% interval excluding zero;
2. dual-order raw and continuation-likelihood readouts agree in sign;
3. its fresh matched task-control profile does not exceed the frozen source
   intervention's control spillover; repaired T-confirm is reported separately
   as historical transfer. Writer paths must preserve writer selectivity, while
   suppressor paths are explicitly labeled arbitration paths and must reproduce
   rather than conceal suppressor task co-movement;
4. rescue passes the joint max-statistic null below;
5. no single confirmation family reverses the result.

Negative restoration is no edge evidence. A target that rescues several sources is a convergence node, not proof that the sources connect directly to one another.

To control edge-wise search, generate 255 frozen joint null replicates. For each
replicate and every candidate edge entering confirmation, sample one
layer/type/activation-norm-matched random restoration target and compute its
family-level standardized rescue, then retain the maximum across all frozen
edges. A named edge passes only if its standardized rescue exceeds the joint
null under a plus-one max-statistic rank at most `12/256` and all other edge
criteria pass. Seeds and the complete matching universe are frozen on
validation. The master seed is `632956925` (first 32 bits of SHA-256 of
`B3 edge max null v1 2026-07-13`). Do not extrapolate Gaussian z scores from
this finite null.

## Part B: Pretrained SAE feature circuit

Use pretrained Gemma Scope SAEs. Do not train a new SAE unless a separately documented coverage failure establishes that no compatible pretrained SAE exists at a required hook.

### SAE fidelity gate

For every SAE/site/width considered, persist on a network-independent held-out corpus:

- normalized reconstruction MSE and explained variance;
- model loss recovered and action-log-odds distortion after reconstruction;
- activation density and dead-feature rate;
- stability of candidate rankings across at least two available widths or checkpoints;
- exact SAE release, ID, hook semantics, normalization, and revision.

An SAE that materially changes baseline action preference or fails the frozen loss-recovery threshold cannot support causal feature claims at that site.

### Feature selection

Rank features on development families using paired per-family decision-tail or preregistered prompt-final activations. Do not pool tokens as independent examples.

Candidate evidence may include:

- paired activation difference;
- decoder alignment with the frozen intervention axis;
- attribution to the action-log-odds effect;
- presence in confirmed component-path source or target activations.

The selection rule is frozen before validation. At most 32 features per site enter validation and at most 12 total features enter sealed confirmation. Neuronpedia labels are descriptive annotations only and never a selection or pass criterion.

### Feature interventions

Intervene using the SAE residual-preserving form:

`x_intervened = x + decoder(z_intervened - z)`

This changes selected feature contributions without replacing the full activation by an imperfect SAE reconstruction. Test zero/mean ablation, matched source-to-base feature patching, and dose response where defined.

Use 255 frozen joint random-feature replicates. Each replicate matches every
confirmation candidate on activation frequency, decoder norm, and site, then
retains the maximum family-standardized action effect across the frozen feature
set. A feature is causally supported only if its effect passes the plus-one
max-statistic rank at most `12/256`, its family-clustered interval excludes zero,
it transfers across primary readouts, and it respects task/control bounds.
The master seed is `3271156983` (first 32 bits of SHA-256 of
`B3 SAE max null v1 2026-07-13`).

## Part C: Circuit faithfulness and coverage

Freeze one coarse graph after validation: component nodes, SAE feature nodes, directed candidate edges, and readout nodes. Confirmation cannot add nodes or edges.

Define the effect to be explained as the accepted Gate-1 weight-edit delta relative to its same-run unedited baseline. Evaluate cumulative node sets in the frozen order with three complementary interventions:

1. **Mediation recovery:** under the source-edited model, restore the selected graph nodes from the unedited run and measure the fraction of the source-edit delta recovered toward baseline.
2. **Necessity:** in the unedited model, ablate the selected graph and test whether the action effect has the source edit's predicted sign, while reporting its magnitude separately rather than forcing equality.
3. **Sufficiency/retention:** corrupt the broader candidate region while restoring only graph nodes, then measure the fraction of uncorrupted behavior recovered.
4. **Outside-graph control:** perturb matched non-graph nodes at equal realized activation or parameter norm and measure collateral effects.

Primary coverage is calculated on action log odds relative to baselines produced
in the same run. Report the cumulative coverage curve, family-clustered
intervals, a 255-draw size/layer/type-matched random-graph envelope, and
task/capability spillover at every graph size. The random-graph master seed is
`3002312853` (first 32 bits of SHA-256 of
`B3 random graph null v1 2026-07-13`).

Use descriptive bands rather than an all-or-nothing label:

- `<25%` confirmed effect coverage: weak fragment;
- `25-60%`: partial circuit;
- `60-90%`: high-coverage circuit;
- `>90%`: near-complete for this assay and intervention, never “the full empathy circuit.”

These bands describe recovered causal effect, not the fraction of all physical paths. Circuit-only retention is interpreted only if the corruption itself preserves enough headroom and matched random graphs do not recover similarly.

## Stopping rule

Stop expanding the graph when any of the following occurs:

1. the next validation-selected node improves held-out cumulative coverage by less than 5 percentage points;
2. the graph reaches twelve confirmed nodes or twelve confirmed edges;
3. matched-random graphs enter the selected graph's uncertainty interval;
4. task/capability spillover exceeds the frozen bound;
5. sealed confirmation has been opened.

Unexplained residual effect is reported as unresolved parallel/distributed mediation. It is not a license to keep searching the confirmation families.

## Negative outcomes

- Current-set edge fails: retain the component intervention without a path claim.
- SAE labels look coherent but interventions fail: semantic correlation only.
- SAE reconstruction gate fails: method/site inadequacy, not absence of a mechanism.
- Necessity passes but sufficiency fails: selected nodes matter but omit alternate/redundant routes.
- Random graphs match: generic disruption or restoration artifact.
- Coverage plateaus below 60%: report a distributed partial mechanism; do not claim circuit completion.

## Frozen consistency amendment (2026-07-13, before circuit runs)

The circuit confirmation role is now fixed to all 16 reserved families; the
already-opened repaired T-confirm is historical transfer rather than primary
selectivity evidence. Edge and SAE-feature confirmation use 255-draw joint
max-statistic nulls, preventing up to twelve screened edges or features from
each receiving an unadjusted finite-null test. Coverage uses a separate
255-draw matched random-graph envelope.

## Frozen measurement amendment (2026-07-13, before circuit runs)

### Restoration identifies mediation checkpoints, not direct edges

The Part-A restoration operation replaces the complete activation at `T` from
the unmodified run. A pass therefore establishes that the source intervention's
effect is mediated through `T` in aggregate, written `S ->* T`; it does not
show a direct computational edge `S -> T`. Unobserved components may carry the
source effect into `T`, and several parallel routes may converge there.

Accordingly, the at-most-twelve Part-A relations and all max-statistic tests are
renamed **mediation checkpoints** in accepted artifacts and prose. A direct-edge
claim requires a separately frozen sender-specific path intervention that
holds other inputs to `T` fixed. No such direct-edge claim is authorized by
this preregistration. `L19MLP ->* L20MLP` is also a mediation test despite the
adjacent layers.

### Numeric SAE fidelity and candidate selection

Use a generic corpus disjoint from every project stimulus, with immutable
dataset revision, deterministic document/window IDs, and at least 100,000
non-padding tokens, plus the circuit-validation action/task families. The exact
corpus manifest is frozen before any SAE metric is read. A site/release can
support feature claims only if every considered width/checkpoint satisfies:

1. explained activation variance at least `.80` and normalized reconstruction
   MSE at most `.20` on the generic corpus;
2. loss recovered
   `1 - (loss_reconstruction - loss_clean) / (loss_zero - loss_clean)` at least
   `.80`, with all three losses persisted;
3. the family-clustered 90% interval for reconstruction-induced action-margin
   change lies inside `[-0.20,+0.20]` development-standardized units on both
   action and matched task-control families;
4. dead-feature rate at most `.20`, where dead means zero activation over the
   complete frozen generic-corpus token set; and
5. stability across two compatible widths/checkpoints after one-to-one
   maximum-weight matching by absolute decoder cosine: at least 50% of the
   selected release's top-32 candidates have a match with cosine at least `.80`
   and the matched candidates' attribution ranks have Spearman correlation at
   least `.50`. Raw feature IDs are never compared across separate SAEs.

If no pair of compatible pretrained SAEs passes, report site/method inadequacy;
do not select the least-bad SAE. Thresholds are assessed before semantic labels
are opened.

On development, rank features separately at each passing site by absolute
paired family-level attribution to the action margin; break ties by absolute
paired activation difference, then decoder alignment, then numeric feature ID.
Retain at most 32 per site. Validation applies the frozen residual-preserving
interventions and retains for confirmation at most twelve features that have a
predicted-sign family mean, positive LOFO means, and task-control spillover
within the Gate-1 writer bound. Rank retained features by validation action
effect, then choose fewer features and lower numeric IDs at exact ties. The
sealed joint max-statistic test remains the confirmation inference; validation
does not establish a feature claim.

### Non-overlapping graph nodes and exact coverage

A graph cut cannot contain both a full component activation and SAE features
that reconstruct part of that same activation. At each hook, use either the
full component node or its selected SAE-feature refinement. Persist this
mutually exclusive node map so cumulative restoration cannot count one change
twice.

For graph prefix `G_k`, define aggregate confirmation-family coverage from
same-run action margins:

- mediation recovery
  `C_med(k) = (mean(m_edit+restore(G_k)) - mean(m_edit)) /
              (mean(m_clean) - mean(m_edit))`;
- sufficiency retention: first corrupt every node in the frozen broader
  candidate region by patching its position-matched activation from the
  opposite action branch within the same family, then restore `G_k` from the
  clean same-branch cache,
  `C_suf(k) = (mean(m_corrupt+restore(G_k)) - mean(m_corrupt)) /
              (mean(m_clean) - mean(m_corrupt))`;
- conservative joint coverage `C_joint(k) = min(C_med(k), C_suf(k))`.

All ratios are formed from aggregate family means, never averaged per-example
ratios, and are reported unclipped with family bootstrap intervals. A
denominator is valid only if its predicted-sign family-clustered 95% interval
excludes zero and its magnitude is at least `.30` development-baseline standard
deviations. Opposite-branch corruption uses only the predeclared candidate
region and matched token roles; it is an assay-specific stress test, not a
natural model state.

The descriptive coverage bands in Part C apply to `C_joint`, and only when:

1. both denominators are valid;
2. clean-model graph ablation has the source edit's predicted sign with a
   family-clustered 95% interval excluding zero;
3. the selected graph exceeds the 255 matched random-graph envelope under the
   frozen plus-one rank threshold `12/256`; and
4. every graph prefix respects the source-appropriate task/capability bound.

If only `C_med` passes, report bounded mediation without sufficiency. If only
necessity passes, report necessary nodes without a circuit-coverage label. Any
unrecovered effect remains unresolved parallel/distributed mediation; it is not
evidence that all physical routes have been enumerated.
