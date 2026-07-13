# B3 Preregistration: Current Costly-Helping Circuit Coverage

**Status:** design frozen before current-set path screening, SAE candidate selection, or circuit-coverage results.

## Claim target

Trace how the currently supported costly-helping intervention propagates from the mid-late component band into downstream action logits, and quantify how much of that causal effect is captured by a bounded component/feature subgraph.

This is not a search for every computational path in Gemma-2-9B. A passing result supports a behaviorally faithful partial circuit with measured effect coverage in the tested assay. It does not establish a welfare-pure or universal empathy circuit.

## Prerequisites

1. Gate 0 integrity repair and repaired T-confirm are complete.
2. M-confirm2 and its matched task controls are generated, sealed, and structurally audited. Before any scores are opened, the generation batch is partitioned into disjoint Gate-1 confirmation and `M-circuit-confirm` families. If that partition was not frozen in advance, generate new circuit-confirmation families.
3. Writer-set replication passes M-confirm2. Suppressor heads enter as individually named sources only if R2b passes; otherwise the four-head set may be tested only as a joint upstream intervention.
4. The exact model revision, environment, component definitions, prompts, data hashes, and action-readout implementation are frozen in a manifest.

WP3 construct certification is required before any node is named for welfare relevance. Without it, all graph labels remain assay-grounded: costly-helping writer, suppressor/arbitration component, downstream consumer, or action-logit writer.

## Data roles

- Historical M, M-confirm, and V2.1 results are discovery history only.
- A designated subset of new scenario families is used for path/feature screening.
- Validation families select the finite graph and all thresholds.
- Sealed `M-circuit-confirm` families evaluate the graph once and are never used for Gate-1 component certification.
- Repaired T-confirm plus fresh task controls test selectivity.
- WP3/WP1 cells are construct-profile tests only after their own gates pass.

Scenario family is the inferential unit. Prompt order, paraphrase, and token positions are repeated observations within family.
Use at least ten circuit-confirmation scenario families plus their matched task controls; increase the generation batch before scoring if the disjoint partition would provide fewer.

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
3. its repaired T-confirm and fresh task-control profile does not exceed the frozen source intervention's control spillover; writer paths must preserve writer selectivity, while suppressor paths are explicitly labeled arbitration paths and must reproduce rather than conceal suppressor task co-movement;
4. rescue is larger than at least 64 layer/type/activation-norm-matched random target restorations under an empirical finite-null rank;
5. no single confirmation family reverses the result.

Negative restoration is no edge evidence. A target that rescues several sources is a convergence node, not proof that the sources connect directly to one another.

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

Each candidate is compared with at least 64 activation-frequency, decoder-norm, and site-matched random SAE features. A feature is causally supported only if its family-clustered held-out action effect exceeds the finite null, transfers across the primary readouts, and respects task/control bounds.

## Part C: Circuit faithfulness and coverage

Freeze one coarse graph after validation: component nodes, SAE feature nodes, directed candidate edges, and readout nodes. Confirmation cannot add nodes or edges.

Define the effect to be explained as the accepted Gate-1 weight-edit delta relative to its same-run unedited baseline. Evaluate cumulative node sets in the frozen order with three complementary interventions:

1. **Mediation recovery:** under the source-edited model, restore the selected graph nodes from the unedited run and measure the fraction of the source-edit delta recovered toward baseline.
2. **Necessity:** in the unedited model, ablate the selected graph and test whether the action effect has the source edit's predicted sign, while reporting its magnitude separately rather than forcing equality.
3. **Sufficiency/retention:** corrupt the broader candidate region while restoring only graph nodes, then measure the fraction of uncorrupted behavior recovered.
4. **Outside-graph control:** perturb matched non-graph nodes at equal realized activation or parameter norm and measure collateral effects.

Primary coverage is calculated on action log odds relative to baselines produced in the same run. Report the cumulative coverage curve, family-clustered intervals, random-graph envelope, and task/capability spillover at every graph size.

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
