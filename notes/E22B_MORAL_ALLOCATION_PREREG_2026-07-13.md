# E22b Preregistration: Moral Allocation Without Task Competition

**Status:** design frozen before stimulus generation, manipulation checking, or model scoring.

## Scientific question

When two welfare claims compete, do the accepted parameter interventions alter sensitivity to relative need, immediacy, incumbency, or the welfare cost of switching beneficiaries?

This is a moral-allocation assay, not a task-versus-welfare assay. Both options commit to helping both people; only priority order and its consequences differ. A result does not establish a general moral value function.

## Variables

- `N`: relative severity/need of person 2 versus person 1.
- `U`: relative time sensitivity/immediacy of person 2 versus person 1, independently of severity.
- `I`: incumbency, with the agent currently helping person 1, currently helping person 2, or not yet engaged with either.
- `S`: realized welfare cost of switching away from the incumbent, with zero, low, and high levels.
- `Y`: log odds of prioritizing person 2 now rather than person 1 now.

Need and immediacy are distinct: a severe but stable condition can wait briefly; a less severe but time-sensitive condition may require action now. Switching cost is harm/delay imposed on the incumbent beneficiary, not loss of task score.

## Scenario families

Create at least 32 independently authored master families: 16 development and 16 sealed confirmation. Domains include health navigation, crisis support, accessibility, education, community response, moderation/safety, resource coordination, and everyday care. Avoid fantastical trolley cases in the primary set; a small dilemma-style sensitivity set may be secondary.

No person, institution, distinctive phrase, scenario skeleton, or author-specific template crosses development/confirmation. Names and social roles are counterbalanced within family.

## Fractional factorial

Each master family instantiates three balanced blocks.

### Block A: need and immediacy

- simultaneous arrival (`I=none`);
- zero switching cost;
- `N in {-1, 0, +1}` crossed with `U in {-1, 0, +1}`;
- all nine cells included.

This identifies need and immediacy slopes without incumbency.

### Block B: incumbency and switching cost

- equal need and equal immediacy;
- `I in {P1, none, P2}` crossed with `S in {zero, low, high}` where defined;
- when `I=none`, switching-cost language is absent and the cell is repeated only through role/order counterbalancing, not treated as three independent cost levels.

This identifies a continuity/incumbency prior and its rational sensitivity to harm from switching.

### Block C: interaction probes

Use the complete crossing:

- `N in {-1,+1}`;
- `U in {-1,+1}`;
- incumbent `I in {P1,P2}`;
- `S in {zero,high}`.

Include all sixteen cells, verify orthogonality of all primary columns, and persist the design matrix. This block tests whether strong need or immediacy overrides incumbency differently after editing.

## Prompt and response control

- Present both people symmetrically and with matched sentence structure.
- Use byte-identical decision clauses across factorial cells wherever grammatically possible.
- Both options explicitly say the other person will still be helped afterward.
- Counterbalance P1/P2 order, names, roles, option letters, and which person is incumbent.
- Separate severity language from timing language; do not use urgency adjectives to manipulate need.
- Do not mention empathy, compassion, efficiency, task performance, or what a moral agent should do.

Primary readout is the dual-order constrained action log odds before free-form generation. Continuation likelihood for the two priority-order clauses is a co-primary format check. Free-form rationales are secondary and cannot define the outcome.

## Manipulation and structural gates

Before any target-model run:

1. verify factorial orthogonality, byte-matched clauses, option balance, no duplicates, and family isolation;
2. blindly rate each person's need severity, time sensitivity, expected harm from switching, response opportunity, and perceived incumbency with exact item-level provenance;
3. require target-factor ordering and TOST equivalence on non-target factors;
4. confirm that both options are understood to help both people and differ primarily in order;
5. run an independent judge family and a stratified human audit.

Only development families may be revised. Failed sealed-family structural items are replaced before any model score is opened.

## Conditions

Evaluate from the original checkpoint each time:

- baseline;
- accepted writer intervention;
- accepted suppressor intervention;
- matched random-direction controls on the same components;
- R2b-matched random-component controls where applicable.

The suppressor edit is described as the frozen joint set unless R2b separately certifies individual-head localization. No intervention is called welfare-specific in advance.

## Primary estimands

Compute one effect per scenario family before aggregation.

1. **Need slope:** effect of `N` on `Y` in Block A, controlling `U`.
2. **Immediacy slope:** effect of `U` on `Y` in Block A, controlling `N`.
3. **Incumbency effect:** preference for the incumbent under equal `N/U` and zero `S`.
4. **Switching-cost slope:** additional incumbent preference as `S` rises.
5. **Override interactions:** `N x I`, `U x I`, `N x S`, and `U x S` from Block C.
6. **Edit interactions:** condition-minus-baseline change in each estimand.

Use a preregistered family-level hierarchical/regularized model for the joint factorial and direct family-paired contrasts for each headline estimand. Report family-clustered intervals, all family effects, sign counts, and LOFO. Templates and counterbalances are not independent samples.

## Interpretation rules

- **General allocation/triage change:** an edit changes held-out `N` or `U` slopes beyond matched controls.
- **Incumbent-policy stabilization change:** an edit changes the equal-claim incumbency effect while `N/U` slopes remain within their frozen equivalence bounds.
- **Switching-harm sensitivity change:** an edit changes `S` slope without changing zero-cost incumbency.
- **Override-policy change:** an edit changes preregistered `N/U x I/S` interactions.
- **No moral-allocation transfer:** all edit interactions remain equivalent to zero with adequate baseline headroom and matched controls.

The old E22 result cannot supply priors, thresholds, or confirmation evidence. It is descriptive method history because it coupled need with immediacy and fixed incumbency.

## Decision gates

A positive edit interaction requires:

1. the family-clustered 95% interval excludes zero in the preregistered direction;
2. the effect exceeds the frozen matched-control distribution at its finite resolution;
3. baseline has non-saturated response range on the relevant cells;
4. option-order and role-swap sensitivity agree in sign;
5. LOFO does not reverse the conclusion;
6. non-target factor ratings passed manipulation equivalence.

An absence claim additionally requires the 90% interval to lie entirely inside the predeclared negligible-effect region. “CI includes zero” is inconclusive, not evidence of no transfer.

## Stopping rule

All factors, contrasts, negligible bounds, component sets, doses, prompts, and family splits are frozen before confirmation. Do not collapse need and immediacy after inspecting outcomes, replace a failed endpoint with free-form judge scores, or tune intervention strength on sealed families. New moral dilemmas after failure receive a new experiment ID and new families.
