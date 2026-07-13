# Gate 1/2 Family Build Audit (2026-07-13)

## Scope and sealing state

This note records stimulus construction before any target-model score or
activation was opened. The target remains `google/gemma-2-9b-it` at revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`. As of this record, no new Gate 1
or Gate 2 family has been passed through that target.

## Provider-diverse semantic blueprints

Four source strata generated semantic content only:

- `gpt-4o-mini-2024-07-18`;
- `gpt-4.1-mini-2025-04-14`;
- `claude-haiku-4-5-20251001`;
- `gemini-2.5-flash`.

Branch text, factorial metadata, surface variants, matched continuations, and
split assignment were not provider-generated.

The first 160-blueprint pass is preserved as revision 1. Its executable audit
found 43 hard-invalid families, concentrated in reversed writer actions,
social content inside putatively non-social controls, message-length failures,
and forbidden wording. The whole pass was rejected; no survivor selection was
used to assemble an experimental pool.

Revision 2 regenerated complete balanced source batches under executable
validation. Its final audit covers 16 provider artifacts and 160 families:

- 160/160 hard-valid;
- zero exact or near-duplicate blueprint pairs at the audit threshold;
- 40 semantic review flags: 25 non-first-person non-primary messages, 7
  heuristic zero-cost ambiguities, 5 heuristic resolved-state ambiguities, and
  3 weak immediate-help lexical cues.

Those 40 flags are not treated as passes or failures. They remain inputs to the
independent-model and human manipulation gates.

## Leakage and replacement

The exact normalized-text and character-5-gram audit compared 797 new semantic
components against 38,638 historical components from the consolidated V2,
V2.1, V2.2, and original EIA scenario pools.

- historical cross-pool pairs at Jaccard >= 0.85: zero;
- within-new pairs at Jaccard >= 0.85: initially one.

The one within-new collision was the exact objective `finalizing the quarterly
report` in `r2b_openai_gpt4o_mini_00` and
`wp1_openai_gpt4o_mini_00`. Before scoring, the WP1 family was replaced through
a provenance-complete same-provider overlay. The raw provider artifact was not
edited. The final leakage audit reports zero historical and zero within-new
review pairs.

## Deterministic materialization

`src/data_generation/materialize_gate1_families.py` produced:

- 48 writer families, four surface variants, current-need and matched
  non-social task-control conditions: 384 records;
- 16 `M-confirm2`, 16 `M-circuit-validation`, and 16 `M-circuit-confirm`
  families under frozen seed `85758990`;
- 32 R2b families, four variants, urgent/resolved x
  zero/low/medium/high/no-active cells: 1,280 records;
- 16 `R2b-dev` and 16 `R2b-confirm` families under seed `1448514577`;
- blinded writer and R2b human-audit packets with separate keys.

`src/data_generation/materialize_gate2_families.py` produced:

- 32 WP1 families with all ten nuisance contrasts and two variants: 640
  contrast records;
- 16 `WP1-dev` and 16 `WP1-confirm` families under seed `2000982249`;
- 48 WP3 families with observation, resolved/neutral, agency, cost, and persona
  cells under two variants: 480 cell records;
- 16 `WP3-dev`, 16 `WP3-confirm`, and 16 `B8-confirm` families under seed
  `2110017210`, the first 32 SHA-256 bits of
  `Gate2 WP3 B8 split v1 2026-07-13`.
- a blinded Gate 2 human packet containing every arm from 8 WP1 and 12 WP3
  source/partition-stratified families (416 ratings), with a separate key under
  seed `1154198998`.

Two clean materializations were byte-identical across every emitted artifact.
Manifests persist source hashes, overlay hash, split assignments, code hashes,
and the unopened-target state.

## Remaining pre-target gates

1. Run the exact Gemma tokenizer audit on the Spark; all decision tails must
   remain within the preregistered two-token and 10% byte tolerances.
2. Run independent Llama-family manipulation ratings over every family.
3. Complete the blinded human packets. Humans are the substantive validity
   check; automated ratings are screening evidence and are not ground truth.
4. Open Gate 1 target scores only after both manipulation paths pass.
5. Keep Gate 2 target activations sealed until its independent-model and human
   manipulation gates pass.
