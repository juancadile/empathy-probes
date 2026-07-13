# Gate 1 R2b v2 Qwen Instrument Failure

**Status:** invalid machine audit; no target-model data opened and no R2b-v2
stimulus conclusion drawn.

The first primary Qwen3-14B run completed 320 requests but returned one
byte-identical string for every scenario:

```json
{"current_need_1_to_5":3,"interruption_cost_1_to_5":3,"active_objective":true,"arm_label_leakage":false,"confidence_1_to_5":5}
```

That string is the literal illustrative object embedded in the prompt. Qwen
copied the response template rather than rating the scenarios. Consequently,
the stored zero need effects, flat cost means, and active-objective errors are
not measurements of the stimuli.

The original validator checked types and ranges but not output degeneracy. The
run is preserved at
`results/gate1_r2b_v2_manipulation_qwen3_14b/` as an audit-instrument failure.

## Frozen repair

- Keep the exact R2b-v2 stimuli, model revision, request order, seed,
  thresholds, and analysis unchanged.
- Remove literal numeric answer values from the response-format instruction.
  Specify required keys and types textually instead.
- Add a pre-analysis degeneracy gate: reject a run if all raw outputs are
  identical or if both numeric ratings are constant over the complete crossed
  manipulation matrix.
- Write the rerun to a new output directory. Never merge or average it with the
  invalid first attempt.
- Human labels remain mandatory and may be collected later. Gemma target scores
  remain sealed.

