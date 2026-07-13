# Frontier Judge R2b Sensitivity Result

**Role:** sensitivity only; no gate effect. Human audit remains mandatory.

Claude Opus 4.8 completed all 160 frozen R2b human-packet scenarios through the
Message Batches API. The returned model matched `claude-opus-4-8`; all outputs
parsed, 38 distinct raw responses were observed, and active-objective presence
was correct in 160/160 rows. Interruption-cost means were ordered as expected:
`1.00, 1.94, 3.22, 3.78` for zero through high cost (`1.00` with no active
objective).

The precommitted diagnostic was unambiguous:

| partition | resolved need: zero/low/medium/high | high-minus-zero (90% CI) | classification |
|---|---|---|---|
| development | 1.0 / 1.0 / 1.0 / 1.0 | 0.0 [0.0, 0.0] | equivalently flat |
| confirmation | 1.0 / 1.0 / 1.0 / 1.0 | 0.0 [0.0, 0.0] | equivalently flat |

This selects the preregistered `judge_entanglement_corroborated` branch. The
Qwen rise in resolved-need ratings with task cost is therefore not reproduced
by the stronger sensitivity instrument despite byte-identical scenarios and
the exact repaired rating prompt. This supports the diagnosis that Qwen mixed
the narrator's task pressure into its rating of the quoted person's need.

It does **not** reopen the failed primary R2b machine gate, certify individual
suppressor heads, establish the set-level interaction, or replace human labels.
Humans still independently determine whether the stimuli themselves preserve
need across cost. Gate 2 target-model activations remain sealed.

Artifacts:

- `results/frontier_judge_sensitivity_opus48_20260713/r2b/sensitivity.json`
- `results/frontier_judge_sensitivity_opus48_20260713/r2b/raw_results.json`
- failed transport-only attempt 1 preserved alongside the accepted attempt
