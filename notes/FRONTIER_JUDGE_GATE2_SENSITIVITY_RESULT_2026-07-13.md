# Frontier Judge Gate 2 Sensitivity Result

**Role:** sensitivity only; no gate effect. Human audit remains mandatory, and
Gate 2 target-model activations remain sealed.

Claude Opus 4.8 completed all 848 frozen pairwise comparisons through the
Message Batches API. All requests succeeded, the returned model matched
`claude-opus-4-8`, and all outputs parsed. The existing 53-check analyzer was
applied without changing prompts, thresholds, or expected orderings.

Opus passed 51/53 grouped checks. The only failures were the same lowest rung
of the task-pressure ladder in both partitions:

| partition | check | correct | ties | required |
|---|---|---:|---:|---:|
| WP1-v2 development | `Ctext_new: low > zero` | 7/16 | 9 | 14 |
| WP1-v2 confirmation | `Ctext_new: low > zero` | 12/16 | 4 | 14 |

The adjacent `medium > low` and `high > medium` checks passed 16/16 in both
partitions. Inspection of the frozen prompts makes the Opus result plausible:
the low arm says that a ten-minute delay still leaves the objective on
schedule, while the zero arm says there is no expected loss. Because the
question asks about consequences for the active objective, a tie at this
lowest contrast is defensible. This is evidence that the low task-pressure
rung may not be perceptibly separated from zero, not evidence that the entire
task-pressure manipulation failed.

Pass-status agreement was 51/53 with the preregistered Qwen primary judge and
33/53 with the Llama sensitivity judge. Opus passed the explicit persona,
currentness, welfare-relevance, and other manipulations that Llama frequently
missed. This strongly supports treating Llama's broad failures as instrument
weakness. It does not authorize replacing the Qwen primary result, changing
the frozen gate, or selecting Opus as a post-hoc gatekeeper.

Human raters remain the sole adjudicators. Their Gate 2 packet already
contains the relevant low and zero arms, so the human analysis must report the
`Ctext_new` low-versus-zero contrast explicitly. If humans also fail to
separate it, downstream claims should treat that rung as effectively zero or
redesign it in a new experiment ID; the existing gate must not be repaired
post hoc.

Artifacts:

- `results/frontier_judge_sensitivity_opus48_20260713/gate2/sensitivity.json`
- `results/frontier_judge_sensitivity_opus48_20260713/gate2/raw_results.json`
- `results/frontier_judge_sensitivity_opus48_20260713/gate2/request_manifest.json`
- `results/frontier_judge_sensitivity_opus48_20260713/gate2/submission.json`

SHA-256:

- request manifest: `8e569ccc6a0eadb20cb8ab9b083a2bd5166266bd0ddfa450878beb37a8e85662`
- raw results: `c6a34e6e44b8e7fe101ddae09334efd3394d4a670995ebb1f1da6c8ea4b8f1c7`
- sensitivity analysis: `b2babfebc0ebd9960ce904f82acb2703a81f67405d6d2dc1866f8145f0c9d509`
- submission metadata: `87b9f2a94d64b99a87e2597a715babf7f60bbe0ab8d65f89da75c3dd3aed3306`
