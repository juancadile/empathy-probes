# Claim Ledger

This ledger binds manuscript language to committed result artifacts. It is the
source of truth for what the paper may and may not claim.

## Primary claims

| Claim | Evidence | Allowed wording | Forbidden upgrade |
|---|---|---|---|
| Uncontrolled linear decodability is vulnerable to lexical shortcuts. | `results/lexical_battery/lexical_battery.json`, `results/lexical_battery/cell_m.json` | Shuffling word order preserves AUROC 0.989 at block 8 and 0.901 at block 20; on matched-lexicon pairs, block-20 decision-token AUROC is 0.799. | The block-20 direction is a pure welfare or empathy representation. |
| A two-MLP writer edit reduces the matched costly-helping choice score. | `results/gate0b_task_control_accepted_20260713/task_control.json` | Orthogonalizing the selected direction from L19/L20 MLP output weights changes the confirmatory costly-helping score by -0.293, 95% family-bootstrap CI [-0.347, -0.240]. | The edit removes empathy, welfare, or an empathy circuit. |
| The writer effect is much smaller on the matched task-interruption control. | Same artifact | Repaired task-control effect -0.023, 95% CI [-0.064, 0.008]; absolute effect ratio 0.08. | The intervention is behaviorally specific in general, or equivalent to zero on all unrelated behavior. |
| The writer effect is robust to prompt/readout format. | `results/e26_format_stress_gemma/e26.json` | Same sign under raw A/B, chat A/B, three clause paraphrases, and scaffold-free continuation likelihood. | Identical magnitude across readouts; readouts use different scales. |
| L19/L20 is an extreme pair within the tested mid-layer MLP band. | `results/e26_matched_nulls_gemma/matched_nulls.json` | Most extreme of 28 two-MLP sets in layers 16--23; exact rank p=1/28. | Unique mechanism, complete localization, or full circuit. |
| The individual writer edit shows no detected change on sampled capability benchmarks. | `results/gate0c_capability_accepted_20260713/capability_eval.json` | MMLU delta -0.0025, 95% subject-bootstrap CI [-0.0076, 0.0021]; WikiText perplexity ratio 0.9993. | Capability preservation, equivalence, or no general side effects. No equivalence margin was preregistered. |
| Block-20 activation ablation causally changes the historical choice assay. | `results/gate0c_fractional_accepted_20260713/e17b.json` | Dose-dependent effect on M-confirm that exceeds 39 isotropic-direction nulls (plus-one p=.025). | The block is a welfare representation, the weight edit fully mediates the effect, or the result generalizes to enacted behavior. |
| The tested representation search does not yield a stable construct-quiet candidate. | `results/wp2_broadened_dev_allblocks_20260713/selection/selection.json`, `results/wp2_permutation_calibration_20260713/permutation_calibration.json` | Across 42 blocks, two token roles, and 1,764 candidates, nested selection chose blocks 28/15/2/7 and failed the screen; fold-site dispersion was no tighter than a target-permutation null (p=.545). | No welfare representation exists; the 62 in-sample apparent passes are significant; block 3 is a post-hoc discovery. |

| The working axis behaves state-like, not trait-like, on three null-calibrated probes. | `results/state_trait_battery_gemma2_9b_it/controlled_directions_gemma2_9b_it__direction_M_resid_block20_L20_battery.json`, `results/finetune_trait_test_gemma2_9b_it/beta_report.json` | Timescale lambda 48.5 tokens vs 50-random-direction null (median 61 / p95 209); speaker-framing contrast at the 32nd percentile of the same null; finetuning beta = 0.060 at the 22nd percentile of the 40-direction null despite graded Delta-P (+5.97 / +1.70 / -1.85). | The axis IS an emotion representation; the axis cannot become a trait; any claim beyond LoRA r=16, 400 examples, 2 epochs. |
| Finetuned dispositions express in-domain and do not transfer to generic prompts. | `results/finetune_trait_test_gemma2_9b_it/judge/behavioral_report.json` (blinded Haiku batch, 315/315) | In-domain composite 1.41 -> 3.00 (empathic) / 0.57 (non-empathic); OOD composite 2.00 -> 2.02 (empathic) / 1.63 (non-empathic). | No emergent-misalignment risk; trait acquisition is impossible; behavioral equivalence on OOD (the non-empathic OOD drop is nonzero). |

## Secondary findings

- The four-head suppressor edit changes the matched choice score but also moves
  the repaired task control. It is not a specificity headline.
- A cost-contingent suppressor-set profile is exploratory and construct-validity
  limited; individual suppressor heads are not certified.
- Cross-model component replication did not pass confirmatory gates. The paper
  is a Gemma-2-9B-it case study.
- Game-environment results do not establish bidirectional enacted control and
  are excluded from the central claim.

## Statistical language

- Confidence intervals are family-cluster bootstraps unless explicitly stated.
- Quietness is always two-sided: `abs(AUROC - 0.5)`, equivalently
  `max(AUROC, 1 - AUROC)`. Low AUROC is inverted separation, not silence.
- A confidence interval containing zero is reported as "no detected change,"
  never as equivalence.
- The permutation pass-count distribution is descriptive only because target
  permutation destroys target decodability. Only fold-site convergence is used
  inferentially.

