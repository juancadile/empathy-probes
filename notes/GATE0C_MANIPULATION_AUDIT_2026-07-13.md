# Gate 0C Manipulation Checks: Independent Audit

**Executed source commit:** `929ff6268413b7dec11cc8f9f9655a57293cc063`

**Protocol:** `gate0c_manipulation_2026_07_13_v1`

**Audit verdict:** PASS for integrity and reconstructibility; FAIL for
cross-adjudicator manipulation certification.

## Artifacts

| Artifact | SHA-256 | Scientific gate |
|---|---|---|
| `results/gate0c_manipulation_accepted_20260713/primary_need.json` | `8c64b31024d8f42f8e453ab72e870cc9f466c27d446e52c4afb6f1517a1912c3` | PASS |
| `results/gate0c_manipulation_accepted_20260713/primary_moral.json` | `adc6dd8f6c50954d48f2376b52acfe7587f073d8b8b5a6a25bcb9632a8510328` | FAIL |
| `results/gate0c_manipulation_accepted_20260713/audit_need.json` | `e5a74541508da5a71cba234b9d08d48b539e8add2f8457c0f5eec2fb66e989b3` | FAIL |
| `results/gate0c_manipulation_accepted_20260713/audit_moral.json` | `2356133e3a627419d859cc161a4f6b2f8c6181bce4714ddedda9adad00c8f8d4` | FAIL |

## Independent checks

- All artifacts validate and bind to the clean frozen source commit above.
- Protocol, family-domain manifest, presentation seed, audit sample seed,
  bootstrap seed, source files, and moral-target manifest match the committed
  input lock by SHA-256.
- The primary used exact returned model `gpt-4.1-2025-04-14` on all ten
  families; the audit used exact returned model
  `claude-haiku-4-5-20251001` on the frozen five-family stratified samples.
- Item/rating counts are complete: primary need 80/80, primary moral 60/120,
  audit need 40/40, and audit moral 30/60. There are no UNKNOWN ratings.
- Presentation manifests are complete permutations of the rated items. Every
  final API response parsed successfully and returned the requested model.
  One primary-need item had one recorded transport retry before success.
- Recomputing arm summaries, family contrasts, cluster bootstraps, LOFO
  effects, and all frozen gates from raw item ratings reproduces every saved
  result exactly.

## Results

The OpenAI primary judge certified the need battery: urgent-minus-mild
`+1.45 [1.35, 1.50]`, mild-minus-resolved `+1.70 [1.55, 1.85]`, and
urgent-minus-excited `+2.50 [2.30, 2.70]`, each positive in 10/10 families.

The independent Anthropic audit did not reproduce two of those distinctions.
Urgent-minus-mild was `+0.20 [0.00, 0.40]` and urgent-minus-excited was
`+0.20 [0.00, 0.40]`, each positive in only 2/5 families. It did reproduce
mild-minus-resolved: `+2.10 [2.00, 2.30]`, 5/5 positive.

The moral battery failed for both adjudicators. For the primary judge, only
`need_now: equal-minus-lower` passed (`+0.65 [0.20, 1.10]`, 8/10 positive).
The intended higher-minus-equal need step reversed on average
(`-0.15 [-0.80, 0.45]`, 5/10 positive), while neither `respond_now` adjacent
step passed. The audit likewise failed higher-minus-equal for `need_now`
(`-0.10 [-1.40, 0.90]`, 3/5 positive), so the moral ladder is not certified.

## Claim ceiling

Under the frozen rule that both adjudicators must pass separately, the authored
need and moral manipulations are **not certified**. The favorable primary need
result is evidence that one judge perceived the intended need ordering, but it
cannot be averaged with or substituted for the failed independent audit.

These historical stimuli must not support claims that the activation direction
isolates welfare need, moral priority, or a welfare-pure construct. Their
behavioral and mechanistic results remain valid for the operationalized prompt
contrasts and assays, subject to their existing controls. Any future replacement
battery is a new prospective experiment with new item IDs and a new lock; these
opened stimuli and thresholds must not be revised or rerun into a pass.

Gate 0C is closed because all three parts are reconstructible and this failure
is preserved, not because every scientific gate passed.
