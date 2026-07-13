# Gate 2 v2 Qwen Pairwise Audit

**Verdict:** primary pairwise machine screen passes. Absolute-rating and human
gates remain pending; no target-model activations were opened.

## Artifact

- Report: `results/gate2_v2_pairwise_qwen3_14b/pairwise_audit.json`
- Model: Qwen3-14B, pinned revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Requests: 848 opaque-order A/B comparisons.
- Checks: 53 partition-by-contrast checks over WP1 development/confirmation and
  WP3 development/confirmation/B8-confirm.
- Independent local recomputation exactly reproduced the stored report.

## Result

All 53 checks pass the frozen `14/16` rule. Overall accuracy is 836/848
(98.6%). Expected labels were balanced by deterministic opaque order
(`A=436`, `B=412`); observed choices track that balance (`A=431`, `B=411`,
`TIE=6`), so this is not a one-label or template-copying artifact.

The primary WP3 variables pass in every partition:

- current versus archived status;
- stipulated actual versus fictional person;
- current need versus resolved and neutral controls;
- response available versus unavailable;
- caring versus neutral persona; and
- high versus zero task pressure.

WP1 task persistence, warmth, persona, motive, active-objective state,
positive valence, recognition, opportunity, and status/actuality controls also
pass in both development and confirmation.

The closest cells to threshold are deliberately adjacent cost steps:

- zero to low: 14/16 in WP1 development and confirmation, with two ties each;
- medium to high: 14/16 in WP1 confirmation and 15/16 in development.

Those results pass but should remain visible. The pairwise screen establishes
that Qwen can identify the intended ordering; it does not establish the
absolute effect-size or equivalence requirements.

## Consequence

Gate 2 v2 advances to its sensitivity/absolute-rating and human manipulation
checks. The 444-row blinded human packet is ready. The user has chosen to defer
human labeling; therefore Gemma representation fitting and confirmation remain
sealed. No welfare-pure representation claim is currently reopened.

