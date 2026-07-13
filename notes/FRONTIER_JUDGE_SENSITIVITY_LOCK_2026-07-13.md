# Frontier Judge Sensitivity Lock

**Status:** frozen before submitting any frontier-judge request.

## Role and non-role

`claude-opus-4-8` is an additional sensitivity instrument. It cannot open,
close, rescue, replace, or modify Gate 1 R2b v2 or Gate 2 v2. Qwen3-14B remains
the preregistered primary machine judge, Llama-3.1-8B remains the preregistered
sensitivity judge, and the blinded human path remains mandatory and unresolved.
The frontier result is reported even when it disagrees with either prior model.

No target-model activation or score is opened by this work.

## Frozen order and prompts

1. Run the 160 R2b rows already selected for the human packet.
2. Collect, validate, and report that batch before submitting Gate 2.
3. Run all 848 existing Gate 2 pairwise requests as a secondary sensitivity.

R2b uses the exact repaired Qwen rating prompt from
`audit_gate1_r2b_v2.qwen_r2b_requests`, filtered to the 160 human-packet rows.
The scenario text is verified byte-identical to the blinded packet. Gate 2 uses
the exact 848 prompts and opaque A/B order produced by
`audit_gate2_v2_pairwise.requests_for`. No wording, question, arm order, sample,
threshold, or parser may be revised after submission under this experiment ID.

The requested and required returned model is `claude-opus-4-8`, listed by the
account API on 2026-07-13 as Claude Opus 4.8. Requests run through the Message
Batches API. A returned-model mismatch invalidates the run.

### Transport amendment after failed attempt 1

Batch `msgbatch_01NhrVyGD11MfmEPJuvrkFHe` returned 160/160 request errors and
zero model outputs because Opus 4.8 rejects the `temperature` parameter as
deprecated. This was learned only from the provider error payload; no rating or
scientific output existed to inspect. For attempt 2, omit `temperature` as
required by the API. Model, prompts, system instruction, opaque order, sample,
analysis, and interpretation rules remain byte-identical. The failed receipt
and all provider error payloads remain preserved.

## Precommitted R2b interpretation

The diagnostic estimand is computed separately in development and confirmation
packet strata. For each sampled family, compute the resolved-speaker current-
need rating at high interruption cost minus its rating at zero cost. Use a
family bootstrap (10,000 draws; fixed seed `1666403293`, the first 32 bits of
SHA-256 of `Frontier judge R2b sensitivity bootstrap 2026-07-13`) for a 90%
interval.

- **Judge entanglement corroborated:** both partition intervals lie wholly
  inside `[-0.50,+0.50]` rating points.
- **Stimulus confound corroborated:** both partition means exceed `+0.50` and
  both 90% intervals exclude zero positively.
- **Mixed/inconclusive:** every other outcome, including split partitions.

All cost-level means, urgent-arm results, individual ratings, raw responses,
and parser failures are retained. The result never changes the already-failed
R2b machine gate or certifies individual suppressor heads. Humans independently
rate the same 160 rows regardless of this result. If humans meet the confound
branch, the set-level conjunctive need-by-cost claim is downgraded; the frontier
judge alone cannot trigger that scientific decision.

## Precommitted Gate 2 interpretation

Apply the existing pairwise analyzer without changing its 53 grouped checks.
Report agreement/disagreement with Qwen and Llama per check and overall. This
is informational only: it cannot satisfy the human gate or change the frozen
primary/sensitivity designations.
