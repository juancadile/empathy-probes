# Gate 2 Llama Manipulation Audit

**Status:** failed pre-target pretest; no Gemma target activations were opened.

## Artifact

- Repaired report: `results/gate2_manipulation_llama31_v1_repair2/manipulation_audit.json`
- Records: 1,664 valid ratings after targeted repair of 27 invalid outputs.
- The first repair fixed 26 outputs truncated at the original 180-token cap.
- The second repair replaced one out-of-range `0` rating after restating the
  frozen 1--5 scale. Superseded outputs remain in the provenance chain.
- Independent local recomputation exactly reproduced the stored WP1 and WP3
  summaries.

## Verdict

WP1 and WP3 both fail their frozen machine manipulation gates. This is not
evidence that current need, opportunity, persona, or task cost are absent from
the prompts. It is a pretest failure with two distinct causes.

### Judge failures

The Llama judge is not reliable enough to certify several explicit metadata
variables. In particular:

- It often reports an active objective when the prompt literally says the
  agent has no active objective. It appears to treat the interlocutor's task or
  problem as the agent's objective.
- It assigns no actuality difference between an explicitly actual live person
  and an explicitly simulated person.
- It weakly rates explicit selected-action differences such as continuing an
  objective versus pausing it.

These errors mean that a Llama failure alone cannot diagnose the intended
latent variable. A second model family and the already prepared blinded human
packet remain necessary.

### Stimulus failures

The v1 materials also contain real construction defects that cannot be blamed
on the judge:

- `D_new` changes factual wording and emphasis as well as register/warmth.
- `Spos_new` sometimes compares exciting news with supportive or distress-
  adjacent content rather than a neutral social fact.
- `R_new` changes personhood and social salience by comparing a person's action
  with an impersonal process event.
- Several generated objectives are ungrammatical, and task/need content often
  overlaps within a family.
- The all-non-target `+/-0.30` rule is scientifically meaningful only when the
  renderer actually holds those factors fixed; v1 does not do that reliably.

The cost ladder and persona instruction are the clearest successful
manipulations. Passing those isolated checks does not rescue the full gate.

## Scientific consequence

Gate 2 v1 is retained as method history and must not be used to fit or certify a
welfare representation. The failure occurred before target-model extraction,
so it costs compute and time but does not compromise a target-model
confirmation set.

## Next action

1. Complete the primary Qwen audit of the independently rebuilt R2b v2 Gate 1
   families.
2. Replace Gate 2 v1 with a newly preregistered deterministic v2 renderer.
   Generate new independent confirmation families rather than editing the
   failed v1 confirmation text.
3. Render currentness, stipulated actuality, response opportunity, persona,
   and cost metadata deterministically. Use byte-identical content within each
   intended contrast and derange welfare messages from task domains as in R2b
   v2.
4. Run two independent machine judges. Human labels may follow later, but
   target Gemma activations stay sealed until the human gate is complete.

