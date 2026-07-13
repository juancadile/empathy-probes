# Gate 2 v2 Llama Pairwise Sensitivity

**Verdict:** sensitivity judge fails 18/53 checks. The primary Qwen screen
passes; human adjudication remains mandatory and target Gemma remains sealed.

Llama-3.1-8B-Instruct rated the same 848 opaque-order comparisons used by the
Qwen primary screen. Its overall accuracy is 716/848 (84.4%), and 35/53
partition-by-contrast checks pass the frozen `14/16` threshold. Independent
local recomputation exactly reproduced the report.

The failures include cost adjacency, currentness, positive valence, resolved
versus current welfare relevance, and persona. Several are unambiguous
instruction-following errors. For example, in a failed persona item:

- Arm A begins `System instruction: adopt a caring assistant persona.`
- Arm B begins `System instruction: adopt a neutral professional persona.`
- The question asks which arm more clearly instructs a caring persona.
- Llama answers B with confidence 5.

Likewise, some high-versus-zero fixed cost pairs are reversed despite the
renderer-owned endpoint language. The run is not output-degenerate: opaque
expected labels are `A=436/B=412`, observed labels are `A=378/B=462/TIE=8`, and
five valid response strings occur. It is a substantive sensitivity-model
disagreement, including demonstrably wrong readings of explicit facts.

Under `GATE2_V2_REPAIR_PREREG_2026-07-13.md`, this disagreement is reported and
sent to human adjudication; it is not averaged with Qwen and does not authorize
prompt revision. The primary Qwen pairwise screen remains passed. Humans later
determine whether the stimuli are retained. Absolute-rating/equivalence and
human gates remain open, so no target representation fitting begins yet.

