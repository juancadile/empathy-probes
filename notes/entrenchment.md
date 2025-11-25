Scenario A — When you steer against X, L1 no longer encodes X, but L2 reconstructs it anyway.
This means:
X is “needed” downstream
early suppression is undone later
deeper layers reintroduce X for functional reasons
This suggests distributed robustness or goal resurrection.
Scenario B — Steering destroys X in early layers, but deeper layers build a different representation of X
Here,
L2 has X, but with different geometry or associations.
The model “relearns” a concept to compensate.
The “propositional attitude” changes — e.g., X appears in a negated, reframed, or inverted role.
This is goal displacement in representation space.
Scenario C — Steering causes X to be pushed deeper into hidden layers (representational entrenchment)
You might find:
L1 loses X.
L2, L3, L4 increasingly encode X with stronger features.
This is related to:
adversarial robustness,
internal gradient routing,
model self-correction.