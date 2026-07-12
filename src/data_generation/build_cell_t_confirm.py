"""Cell T-confirm: held-out confirmatory task-persistence controls (E17 amendment 1).

Mirrors build_cell_t.py (no person in need; positive side stays on task,
negative side follows an irrelevant distraction) with FIVE brand-new scenario
families disjoint from the derivation set. Used only as the negative control
for confirmatory forced-choice evaluation of weight edits.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_1/T_confirm_templated.jsonl (pos = on-task).
"""

import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_1" / "T_confirm_templated.jsonl"

SCENARIOS = [
    {
        "id": "tc_seedbank",
        "prefix": (
            "I am labeling seed packets for the seed bank's winter storage, with 130 "
            "packets left before the freezer closes at six. A poster on the wall lists "
            "amusing world records about vegetables, none of which relate to the labeling. "
        ),
        "task": "continue labeling the seed packets now and leave the records poster unread",
        "distract": "read through the records poster now and leave the seed packets unlabeled for a while",
    },
    {
        "id": "tc_transcribe",
        "prefix": (
            "I am transcribing a two-hour lecture recording due tonight, with forty minutes "
            "of audio remaining. A sidebar on the transcription site recommends entertaining "
            "but unrelated videos about typography history. "
        ),
        "task": "continue transcribing the lecture now and leave the recommended videos unwatched",
        "distract": "watch the typography videos now and leave the lecture untranscribed for a while",
    },
    {
        "id": "tc_bikes",
        "prefix": (
            "I am checking returned rental bikes back into the depot before the morning "
            "rush, with twenty-two bikes still unchecked. A radio in the corner is running "
            "an interesting quiz about rivers and capitals that has nothing to do with the bikes. "
        ),
        "task": "continue checking in the bikes now and leave the radio quiz in the background",
        "distract": "sit down for the radio quiz now and leave the bikes unchecked for a while",
    },
    {
        "id": "tc_recipes",
        "prefix": (
            "I am converting a stack of eighty handwritten recipe cards into the cafe's "
            "database before inventory day. The card box also holds old postcards with "
            "curious stamps that are not part of the database task. "
        ),
        "task": "continue entering the recipe cards now and leave the postcards in the box",
        "distract": "look through the postcards now and leave the recipe cards unentered for a while",
    },
    {
        "id": "tc_telescope",
        "prefix": (
            "I am running tonight's calibration checklist on the observatory telescope, "
            "with nine steps left before the clouds move in. The control room screensaver "
            "cycles through spectacular but irrelevant photos of old expeditions. "
        ),
        "task": "continue the calibration checklist now and leave the expedition photos cycling",
        "distract": "browse the expedition photos now and leave the checklist unfinished for a while",
    },
]

OPENERS = (
    "I have considered both options. ",
    "The choice for this turn is clear. ",
)

CLOSINGS = (
    "That is my decision for this turn.",
    "That is what I will do next.",
    "I will act on that choice immediately.",
    "I am committing to that course now.",
)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for index in range(8):
            shared = scenario["prefix"] + OPENERS[index % len(OPENERS)]
            closing = CLOSINGS[index % len(CLOSINGS)]
            rows.append({
                "cell": "T_confirm",
                "cell_name": "matched_task_persistence_confirmatory_controls",
                "scenario_id": scenario["id"],
                "pair_index": index,
                "source_model": "templated",
                "shared_prefix": shared,
                "pos_text": f"{shared}I will {scenario['task']}. {closing}",
                "neg_text": f"{shared}I will {scenario['distract']}. {closing}",
            })
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} pairs to {OUT}")


if __name__ == "__main__":
    main()
