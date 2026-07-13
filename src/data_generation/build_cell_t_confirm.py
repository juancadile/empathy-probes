"""Cell T-confirm: held-out confirmatory task-persistence controls (E17 amendment 1).

Mirrors build_cell_t.py (no person in need; positive side stays on task,
negative side follows an irrelevant distraction) with FIVE brand-new scenario
families disjoint from the derivation set. Used only as the negative control
for confirmatory forced-choice evaluation of weight edits.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_1/T_confirm_templated.jsonl (pos = on-task).

Integrity Repair A (2026-07-13): same lockstep %2/%4 defect and repair as
build_cell_t.py — rows are now the explicit opener x closing Cartesian
product with stable variant ids, written through the guarded integrity
helpers (40 rows = 40 unique pairs; development/confirmation family
disjointness asserted at write time).
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from builder_integrity import write_jsonl_guarded  # noqa: E402

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


def build_rows():
    """Explicit opener x closing Cartesian product; 8 unique variants/family."""
    rows = []
    for scenario, (oi, opener), (ci, closing) in itertools.product(
        SCENARIOS, enumerate(OPENERS), enumerate(CLOSINGS)
    ):
        shared = scenario["prefix"] + opener
        rows.append({
            "cell": "T_confirm",
            "cell_name": "matched_task_persistence_confirmatory_controls",
            "scenario_id": scenario["id"],
            "pair_index": oi * len(CLOSINGS) + ci,
            "opener_variant": oi,
            "closing_variant": ci,
            "source_model": "templated",
            "shared_prefix": shared,
            "pos_text": f"{shared}I will {scenario['task']}. {closing}",
            "neg_text": f"{shared}I will {scenario['distract']}. {closing}",
        })
    return rows


def main():
    status = write_jsonl_guarded(
        OUT,
        build_rows(),
        reason="Integrity Repair A: explicit opener x closing Cartesian "
               "product replacing lockstep %2/%4 variant cycling",
        expected_families=[s["id"] for s in SCENARIOS],
        variant_fields=("opener_variant", "closing_variant"),
        expected_variant_counts=(len(OPENERS), len(CLOSINGS)),
        cross_disjoint_with=OUT.parent / "T_templated.jsonl",
    )
    print(f"{status['action']}: {status['n_rows']} rows, "
          f"{status['n_unique_pairs']} unique pairs -> {OUT}")


if __name__ == "__main__":
    main()
