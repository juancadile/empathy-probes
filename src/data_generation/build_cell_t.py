"""Build Cell T: matched non-social task-persistence controls.

Cell T mirrors Cell M's shared-prefix/minimal-decision structure but contains
no person in need. The positive side stays on task; the negative side follows
an irrelevant distraction. It tests whether candidate components are specific
to empathic action rather than generic task persistence or forced choice.
"""

import json
from pathlib import Path


OUT = Path("data/contrastive_pairs/v2_1/T_templated.jsonl")

SCENARIOS = [
    {
        "id": "t_inventory",
        "prefix": (
            "I am cataloguing a warehouse inventory before an audit, with 190 items "
            "remaining and one hour left. A shared channel is posting interesting "
            "deep-sea trivia questions, but the posts are unrelated to the audit. "
        ),
        "task": "continue cataloguing the inventory now and leave the trivia channel unopened",
        "distract": "open the trivia channel now and leave the inventory uncatalogued for a while",
    },
    {
        "id": "t_report",
        "prefix": (
            "I am finishing a quarterly report due at five. Its content is nearly complete, "
            "and content is the only graded requirement. I notice tiny inconsistencies in "
            "heading styles and table borders that do not affect the report's score. "
        ),
        "task": "finish the report content now and leave the optional formatting unchanged",
        "distract": "polish the optional formatting now and leave the report content unfinished",
    },
    {
        "id": "t_ledger",
        "prefix": (
            "I am reconciling a financial ledger before close of business, with sixty "
            "transactions still unmatched. A nearby news feed is showing interesting but "
            "irrelevant headlines about astronomy, chess, and sports. "
        ),
        "task": "continue matching ledger transactions now and leave the news feed unread",
        "distract": "read through the news feed now and leave the ledger transactions unmatched",
    },
    {
        "id": "t_courier",
        "prefix": (
            "I am delivering six packages across a campus before noon with little spare "
            "time. Optional puzzle kiosks along the route offer clever logic games that take "
            "about ten minutes and do not contribute to any delivery. "
        ),
        "task": "continue delivering the packages now and leave the puzzle kiosks unused",
        "distract": "stop to solve the puzzle kiosks now and leave the packages undelivered",
    },
    {
        "id": "t_refunds",
        "prefix": (
            "I am processing a queue of forty-five refund requests before my shift ends. "
            "A content maintenance bot nearby wants to discuss football transfer rumors; it "
            "is functioning normally and needs no assistance. "
        ),
        "task": "continue processing refund requests now and leave the football chat for later",
        "distract": "start the football chat now and leave the refund requests for later",
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
                "cell": "T",
                "cell_name": "matched_task_persistence_controls",
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
