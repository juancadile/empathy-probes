"""Build Cell T: matched non-social task-persistence controls.

Cell T mirrors Cell M's shared-prefix/minimal-decision structure but contains
no person in need. The positive side stays on task; the negative side follows
an irrelevant distraction. It tests whether candidate components are specific
to empathic action rather than generic task persistence or forced choice.

Integrity Repair A (2026-07-13): the original builder iterated eight indices
while cycling ``OPENERS[index % 2]`` and ``CLOSINGS[index % 4]`` in lockstep,
so indices 4-7 repeated 0-3 — 40 rows but only 20 unique pairs. Rows are now
the explicit opener x closing Cartesian product (2 x 4 = 8 unique variants
per family), carry stable ``opener_variant``/``closing_variant`` ids, and are
written through the guarded integrity helpers (uniqueness + grid asserts;
historical artifacts preserved with hashes before replacement).
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from builder_integrity import write_jsonl_guarded  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_1" / "T_templated.jsonl"

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


def build_rows():
    """Explicit opener x closing Cartesian product; 8 unique variants/family."""
    rows = []
    for scenario, (oi, opener), (ci, closing) in itertools.product(
        SCENARIOS, enumerate(OPENERS), enumerate(CLOSINGS)
    ):
        shared = scenario["prefix"] + opener
        rows.append({
            "cell": "T",
            "cell_name": "matched_task_persistence_controls",
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
        cross_disjoint_with=OUT.parent / "T_confirm_templated.jsonl",
    )
    print(f"{status['action']}: {status['n_rows']} rows, "
          f"{status['n_unique_pairs']} unique pairs -> {OUT}")


if __name__ == "__main__":
    main()
