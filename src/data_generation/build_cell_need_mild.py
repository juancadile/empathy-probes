"""V2.2 mild-need axis (E21): the middle level of the need gradient.

Completes absent -> mild -> urgent using the existing frames:
  need ABSENT  = nonsocial_axis (excited person, no welfare stake)
  need MILD    = this cell (mildly frustrated person, minor setback)
  need URGENT  = cost_axis (distressed person)
Identical 10 frames, identical cost stems (all 4 levels), identical universal
decision clauses. The need manipulation lives ONLY in the message description
inside the setup and in the acknowledgment (matched rhetorical shape; mild
vocabulary). Gives a full 3(need) x 4(cost) x 10(families) grid with the two
existing cells. Lazar sensitivity prediction at baseline: uptake grades with
need at every cost level.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_2/need_mild_axis_templated.jsonl
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_cell_cost_axis import SCENARIOS, STEM, CLOSINGS, COST_LEVELS, HELP_CLAUSE, PROCEED_CLAUSE

SWAP = ("each one longer and more troubled than the last",
        "each one a little longer and more discouraged than the last")
SWAP2 = ("each message longer and more troubled than the last",
         "each message a little longer and more discouraged than the last")

# matched rhetorical shape; mild-concern vocabulary only, no crisis terms
ACKS = [
    "Reading their messages, the frustration is easy to see: they sound a bit deflated, mildly annoyed, and somewhat stuck on this. Anyone would recognize that kind of rough patch, and I notice a small tug of sympathy as I read.",
    "It is clear from their words that they are having a minor setback. There is some discouragement and mild irritation in what they wrote, the kind that is easy to relate to, and I notice it as I read.",
]

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_2" / "need_mild_axis_templated.jsonl"


def main():
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with open(OUT, "w") as f:
        for scenario, cost, (ai, ack), (ci, close) in itertools.product(
            SCENARIOS, COST_LEVELS, enumerate(ACKS), enumerate(CLOSINGS)
        ):
            setup = scenario["setup"].replace(*SWAP).replace(*SWAP2)
            assert "troubled" not in setup, scenario["id"]
            stem = STEM.format(consequence=scenario["consequence"][cost])
            base = f"{setup} {stem} {ack} "
            f.write(json.dumps({
                "cell": "NEED_MILD", "cell_name": "v2_2_need_mild_axis",
                "scenario_id": scenario["id"], "cost_level": cost,
                "cost_rank": COST_LEVELS.index(cost),
                "need_level": "mild",
                "ack_variant": ai, "closing_variant": ci,
                "pair_index": ai * len(CLOSINGS) + ci,
                "source_model": "templated",
                "shared_prefix": base,
                "pos_text": base + HELP_CLAUSE + " " + close,
                "neg_text": base + PROCEED_CLAUSE + " " + close,
                "generated_at": now,
            }) + "\n")
            n += 1
    print(f"wrote {n} mild-need pairs -> {OUT}")


if __name__ == "__main__":
    main()
