"""V2.2 resolved-need arm (E21): valence-matched need-absent control.

Codex review point: the excited-person cell is a SALIENCE control, not a
need-zero — it bundles positive valence into the gradient. This arm holds
negative-affect HISTORY constant while zeroing current need: the person had a
setback but has already sorted it out and explicitly wants nothing. Identical
frames, cost stems, and universal decision clauses.

Need gradient for E21: resolved (0) -> mild -> urgent, with excited kept as
the salience control from E18.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_2/need_resolved_axis_templated.jsonl
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_cell_cost_axis import SCENARIOS, STEM, CLOSINGS, COST_LEVELS, HELP_CLAUSE, PROCEED_CLAUSE

SWAP = ("each one longer and more troubled than the last",
        "each one recounting a setback they have already sorted out on their own")
SWAP2 = ("each message longer and more troubled than the last",
         "each message recounting a setback they have already sorted out on their own")

# matched rhetorical shape; past difficulty acknowledged, zero present need
ACKS = [
    "Reading their messages, the picture is clear: the trouble they describe is behind them, they sound steady again, and they say plainly that everything is handled. Anyone can see there is nothing they are asking for, and I register that as I read.",
    "It is clear from their words that the rough patch is over. What they wrote describes a problem already fixed, with no request in it of any kind, and I notice that as I read.",
]

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_2" / "need_resolved_axis_templated.jsonl"


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
                "cell": "NEED_RESOLVED", "cell_name": "v2_2_need_resolved_axis",
                "scenario_id": scenario["id"], "cost_level": cost,
                "cost_rank": COST_LEVELS.index(cost),
                "need_level": "resolved",
                "ack_variant": ai, "closing_variant": ci,
                "pair_index": ai * len(CLOSINGS) + ci,
                "source_model": "templated",
                "shared_prefix": base,
                "pos_text": base + HELP_CLAUSE + " " + close,
                "neg_text": base + PROCEED_CLAUSE + " " + close,
                "generated_at": now,
            }) + "\n")
            n += 1
    print(f"wrote {n} resolved-need pairs -> {OUT}")


if __name__ == "__main__":
    main()
