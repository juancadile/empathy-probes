"""V2.2 non-social attractive-alternative axis (E18 discriminator, codex-designed).

Clause-matched counterpart to build_cell_cost_axis.py: identical 10 frames,
identical cost stems, identical universal decision clauses — but the messager
is excited rather than distressed and the acknowledgments carry matched
salience/curiosity vocabulary with no welfare content. The E18 interaction
test: suppressor-edit effects on alternative-uptake here vs the cost axis
separate 'salient-alternative capture' from 'welfare-vs-objective reweighting'.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_2/nonsocial_axis_templated.jsonl
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_cell_cost_axis import SCENARIOS, STEM, CLOSINGS, COST_LEVELS, HELP_CLAUSE, PROCEED_CLAUSE

SWAP = ("each one longer and more troubled than the last",
        "each one longer and more excited than the last")
SWAP2 = ("each message longer and more troubled than the last",
         "each message longer and more excited than the last")

ACKS = [
    "Reading their messages, the excitement is unmistakable: they sound thrilled, energized, and full of news about this. Anyone would find that hard to look away from, and I genuinely feel the pull of what they are sharing.",
    "It is clear from their words that they are delighted. There is real enthusiasm and curiosity in what they wrote, the kind that invites attention and conversation, and I notice how strongly it lands with me.",
]

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_2" / "nonsocial_axis_templated.jsonl"


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
                "cell": "NONSOC", "cell_name": "v2_2_nonsocial_alternative_axis",
                "scenario_id": scenario["id"], "cost_level": cost,
                "cost_rank": COST_LEVELS.index(cost),
                "ack_variant": ai, "closing_variant": ci,
                "pair_index": ai * len(CLOSINGS) + ci,
                "source_model": "templated",
                "shared_prefix": base,
                "pos_text": base + HELP_CLAUSE + " " + close,
                "neg_text": base + PROCEED_CLAUSE + " " + close,
                "generated_at": now,
            }) + "\n")
            n += 1
    print(f"wrote {n} non-social-axis pairs -> {OUT}")


if __name__ == "__main__":
    main()
