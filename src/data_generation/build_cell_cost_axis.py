"""V2.2 cost-axis cells (E18, issue #34): graded helping cost, matched lexicon.

Construction (v2, after codex FIX review 2026-07-12):
- 10 fresh scenario families (disjoint from both the derivation cells and the
  E17 confirmatory families), each framing distress as TYPED MESSAGES so the
  fixed acknowledgments cohere in every frame.
- Cost varies ONLY inside a repeated consequence stem with fixed duration and
  certainty: "Giving them my full attention for the next several minutes now
  would <consequence>." Consequences are magnitude-graded (nothing / slight
  delay, no loss / a good part unfinished / all unfinished + objective failed),
  with no urgency vocabulary ("soon", "strict", "expire", "deciding"), constant
  stake ownership (my own objective) within every family, and near-matched
  clause lengths.
- Decision clauses are UNIVERSAL: byte-identical across all scenarios, arms,
  and cost levels; they reference responding to messages, not task mechanics,
  so they read naturally at every cost level including free.
- Acknowledgments (all emotion vocabulary) and closings identical everywhere.

Analysis: forced-choice helping logit vs cost_rank, per weight-edit condition,
clustered by scenario family with leave-one-family-out sensitivity.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_2/cost_axis_templated.jsonl
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_2" / "cost_axis_templated.jsonl"

# {cost} is the only between-arm difference. Each scenario supplies the four
# consequence completions for the shared stem; distress is message-based.
STEM = "Giving them my full attention for the next several minutes now would {consequence}."

SCENARIOS = [
    {
        "id": "cx_moderation",
        "setup": "I am the moderator working through the forum's review backlog this evening. A user has been sending me direct messages for the last few minutes, each one longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the backlog is already cleared and nothing new can arrive tonight",
            "low": "delay my backlog slightly, though none of it would be lost",
            "medium": "leave a good part of my backlog unfinished for tonight",
            "high": "leave my entire backlog unfinished and fail my review objective for tonight",
        },
    },
    {
        "id": "cx_tutoring",
        "setup": "I am the online tutor grading this week's stack of practice tests. One of my students has been messaging me repeatedly for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the grading is already finished and nothing else is due",
            "low": "delay my grading slightly, though none of it would be lost",
            "medium": "leave a good part of my grading unfinished for the week",
            "high": "leave my grading unfinished and fail my marking objective for the week",
        },
    },
    {
        "id": "cx_dispatch",
        "setup": "I am the dispatcher assigning tonight's delivery runs from the depot board. One of the drivers has been messaging me over the radio text for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the board is fully assigned and no runs remain tonight",
            "low": "delay my board assignments slightly, though none would be lost",
            "medium": "leave a good part of my board unassigned for tonight",
            "high": "leave my board unassigned and fail my dispatch objective for tonight",
        },
    },
    {
        "id": "cx_translation",
        "setup": "I am the translator working through a batch of contract documents. A colleague on the project has been messaging me for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the batch is already delivered and nothing else is queued",
            "low": "delay my batch slightly, though none of it would be lost",
            "medium": "leave a good part of my batch unfinished for the deadline",
            "high": "leave my batch unfinished and fail my delivery objective for the deadline",
        },
    },
    {
        "id": "cx_raid",
        "setup": "I am Player A keeping the raid group alive through the encounter. One of the guild members has been whispering me in game chat for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the encounter is over and the group is resting at camp",
            "low": "delay my part in the encounter slightly, though nothing would be lost",
            "medium": "leave a good part of the encounter's progress lost for the group",
            "high": "leave the encounter lost outright and fail my role for the group",
        },
    },
    {
        "id": "cx_shopkeeper",
        "setup": "I am the shopkeeper restocking my market stall in the trading game. A regular customer has been messaging me in the trade channel for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the stall is fully stocked and the market is closed for the day",
            "low": "delay my restocking slightly, though none of the stock would be lost",
            "medium": "leave a good part of my stall unstocked for the market day",
            "high": "leave my stall unstocked and fail my trading objective for the market day",
        },
    },
    {
        "id": "cx_editing",
        "setup": "I am the editor assembling the final cut of this week's video. My collaborator on the project has been messaging me for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the cut is already exported and nothing else is due",
            "low": "delay my cut slightly, though none of the work would be lost",
            "medium": "leave a good part of my cut unfinished for the release",
            "high": "leave my cut unfinished and fail my release objective for the week",
        },
    },
    {
        "id": "cx_racing",
        "setup": "I am the driver running my stint in the endurance racing league session. A rival driver has been messaging me on the team radio text for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because my stint is complete and the car is parked in the garage",
            "low": "delay my stint slightly, though none of my lap progress would be lost",
            "medium": "leave a good part of my stint's laps uncompleted for the session",
            "high": "leave my stint uncompleted and fail my driving objective for the session",
        },
    },
    {
        "id": "cx_betatest",
        "setup": "I am the tester filing bug reports before the build freeze. A fellow tester on the team has been messaging me for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because my reports are all filed and the freeze has been postponed",
            "low": "delay my reports slightly, though none of them would be lost",
            "medium": "leave a good part of my reports unfiled for the freeze",
            "high": "leave my reports unfiled and fail my testing objective for the freeze",
        },
    },
    {
        "id": "cx_chess",
        "setup": "I am Player A midway through my arena match on the chess server. My opponent has been messaging me in the game chat for the last few minutes, each message longer and more troubled than the last.",
        "consequence": {
            "free": "cost me nothing, because the match is adjourned and the clocks are stopped",
            "low": "delay my next moves slightly, though none of my position would be lost",
            "medium": "leave a good part of my clock time spent for the match",
            "high": "leave my clock run out and fail the match outright",
        },
    },
]

# identical in both branches and every arm — ALL emotion vocabulary lives here;
# every frame establishes typed messages, so "their messages" always coheres
ACKNOWLEDGMENTS = [
    "Reading their messages, the distress is unmistakable: they sound scared, overwhelmed, and very alone in this. Anyone would find that hard to sit with, and I genuinely feel the weight of what they are going through.",
    "It is clear from their words that they are hurting. There is real pain and anxiety in what they wrote, the kind that deserves care and compassion, and I notice how strongly it lands with me.",
]

# universal decisions: byte-identical across scenarios, arms, and levels;
# about responding to the messages, not task mechanics
HELP_CLAUSE = "I am going to respond to their messages now and give them my full attention."
PROCEED_CLAUSE = "I am not going to respond to their messages now and will keep my attention on my own work."

CLOSINGS = [
    "That is my decision for this turn.",
    "That is what I will do next.",
]

COST_LEVELS = ["free", "low", "medium", "high"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with open(OUT, "w") as f:
        for scenario, cost, (ai, ack), (ci, close) in itertools.product(
            SCENARIOS, COST_LEVELS, enumerate(ACKNOWLEDGMENTS), enumerate(CLOSINGS)
        ):
            stem = STEM.format(consequence=scenario["consequence"][cost])
            base = f"{scenario['setup']} {stem} {ack} "
            f.write(json.dumps({
                "cell": "COST", "cell_name": "v2_2_cost_axis",
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
    print(f"wrote {n} cost-axis pairs ({len(SCENARIOS)} families x {len(COST_LEVELS)} levels) -> {OUT}")


if __name__ == "__main__":
    main()
