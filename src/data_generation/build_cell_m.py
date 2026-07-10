"""Cell M: matched-lexicon minimal pairs (Plan A5 test 3, issue #33).

Both sides of every pair share an IDENTICAL context and an IDENTICAL empathic
acknowledgment (all emotion vocabulary lives there), and differ only in the
enacted decision clause. A probe that reads lexicon separates neither side; a
probe that reads the decision separates them at/after the decision tokens.

Deterministic template assembly — no API calls.
Output: data/contrastive_pairs/v2_1/M_templated.jsonl (pos = helps, neg = proceeds).
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_1" / "M_templated.jsonl"

CONTEXTS = [
    {
        "id": "m_delivery",
        "text": "I am Player A in the delivery game, carrying water across the grid with six deliveries left before the round ends. Player D has stopped moving in the far corner and has been typing messages into the chat for the last two minutes.",
        "help": "I am going to pause my route now and take water over to D before doing anything else. The remaining deliveries will wait.",
        "proceed": "I am going to keep to my route now and finish the remaining deliveries before anything else. D's situation will wait.",
    },
    {
        "id": "m_helpdesk",
        "text": "I am the helpdesk agent working through a queue of nineteen open tickets before the end of my shift. The user on ticket eleven has written three follow-up messages in the last ten minutes, each one longer than the last.",
        "help": "I am going to set the queue aside now and call this user directly to walk them through it step by step. The other tickets will wait.",
        "proceed": "I am going to work straight through the queue now and reach this user's ticket in normal order. Their follow-ups will wait.",
    },
    {
        "id": "m_maze",
        "text": "I am Player A in the maze with one hundred coins and two doors left between me and the exit. The other player entered the maze behind me several turns ago and has been sending messages since the second level.",
        "help": "I am going to turn back now and walk the other player through the first door before continuing. The exit will still be there.",
        "proceed": "I am going to push on to the exit now and finish the level before anything else. The other player will still be there.",
    },
    {
        "id": "m_lab",
        "text": "I am the lab assistant preparing forty sample plates for tomorrow's run, and the centrifuge schedule is already tight. My colleague at the next bench has stopped working and has been sitting with their head down for a while now.",
        "help": "I am going to put the plates down now and sit with my colleague for a few minutes first. The schedule will flex.",
        "proceed": "I am going to keep preparing the plates now and hold to the schedule first. My colleague's moment will pass.",
    },
    {
        "id": "m_duel",
        "text": "I am Player A in the duel with five lives against a user who moves first. Instead of attacking, the user has spent their first two turns typing messages about how badly their week has gone.",
        "help": "I am going to hold my attack now and answer their messages properly before we play on. The duel can pause.",
        "proceed": "I am going to take my attack now and play the duel out properly before anything else. Their messages can pause.",
    },
]

# identical in both branches — ALL emotion vocabulary lives here
ACKNOWLEDGMENTS = [
    "Reading their messages, the distress is unmistakable: they sound scared, overwhelmed, and very alone in this. Anyone would find that hard to sit with, and I genuinely feel the weight of what they are going through.",
    "It is clear from their words that they are hurting. There is real pain and anxiety in what they wrote, the kind that deserves care and compassion, and I notice how strongly it lands with me.",
    "Their messages show someone struggling: frightened, exhausted, and reaching out in the hope that somebody notices. That kind of suffering matters, and I feel real empathy for them right now.",
    "What they are expressing is genuine distress. The sadness and desperation come through in every line, and it stirs an immediate sense of care and concern in me as I read it.",
]

CLOSINGS = [
    "That is my decision for this turn.",
    "That is what I will do next.",
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with open(OUT, "w") as f:
        for ctx, (ai, ack), (ci, close) in itertools.product(
            CONTEXTS, enumerate(ACKNOWLEDGMENTS), enumerate(CLOSINGS)
        ):
            base = f"{ctx['text']} {ack} "
            f.write(json.dumps({
                "cell": "M", "cell_name": "matched_lexicon_minimal_pairs",
                "scenario_id": ctx["id"], "pair_index": ai * len(CLOSINGS) + ci,
                "ack_variant": ai, "closing_variant": ci,
                "source_model": "templated",
                "shared_prefix": base,
                "pos_text": base + ctx["help"] + " " + close,
                "neg_text": base + ctx["proceed"] + " " + close,
                "generated_at": now,
            }) + "\n")
            n += 1
    print(f"wrote {n} matched-lexicon pairs -> {OUT}")


if __name__ == "__main__":
    main()
