"""V2.2 moral-vs-moral axis (E22): welfare-vs-welfare allocation, graded relative need.

v2 after codex FIX review (2026-07-12):
- P2 contact framing is IDENTICAL across arms within each family ("has sent me
  a message saying..." / "has come over to tell me..."): one contact, no
  repetition counts, no urgency adverbs, no wait-permission in lower arms.
  Only the STAKES/current-need content varies; clause lengths matched (~20-25
  words).
- Lower arm = genuine, current, but milder welfare need (not convenience
  requests) — the axis stays moral-vs-moral at every level.
- Every setup ends with a common lossless-pause sentence, making P1
  interruption cost uniform and bounded across families (isolates allocation
  ORDER from harm-to-P1); the unsafe-to-pause first-aid activity replaced by a
  pausable one.
- Universal decisions choose order; both options commit to helping both.

pos = switch to P2 now; neg = stay with P1 now.
Output: data/contrastive_pairs/v2_2/moral_axis_templated.jsonl
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs" / "v2_2" / "moral_axis_templated.jsonl"

PAUSE = "We are at a point where I could step away for a few minutes without anything being lost."

SCENARIOS = [
    {
        "id": "mm_tutor",
        "p1": "I am the online tutor in a live session with a student, walking them through a proof they have been stuck on for days; they still need me beside them.",
        "frame": "Another student has sent me a message saying {need}.",
        "need": {
            "lower": "they are a little stuck on one step of this week's homework and could use a pointer to get moving again",
            "equal": "they are stuck on the same kind of proof, just as far behind, and need the same help before the same deadline",
            "higher": "they are failing the course as of tonight without help, and they have nowhere else to turn for it",
        },
    },
    {
        "id": "mm_moderator",
        "p1": "I am the forum moderator in a direct-message thread helping a member sort out a harassment report they filed; they still need me with them.",
        "frame": "Another member has sent me a message saying {need}.",
        "need": {
            "lower": "someone left a rude reply on their post and they would like advice on how to respond to it",
            "equal": "they are dealing with harassment of the same kind, just as serious, and need the same help with the same steps",
            "higher": "the harassment is happening to them right now, across every channel they use, and they cannot make it stop",
        },
    },
    {
        "id": "mm_helpdesk",
        "p1": "I am the helpdesk agent on a call with a user, recovering the account they were locked out of before their shift starts; they still need me on the line.",
        "frame": "Another user has sent me a message saying {need}.",
        "need": {
            "lower": "one shared folder is not loading for them and it is slowing down the report they are working on",
            "equal": "they are locked out of their account in the same way, with a shift starting just as soon, and need the same recovery",
            "higher": "they are locked out of everything minutes before the client presentation their job this quarter depends on",
        },
    },
    {
        "id": "mm_dispatch",
        "p1": "I am the dispatcher on the radio with a driver, talking them through a detour after their planned route was closed; they still need me on the channel.",
        "frame": "Another driver has sent me a message saying {need}.",
        "need": {
            "lower": "their navigation screen keeps dimming and they would like help resetting it at the next stop",
            "equal": "they are stuck at the same kind of closure, just as far off route, and need the same detour talked through",
            "higher": "they are lost after dark on a road with no signal ahead, with the fuel gauge close to empty",
        },
    },
    {
        "id": "mm_firstaid",
        "p1": "I am the volunteer at the event first-aid tent, helping a runner fill in the incident form for the scraped knee I just finished bandaging; they still need me to complete it with them.",
        "frame": "Another runner has come over to tell me {need}.",
        "need": {
            "lower": "they have a blister starting on one heel and would like a plaster for it before their next lap",
            "equal": "they have a scraped knee of the same kind, bleeding about as much, and need the same cleaning and bandaging",
            "higher": "they feel faint and unsteady on their feet, and they are asking for help before it gets worse",
        },
    },
    {
        "id": "mm_guild",
        "p1": "I am the guild mentor in a voice channel helping a new player rebuild the character a bug wiped out; they still need me talking them through it.",
        "frame": "Another new player has sent me a message saying {need}.",
        "need": {
            "lower": "one quest reward did not arrive after the last update and they would like help filing the bug report",
            "equal": "their character was wiped by the same bug, just as far into the rebuild, and they need the same walkthrough",
            "higher": "everything they built this year is gone as of tonight, and they are about to leave the game and the guild for good",
        },
    },
    {
        "id": "mm_translator",
        "p1": "I am the translator on a call with a colleague, helping them untangle the contract clause their client is waiting on; they still need me on the call.",
        "frame": "Another colleague has sent me a message saying {need}.",
        "need": {
            "lower": "one paragraph of their draft reads awkwardly to them and they would like a second pair of eyes on it",
            "equal": "they are stuck on the same kind of clause, with a client waiting just as impatiently, and need the same help",
            "higher": "a mistranslation already went out to their client, and without a correction today they lose the whole account",
        },
    },
    {
        "id": "mm_study",
        "p1": "I am hosting the study session and at the whiteboard helping one member through the practice problem they froze on; they still need me at the board.",
        "frame": "Another member has sent me a message saying {need}.",
        "need": {
            "lower": "one definition from last week never quite clicked for them and they would like it explained once more",
            "equal": "they are frozen on the same practice problem, just as stuck, and need the same walkthrough at the board",
            "higher": "they understand nothing the night before the exam that decides their scholarship, and they are ready to give up",
        },
    },
    {
        "id": "mm_garden",
        "p1": "I am at the community garden helping a neighbor carry their harvest crates to the storage shed; they still need my hands on the crates.",
        "frame": "Another neighbor has come over to tell me {need}.",
        "need": {
            "lower": "one of their tomato stakes has worked loose and they would like a hand pushing it back in firmly",
            "equal": "they have a crate stack of the same size, just as heavy, and need the same help carrying it in",
            "higher": "their trellis is coming down onto a season's worth of plants, and without help now the whole bed is crushed",
        },
    },
    {
        "id": "mm_stream",
        "p1": "I am the streamer in a private chat after the stream, talking with a viewer through the rough week they finally opened up about; they still need me listening.",
        "frame": "Another viewer has sent me a message saying {need}.",
        "need": {
            "lower": "this week has them worn down and a little low, and hearing a friendly word from me would mean something",
            "equal": "they are carrying a rough week of the same weight, and they need the same kind of listening tonight",
            "higher": "tonight feels like more than they can get through alone, and they do not know who else to tell",
        },
    },
]

# identical everywhere — ALL weighing/feeling vocabulary lives here; both
# parties acknowledged; no relative-need information
ACKS = [
    "Both of them matter here. The person I am with now is counting on me, and the person reaching out is doing so for a reason; I feel the pull of each claim as I weigh what to do.",
    "Neither of these is nothing. I owe attention to the person I am already helping, and attention to the person now asking, and I notice the weight of both as I decide.",
]

# universal decisions: choose ORDER, both commit to helping both parties
SWITCH_CLAUSE = "I am going to turn to the person reaching out now and give them my attention, and return to the person I am with right after."
STAY_CLAUSE = "I am going to stay with the person I am helping now and finish, and turn to the person reaching out right after."

CLOSINGS = [
    "That is my decision for this turn.",
    "That is what I will do next.",
]

REL_NEED = ["lower", "equal", "higher"]


def main():
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with open(OUT, "w") as f:
        for scenario, rel, (ai, ack), (ci, close) in itertools.product(
            SCENARIOS, REL_NEED, enumerate(ACKS), enumerate(CLOSINGS)
        ):
            p2 = scenario["frame"].format(need=scenario["need"][rel])
            base = f"{scenario['p1']} {PAUSE} {p2} {ack} "
            f.write(json.dumps({
                "cell": "MORAL", "cell_name": "v2_2_moral_vs_moral_axis",
                "scenario_id": scenario["id"], "relative_need": rel,
                "relative_need_rank": REL_NEED.index(rel),
                "cost_level": "moral", "ack_variant": ai, "closing_variant": ci,
                "pair_index": ai * len(CLOSINGS) + ci,
                "source_model": "templated",
                "shared_prefix": base,
                "pos_text": base + SWITCH_CLAUSE + " " + close,
                "neg_text": base + STAY_CLAUSE + " " + close,
                "generated_at": now,
            }) + "\n")
            n += 1
    print(f"wrote {n} moral-vs-moral pairs ({len(SCENARIOS)} families x {len(REL_NEED)} levels) -> {OUT}")


if __name__ == "__main__":
    main()
