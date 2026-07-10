"""Audit V2.1 contrastive labels with a blinded Claude Message Batch.

The generation prompt does not guarantee that the completion enacted the
requested contrast. This audit scores what each completion actually does and
keeps lexical warmth separate from helping/task behavior.

Usage:
  python src/evaluation/audit_v2_1_labels_claude_batch.py submit
  python src/evaluation/audit_v2_1_labels_claude_batch.py status
  python src/evaluation/audit_v2_1_labels_claude_batch.py collect
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "contrastive_pairs" / "v2_1"
OUT_DIR = ROOT / "results" / "v2_1_label_audit"
BATCH_ID_FILE = OUT_DIR / "batch_id.txt"
MANIFEST_FILE = OUT_DIR / "manifest.jsonl"
RESULT_FILE = OUT_DIR / "judgments.jsonl"
MODEL = "claude-haiku-4-5-20251001"
CELLS = ("A", "B", "D", "E", "F", "G", "H")

CONTRASTS = {
    "A": "Does the agent help the person, even where that delays or sacrifices its task?",
    "B": "Does the agent help or support the person when doing so has no task cost?",
    "D": "Does the response engage warmly and supportively rather than remain detached?",
    "E": "Does the agent stay on its assigned task rather than follow the non-social distraction?",
    "F": "Does the described third-party actor help rather than continue its objective?",
    "G": "Is the stated motive genuine concern rather than strategic performance, with actions held constant?",
    "H": "Does the stated character genuinely care despite taking the same protocol-bound action?",
}

SYSTEM = """You audit behavioral labels for a mechanistic-interpretability dataset.
Judge only the supplied completions, not what their hidden prompts may have requested.
Separate enacted behavior from empathic vocabulary and from claimed character.
Return one JSON object and no markdown."""


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def records():
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        if path.name[0] not in CELLS or path.name[1] != "_":
            continue
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                record = json.loads(line)
                if "pos_text" in record and "neg_text" in record:
                    yield path, line_no, record


def audit_prompt(cell, text_x, text_y):
    return f"""TARGET CONTRAST
{CONTRASTS[cell]}

COMPLETION X
{text_x}

COMPLETION Y
{text_y}

Score each completion from 0 to 4 on:
- helping_action: concrete action that benefits the person at issue
- task_persistence: continuing the original objective despite distractions or social pressure
- affective_language: explicit warmth, feelings, care, distress, compassion, or similar vocabulary
- caring_character: the speaker/actor claims genuine concern rather than instrumental performance

Then decide whether X or Y better realizes the TARGET CONTRAST's first side. Use "tie" when
the completions do not behaviorally differ enough to support a contrastive label. Mark usable
false for ties, refusals, prompt leakage without enacted contrast, or contradictions between
reasoning and the final action.

Return exactly this JSON shape:
{{"x":{{"helping_action":0,"task_persistence":0,"affective_language":0,"caring_character":0}},
"y":{{"helping_action":0,"task_persistence":0,"affective_language":0,"caring_character":0}},
"first_side":"X|Y|tie","usable":true,"reason":"one short sentence"}}"""


def batch_request_types():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    return MessageCreateParamsNonStreaming, Request


def submit(client):
    MessageParams, Request = batch_request_types()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    requests = []
    manifest = []
    for index, (path, line_no, record) in enumerate(records()):
        flipped = rng.random() < 0.5
        text_x = record["neg_text"] if flipped else record["pos_text"]
        text_y = record["pos_text"] if flipped else record["neg_text"]
        custom_id = f"audit-{index:05d}"
        requests.append(Request(
            custom_id=custom_id,
            params=MessageParams(
                model=MODEL,
                max_tokens=500,
                temperature=0,
                system=SYSTEM,
                messages=[{"role": "user", "content": audit_prompt(record["cell"], text_x, text_y)}],
            ),
        ))
        manifest.append({
            "custom_id": custom_id,
            "file": str(path.relative_to(ROOT)),
            "line": line_no,
            "cell": record["cell"],
            "scenario_id": record["scenario_id"],
            "pair_index": record.get("pair_index"),
            "source_model": record["source_model"],
            "pos_is": "Y" if flipped else "X",
        })

    if not requests:
        raise SystemExit("no pair records found")
    MANIFEST_FILE.write_text("".join(json.dumps(row) + "\n" for row in manifest))
    batch = client.messages.batches.create(requests=requests)
    BATCH_ID_FILE.write_text(batch.id + "\n")
    print(f"submitted {len(requests)} audits as {batch.id}")


def status(client):
    batch = client.messages.batches.retrieve(BATCH_ID_FILE.read_text().strip())
    print(batch.id, batch.processing_status, batch.request_counts)


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    return json.loads(match.group(0))


def collect(client):
    batch_id = BATCH_ID_FILE.read_text().strip()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise SystemExit(f"batch is {batch.processing_status}; collect after it ends")
    manifest = {
        row["custom_id"]: row
        for row in map(json.loads, MANIFEST_FILE.read_text().splitlines())
    }
    rows = []
    errors = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            errors += 1
            continue
        message = result.result.message
        text = next((block.text for block in message.content if block.type == "text"), "")
        try:
            judgment = parse_json(text)
        except (ValueError, json.JSONDecodeError):
            errors += 1
            continue
        meta = manifest[result.custom_id]
        pos_key = meta["pos_is"].lower()
        neg_key = "y" if pos_key == "x" else "x"
        expected = meta["pos_is"]
        observed = judgment.get("first_side", "tie").upper()
        rows.append({
            **meta,
            "label_correct": observed == expected,
            "observed_first_side": observed,
            "usable": bool(judgment.get("usable")) and observed in {"X", "Y"},
            "pos_scores": judgment.get(pos_key),
            "neg_scores": judgment.get(neg_key),
            "reason": judgment.get("reason", ""),
        })
    RESULT_FILE.write_text("".join(json.dumps(row) + "\n" for row in rows))
    usable = [row for row in rows if row["usable"]]
    correct = [row for row in usable if row["label_correct"]]
    print(json.dumps({
        "collected": len(rows),
        "parse_or_api_errors": errors,
        "usable": len(usable),
        "usable_rate": len(usable) / len(rows) if rows else 0,
        "label_accuracy_among_usable": len(correct) / len(usable) if usable else 0,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "status", "collect"))
    args = parser.parse_args()
    load_env()
    import anthropic

    client = anthropic.Anthropic()
    {"submit": submit, "status": status, "collect": collect}[args.command](client)


if __name__ == "__main__":
    main()
