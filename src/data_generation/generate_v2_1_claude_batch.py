"""Generate the Claude share of the V2.1 stimulus suite via the Message Batches API.

50% cheaper than synchronous calls (https://platform.claude.com/docs/en/build-with-claude/batch-processing);
most batches finish within an hour. Covers cells A B D E F G H S with
claude-haiku-4-5; cell R still runs via generate_v2_1_suite.py --cells R.

Usage:
  python src/data_generation/generate_v2_1_claude_batch.py submit    # builds missing prompts, submits batch
  python src/data_generation/generate_v2_1_claude_batch.py status
  python src/data_generation/generate_v2_1_claude_batch.py collect   # waits for completion, writes jsonl

Output matches generate_v2_1_suite.py exactly: data/contrastive_pairs/v2_1/{cell}_claude-haiku.jsonl
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_v2_1_suite import (  # noqa: E402
    MODELS, MAX_TOKENS, TEMPERATURE, OUT_DIR,
    existing_keys, load_cells, load_env, render,
)

MODEL = MODELS["claude-haiku"]
MODEL_KEY = "claude-haiku"
BATCH_ID_FILE = OUT_DIR / "claude_batch_id.txt"
SEP = "--"  # custom_id = cell--scenario_id--index--side (all components are [a-z0-9_])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("claude-batch")


def build_requests(cells, fmt, selected):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = []

    def add(custom_id, prompt):
        reqs.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            ),
        ))

    for cid in selected:
        cell = cells[cid]
        done = existing_keys(OUT_DIR / f"{cid}_{MODEL_KEY}.jsonl")
        if cid == "S":
            for scen in cell["scenarios"]:
                for i in range(cell["n_triples_per_model_per_scenario"]):
                    for level, pressure in scen["levels"].items():
                        if (scen["id"], i, level) in done:
                            continue
                        prompt = render(cell["template"], {**scen, "pressure": pressure}, fmt)
                        add(SEP.join(["S", scen["id"], str(i), level]), prompt)
        else:
            pos_t, neg_t = cell["pos_templates"], cell["neg_templates"]
            for scen in cell["scenarios"]:
                for i in range(cell["n_pairs_per_model_per_scenario"]):
                    if (scen["id"], i, "") in done:
                        continue
                    variant = i % min(len(pos_t), len(neg_t))
                    add(SEP.join([cid, scen["id"], str(i), "pos"]), render(pos_t[variant], scen, fmt))
                    add(SEP.join([cid, scen["id"], str(i), "neg"]), render(neg_t[variant], scen, fmt))
    return reqs


def cmd_submit(client, cells, fmt, selected):
    reqs = build_requests(cells, fmt, selected)
    if not reqs:
        log.info("nothing missing — all claude records already exist")
        return
    log.info("submitting batch: %d requests (~$%.2f at 50%% batch discount)", len(reqs), len(reqs) * 0.00075)
    batch = client.messages.batches.create(requests=reqs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_ID_FILE.write_text(batch.id)
    log.info("batch %s submitted (%s) — id saved to %s", batch.id, batch.processing_status, BATCH_ID_FILE)


def cmd_status(client):
    batch = client.messages.batches.retrieve(BATCH_ID_FILE.read_text().strip())
    log.info("batch %s: %s | counts: %s", batch.id, batch.processing_status, batch.request_counts)


def cmd_collect(client, cells, fmt):
    batch_id = BATCH_ID_FILE.read_text().strip()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        log.info("status %s (processing: %d) — waiting 60s",
                 batch.processing_status, batch.request_counts.processing)
        time.sleep(60)
    log.info("ended: %d succeeded, %d errored", batch.request_counts.succeeded, batch.request_counts.errored)

    # results arrive in any order — key by custom_id
    texts = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            texts[result.custom_id] = next((b.text for b in msg.content if b.type == "text"), "").strip()
        else:
            log.warning("%s: %s", result.custom_id, result.result.type)

    now = datetime.now(timezone.utc).isoformat()
    written = 0
    # severity records: one line per level
    s_done = existing_keys(OUT_DIR / f"S_{MODEL_KEY}.jsonl")
    with open(OUT_DIR / f"S_{MODEL_KEY}.jsonl", "a") as f:
        for cid_key, text in texts.items():
            cell, scen_id, idx, side = cid_key.split(SEP)
            if cell != "S" or not text or (scen_id, int(idx), side) in s_done:
                continue
            f.write(json.dumps({
                "cell": "S", "cell_name": cells["S"]["name"], "scenario_id": scen_id,
                "triple_index": int(idx), "level": side, "source_model": MODEL,
                "text": text, "generated_at": now,
            }) + "\n")
            written += 1

    # pair records: join pos+neg
    pair_cells = sorted({k.split(SEP)[0] for k in texts} - {"S"})
    for cid in pair_cells:
        done = existing_keys(OUT_DIR / f"{cid}_{MODEL_KEY}.jsonl")
        with open(OUT_DIR / f"{cid}_{MODEL_KEY}.jsonl", "a") as f:
            keys = {tuple(k.split(SEP)[:3]) for k in texts if k.startswith(cid + SEP)}
            for _, scen_id, idx in sorted(keys):
                if (scen_id, int(idx), "") in done:
                    continue
                pos = texts.get(SEP.join([cid, scen_id, idx, "pos"]))
                neg = texts.get(SEP.join([cid, scen_id, idx, "neg"]))
                if not (pos and neg):
                    continue
                n_templates = min(len(cells[cid]["pos_templates"]), len(cells[cid]["neg_templates"]))
                f.write(json.dumps({
                    "cell": cid, "cell_name": cells[cid]["name"], "scenario_id": scen_id,
                    "pair_index": int(idx), "template_variant": int(idx) % n_templates,
                    "source_model": MODEL, "pos_text": pos, "neg_text": neg, "generated_at": now,
                }) + "\n")
                written += 1
    log.info("wrote %d records into %s", written, OUT_DIR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["submit", "status", "collect"])
    p.add_argument("--cells", nargs="+", default=["A", "B", "D", "E", "F", "G", "H", "S"])
    args = p.parse_args()

    load_env()
    import anthropic
    client = anthropic.Anthropic()
    cells, fmt = load_cells()
    selected = [c for c in args.cells if c in cells and c != "R"]

    if args.command == "submit":
        cmd_submit(client, cells, fmt, selected)
    elif args.command == "status":
        cmd_status(client)
    else:
        cmd_collect(client, cells, fmt)


if __name__ == "__main__":
    main()
