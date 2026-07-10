"""Materialize behaviorally valid V2.1 pairs from the explicit-label audit."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = (
    ROOT / "results" / "v2_1_label_audit" / "v2_explicit_labels" / "judgments.jsonl"
)
DEFAULT_OUT = ROOT / "data" / "contrastive_pairs" / "v2_1_audited"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    accepted = [
        json.loads(line)
        for line in args.audit.read_text().splitlines()
        if line.strip()
    ]
    accepted = [row for row in accepted if row["usable"] and row["label_correct"]]

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.jsonl"):
        old.unlink()

    source_cache = {}
    counts = Counter()
    handles = {}
    try:
        for audit in accepted:
            source = ROOT / audit["file"]
            if source not in source_cache:
                source_cache[source] = source.read_text().splitlines()
            record = json.loads(source_cache[source][audit["line"] - 1])
            record["audit"] = {
                "model": "claude-haiku-4-5-20251001",
                "usable": True,
                "label_correct": True,
                "pos_scores": audit["pos_scores"],
                "neg_scores": audit["neg_scores"],
                "reason": audit["reason"],
            }
            cell = record["cell"]
            if cell not in handles:
                handles[cell] = (args.out / f"{cell}.jsonl").open("w")
            handles[cell].write(json.dumps(record) + "\n")
            counts[cell] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {"total": sum(counts.values()), "by_cell": dict(sorted(counts.items()))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
