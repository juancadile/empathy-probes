"""Compare component ablations on helping decisions and task controls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def classify(help_delta, task_delta, representation_delta):
    help_loss = -help_delta
    representation_loss = -representation_delta
    if help_loss >= 0.15 and task_delta >= 0.04:
        return "policy_tradeoff"
    if help_loss >= 0.15 and abs(task_delta) <= 0.10:
        return "helping_selective"
    if help_loss >= 0.15 and task_delta <= -0.15:
        return "generic_decision"
    if representation_loss >= 0.75 and abs(help_delta) < 0.15:
        return "representation_writer"
    return "mixed_or_small"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helping", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    helping = json.loads(Path(args.helping).read_text())
    task = json.loads(Path(args.task).read_text())
    task_by_component = {row["component"]: row for row in task["results"]}

    rows = []
    for help_row in helping["results"]:
        task_row = task_by_component[help_row["component"]]
        help_loss = -help_row["delta_behavior"]
        task_delta = task_row["delta_behavior"]
        task_loss = -task_delta
        representation_loss = -help_row["delta_separation"]
        rows.append({
            "component": help_row["component"],
            "source": help_row["source"],
            "helping_delta": help_row["delta_behavior"],
            "helping_loss": help_loss,
            "task_delta": task_delta,
            "task_loss": task_loss,
            "representation_delta": help_row["delta_separation"],
            "representation_loss": representation_loss,
            "task_projection_delta": task_row["delta_separation"],
            "policy_tradeoff_score": help_loss + max(task_delta, 0),
            "helping_specificity_score": help_loss - max(task_loss, 0),
            "role": classify(
                help_row["delta_behavior"],
                task_delta,
                help_row["delta_separation"],
            ),
        })

    random_rows = [row for row in rows if row["source"] == "random"]
    controls = {
        key: {
            "mean": float(np.mean([row[key] for row in random_rows])),
            "sd": float(np.std([row[key] for row in random_rows], ddof=1)),
        }
        for key in ("helping_delta", "task_delta", "representation_delta")
    }
    candidates = [row for row in rows if row["source"] == "dfa"]
    roles = {}
    for role in ("policy_tradeoff", "helping_selective", "generic_decision", "representation_writer"):
        roles[role] = sorted(
            (row for row in candidates if row["role"] == role),
            key=lambda row: row["policy_tradeoff_score"]
            if role == "policy_tradeoff"
            else row["helping_specificity_score"]
            if role in {"helping_selective", "generic_decision"}
            else row["representation_loss"],
            reverse=True,
        )

    summary = {
        "helping_baseline": helping["baseline_behavior"],
        "task_baseline": task["baseline_behavior"],
        "n_components": len(rows),
        "random_controls": controls,
        "roles": roles,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["policy_tradeoff_score"], reverse=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
