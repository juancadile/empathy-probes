"""Summarize component-ablation results against DFA rankings and controls.

Produces a compact JSON report and CSV table suitable for issue updates and
paper figures. A component is strongest evidence when it both writes the
readout direction under DFA and causally reduces the controlled Cell M signal
when mean-ablated.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def component_name(record):
    suffix = "MLP" if record["head"] is None else f"H{record['head']}"
    return f"L{record['layer']}{suffix}"


def load_dfa_scores(paths):
    scores = {}
    ranks = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for rank, record in enumerate(payload["top_components"], start=1):
            name = component_name(record)
            score = record.get("mean_paired_diff", record.get("mean_difference"))
            if score is not None and abs(score) > abs(scores.get(name, 0.0)):
                scores[name] = float(score)
            ranks[name] = min(rank, ranks.get(name, rank))
    return scores, ranks


def standardized_effect(value, controls):
    if len(controls) < 2:
        return None
    std = float(np.std(controls, ddof=1))
    if std == 0:
        return None
    return float((value - np.mean(controls)) / std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--dfa", nargs="+", required=True)
    parser.add_argument("--out", default="results/patching_gemma2_9b_it/analysis")
    args = parser.parse_args()

    payload = json.loads(Path(args.ablation).read_text())
    dfa_scores, dfa_ranks = load_dfa_scores(args.dfa)
    results = payload["results"]
    random_sep = [r["delta_separation"] for r in results if r["source"] == "random"]
    random_beh = [r["delta_behavior"] for r in results if r["source"] == "random"]

    rows = []
    for record in results:
        name = record["component"]
        # Negative deltas mean ablation removed signal or helping preference.
        necessity_sep = -float(record["delta_separation"])
        necessity_beh = -float(record["delta_behavior"])
        row = {
            **record,
            "dfa_score": dfa_scores.get(name),
            "dfa_rank": dfa_ranks.get(name),
            "necessity_separation": necessity_sep,
            "necessity_behavior": necessity_beh,
            "separation_control_z": standardized_effect(
                record["delta_separation"], random_sep
            ),
            "behavior_control_z": standardized_effect(record["delta_behavior"], random_beh),
        }
        rows.append(row)

    dfa_rows = [r for r in rows if r["source"] == "dfa"]
    paired = [r for r in dfa_rows if r["dfa_score"] is not None]
    writer_necessity_corr = None
    if len(paired) >= 3:
        writer_necessity_corr = float(
            np.corrcoef(
                [abs(r["dfa_score"]) for r in paired],
                [r["necessity_separation"] for r in paired],
            )[0, 1]
        )

    ranked = sorted(
        dfa_rows,
        key=lambda r: (r["necessity_separation"], r["necessity_behavior"]),
        reverse=True,
    )
    report = {
        "model": payload["model"],
        "readout_block": payload["readout_block"],
        "n_pairs": payload["n_pairs"],
        "baseline_separation": payload["baseline_separation"],
        "baseline_behavior": payload["baseline_behavior"],
        "random_control": {
            "n": len(random_sep),
            "mean_delta_separation": float(np.mean(random_sep)),
            "sd_delta_separation": float(np.std(random_sep, ddof=1)),
            "mean_delta_behavior": float(np.mean(random_beh)),
            "sd_delta_behavior": float(np.std(random_beh, ddof=1)),
        },
        "dfa_writer_vs_separation_necessity_correlation": writer_necessity_corr,
        "top_causal_components": ranked[:15],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fieldnames = list(rows[0])
    with (out / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
