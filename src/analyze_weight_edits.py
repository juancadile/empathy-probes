"""Paired bootstrap analysis for targeted weight-orthogonalization results."""

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values, rng, n_bootstrap):
    values = np.asarray(values, dtype=np.float64)
    means = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def paired_deltas(condition, baseline):
    return {
        "helping": (
            np.asarray(condition["helping_choice"]["per_pair"])
            - np.asarray(baseline["helping_choice"]["per_pair"])
        ),
        "task": (
            np.asarray(condition["task_choice"]["per_pair"])
            - np.asarray(baseline["task_choice"]["per_pair"])
        ),
        "decision_projection": (
            np.asarray(condition["decision_projection"]["per_pair_difference"])
            - np.asarray(baseline["decision_projection"]["per_pair_difference"])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text())
    baseline = payload["baseline"]
    conditions = {
        "positive_writers_k2": payload["positive_writers"][-1]["metrics"],
        "suppressors_k4": payload["suppressors"][-1]["metrics"],
        "random_k2": payload["random"][1]["metrics"],
        "random_k6": payload["random"][-1]["metrics"],
    }
    rng = np.random.default_rng(args.seed)
    deltas = {name: paired_deltas(condition, baseline) for name, condition in conditions.items()}
    summary = {
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "conditions_vs_baseline": {
            name: {
                metric: bootstrap(values, rng, args.n_bootstrap)
                for metric, values in metrics.items()
            }
            for name, metrics in deltas.items()
        },
        "positive_writers_vs_random_k2": {
            metric: bootstrap(
                deltas["positive_writers_k2"][metric] - deltas["random_k2"][metric],
                rng,
                args.n_bootstrap,
            )
            for metric in deltas["positive_writers_k2"]
        },
        "positive_writers_vs_random_k6": {
            metric: bootstrap(
                deltas["positive_writers_k2"][metric] - deltas["random_k6"][metric],
                rng,
                args.n_bootstrap,
            )
            for metric in deltas["positive_writers_k2"]
        },
        "neutral_drift": {
            name: condition["neutral_drift"] for name, condition in conditions.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
