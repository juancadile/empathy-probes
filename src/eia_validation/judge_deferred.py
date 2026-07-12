"""E16 secondary: offline judging of deferred EIA rubric queries.

Replays the judge calls the local driver deferred (deferred_judgements.jsonl),
exactly as the harness would have made them (openai gpt-4.1, JSON mode, seed
passed through; score coerced to int in {0,1,2} else 0 — main.py:408-421).

Row order in each jsonl follows the driver's run loop (scenario-major, then
seed), one judge call per game; the row's own "seed" field corroborates.

Usage: OPENAI_API_KEY=... python src/eia_validation/judge_deferred.py
"""

import json
import os
import time
from pathlib import Path

import requests

CONDITIONS = ["baseline", "positive_writers", "suppressors"]
BATCHES = [
    # (root, scenario order, seeds) matching the driver invocation
    (Path("results/eia_local"),
     ["food_delivery", "the_maze", "the_listener", "the_protector", "the_duel"],
     [42, 1234, 999]),
    (Path("results/eia_local_listener_ext"),
     ["the_listener"],
     [7, 123, 555, 777, 1111, 2024, 4242, 31337, 90210, 808]),
]


def judge(query, system, seed, model="gpt-4.1"):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model, "seed": seed,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": query}]},
        timeout=120,
    )
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    try:
        score = int(str(data.get("score", 0)).strip())
    except Exception:
        score = 0
    if score not in (0, 1, 2):
        score = 0
    return score, str(data.get("reason", ""))[:300]


def main():
    rows = []
    for root, scenarios, seeds in BATCHES:
        keys = [(sc, sd) for sc in scenarios for sd in seeds]
        for cond in CONDITIONS:
            lines = [json.loads(l) for l in open(root / cond / "deferred_judgements.jsonl")]
            assert len(lines) == len(keys), (root, cond, len(lines), len(keys))
            for (scenario, seed), rec in zip(keys, lines):
                assert rec.get("seed") == seed, (root, cond, scenario, seed, rec.get("seed"))
                score, reason = judge(rec["query"], rec["system"], seed)
                rows.append({"root": str(root), "condition": cond, "scenario": scenario,
                             "seed": seed, "score": score, "reason": reason})
                print(f"{cond:18s} {scenario:15s} seed {seed:6d} -> {score}")
                time.sleep(0.3)

    out = Path("results/eia_judge_scores.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nMEAN JUDGE SCORE (0-2)")
    for root, scenarios, seeds in BATCHES:
        print(f"  {root}:")
        for cond in CONDITIONS:
            sel = [r["score"] for r in rows if r["condition"] == cond and r["root"] == str(root)]
            print(f"    {cond:18s} {sum(sel)/len(sel):.3f} (n={len(sel)})")
    print("\n  the_listener pooled (n=13):")
    for cond in CONDITIONS:
        sel = [r["score"] for r in rows if r["condition"] == cond and r["scenario"] == "the_listener"]
        print(f"    {cond:18s} {sum(sel)/len(sel):.3f} (n={len(sel)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
