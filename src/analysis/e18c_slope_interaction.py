"""E18c: correct slope-interaction estimand + need x cost surface (stress-test fix 1).

The quantity previously reported as the 'welfare x cost interaction' was
mean(d_welfare) - mean(d_nonsocial) (axis-mean difference). The pre-registered
E18 hypothesis is about SLOPES: does the edit effect's cost slope differ
between the welfare and non-welfare axes? This script computes, from the
persisted per-pair arrays of the E21 run (which includes both E18 axes):

  1. per-family cost slopes of the edit effect on each axis
  2. slope difference (welfare - nonsocial): family bootstrap CI, LOFO, family
     sign counts
  3. the need x cost surface: the edit effect's cost slope within each need
     level (resolved / mild / urgent), exposing the three-way structure

Usage: python src/analysis/e18c_slope_interaction.py
Writes results/e18c_slope_interaction_gemma2_9b_it/e18c.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/e21_need_gemma2_9b_it/e18.json"
CELL_JSONL = {
    "cost_axis": "data/contrastive_pairs/v2_2/cost_axis_templated.jsonl",
    "nonsocial_axis": "data/contrastive_pairs/v2_2/nonsocial_axis_templated.jsonl",
    "need_mild": "data/contrastive_pairs/v2_2/need_mild_axis_templated.jsonl",
    "need_resolved": "data/contrastive_pairs/v2_2/need_resolved_axis_templated.jsonl",
}
COST_RANK = {"free": 0, "low": 1, "medium": 2, "high": 3}
CONDS = ["positive_writers_k2", "suppressors_k4", "targeted_k6", "random_k6"]


def cell_meta(name):
    rows = [json.loads(l) for l in open(ROOT / CELL_JSONL[name])]
    return (np.array([r["scenario_id"] for r in rows]),
            np.array([COST_RANK[r["cost_level"]] for r in rows]))


def family_cost_slopes(delta, fams, ranks):
    """OLS slope of delta on cost rank within each family."""
    out = {}
    for f in sorted(set(fams)):
        m = fams == f
        out[f] = float(np.polyfit(ranks[m], delta[m], 1)[0])
    return out


def boot_family_stat(per_family, n=5000, seed=42):
    F = sorted(per_family)
    vals = np.array([per_family[f] for f in F])
    rng = np.random.default_rng(seed)
    bs = [vals[rng.integers(0, len(F), len(F))].mean() for _ in range(n)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "lofo": [float(np.delete(vals, i).mean()) for i in range(len(F))],
            "sign_positive_families": int((vals > 0).sum()), "n_families": len(F)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN),
                    help="e18.json with per-pair deltas (any run with these cells)")
    ap.add_argument("--out", default=str(ROOT / "results/e18c_slope_interaction_gemma2_9b_it"))
    args = ap.parse_args()
    run = json.load(open(args.run))
    meta = {n: cell_meta(n) for n in CELL_JSONL}
    out = {"source_run": args.run, "conditions": {}}

    for cond in CONDS:
        e = run["conditions"][cond]
        entry = {}
        # 1-2: slope interaction welfare vs nonsocial
        slopes = {}
        for cell in ["cost_axis", "nonsocial_axis"]:
            d = np.asarray(e[cell]["per_pair_delta"])
            fams, ranks = meta[cell]
            slopes[cell] = family_cost_slopes(d, fams, ranks)
        diff = {f: slopes["cost_axis"][f] - slopes["nonsocial_axis"][f]
                for f in slopes["cost_axis"]}
        entry["slope_welfare"] = boot_family_stat(slopes["cost_axis"])
        entry["slope_nonsocial"] = boot_family_stat(slopes["nonsocial_axis"])
        entry["slope_interaction_welfare_minus_nonsocial"] = boot_family_stat(diff)
        # 3: cost slope within each need level
        entry["cost_slope_by_need"] = {}
        need_slopes = {}
        for label, cell in [("resolved", "need_resolved"), ("mild", "need_mild"),
                            ("urgent", "cost_axis")]:
            d = np.asarray(e[cell]["per_pair_delta"])
            fams, ranks = meta[cell]
            need_slopes[label] = family_cost_slopes(d, fams, ranks)
            entry["cost_slope_by_need"][label] = boot_family_stat(need_slopes[label])
        # pre-registered three-way contrast (codex, rescue3c design):
        # family-paired slope_urgent - slope_resolved, NOT ordering of point estimates
        paired = {f: need_slopes["urgent"][f] - need_slopes["resolved"][f]
                  for f in need_slopes["urgent"] if f in need_slopes["resolved"]}
        entry["slope_urgent_minus_resolved_paired"] = boot_family_stat(paired)
        out["conditions"][cond] = entry
        si = entry["slope_interaction_welfare_minus_nonsocial"]
        print(f"{cond}: slope-interaction {si['mean']:+.4f} CI {si['ci95']} "
              f"({si['sign_positive_families']}/{si['n_families']} fams +) | "
              f"cost-slope by need: " + " ".join(
                  f"{k}={v['mean']:+.3f}" for k, v in entry["cost_slope_by_need"].items()))

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "e18c.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}/e18c.json")


if __name__ == "__main__":
    main()
