"""E22 analysis: moral-vs-moral boundary for re-derived sets (rescue3c item d).

Consumes an e18_interaction.py run whose --cells included
moral=data/contrastive_pairs/v2_2/moral_axis_templated.jsonl and computes, per
condition, the pre-registered E22 contrasts on switch-to-P2 deltas:
  - mean switching boost (family bootstrap 95% CI)
  - relative-need slope (delta vs relative_need_rank, per-family OLS)
  - TOST equivalence for slope flatness: incumbent-stabilization requires the
    90% family-bootstrap CI of the slope inside +/-0.05/level (bound fixed in
    the E22 pre-registration); "CI includes zero" does NOT establish flatness.

Usage: python src/analysis/e22_analysis.py \
  --run results/e22_moral_resid_gemma/e18.json --out results/e22_moral_resid_gemma
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MORAL = ROOT / "data/contrastive_pairs/v2_2/moral_axis_templated.jsonl"
EQUIV_BOUND = 0.05  # per relative-need level, fixed in the E22 pre-registration


def family_slopes(delta, fams, ranks):
    return {f: float(np.polyfit(ranks[fams == f], delta[fams == f], 1)[0])
            for f in sorted(set(fams))}


def boot(vals_by_family, n=10000, seed=42):
    v = np.array(list(vals_by_family.values()))
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
            "n_families": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cell", default="moral")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(MORAL)]
    fams = np.array([r["scenario_id"] for r in rows])
    ranks = np.array([r["relative_need_rank"] for r in rows], dtype=float)

    run = json.load(open(args.run))
    report = {"source_run": args.run, "equivalence_bound_per_level": EQUIV_BOUND,
              "conditions": {}}
    for cond, e in run["conditions"].items():
        d = np.asarray(e[args.cell]["per_pair_delta"])
        assert len(d) == len(rows), f"{cond}: {len(d)} deltas vs {len(rows)} pairs"
        by_fam_mean = {f: float(d[fams == f].mean()) for f in sorted(set(fams))}
        slopes = family_slopes(d, fams, ranks)
        boost = boot(by_fam_mean)
        slope = boot(slopes)
        lo90, hi90 = slope["ci90"]
        entry = {
            "mean_switching_boost": boost,
            "relative_need_slope": slope,
            "slope_TOST_equivalent_flat": bool(lo90 > -EQUIV_BOUND and hi90 < EQUIV_BOUND),
            "slope_ci_excludes_zero": bool(slope["ci95"][0] > 0 or slope["ci95"][1] < 0),
        }
        report["conditions"][cond] = entry
        verdict = ("incumbent-stabilization (boost, flat by TOST)"
                   if entry["slope_TOST_equivalent_flat"] else
                   "welfare-triage (slope resolved nonzero)"
                   if entry["slope_ci_excludes_zero"] else
                   "UNRESOLVED (neither flat by TOST nor slope nonzero)")
        print(f"{cond}: boost {boost['mean']:+.4f} CI {boost['ci95']} | "
              f"slope {slope['mean']:+.4f} 90%CI [{lo90:+.4f},{hi90:+.4f}] -> {verdict}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "e22_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}/e22_analysis.json")


if __name__ == "__main__":
    main()
