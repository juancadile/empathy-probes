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
  4. per-family MEAN edit effect at each need level (resolved / mild /
     urgent) — a LEVEL estimand, kept strictly distinct from the cost-slope
     estimands above — plus family-paired urgent-minus-resolved and
     mild-minus-resolved mean-effect contrasts (bootstrap CIs, sign counts)

Claim analyses (distinct per set):
  - WRITER "need-gated level effect" is an EXPLORATORY, post-audit sensitivity
    analysis because the practical-equivalence bound was introduced after the
    resolved-effect estimate had been inspected. The pattern requires the family-paired
    urgent-minus-resolved MEAN-EFFECT contrast 95% CI to exclude zero AND the
    resolved-need mean effect to be practically null (95% CI within the
    post-audit +/-MEAN_EQUIV_BOUND equivalence band). It must NOT be
    inferred from cost slopes.
  - SUPPRESSOR "conjunctive need x cost" remains the family-paired
    urgent-minus-resolved COST-SLOPE contrast (CI excluding zero).

Usage: python src/analysis/e18c_slope_interaction.py
Writes results/e18c_slope_interaction_gemma2_9b_it/e18c.json
"""

import hashlib
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
NEED_CELLS = [("resolved", "need_resolved"), ("mild", "need_mild"),
              ("urgent", "cost_axis")]
# Post-audit sensitivity band for "resolved mean effect is practically null"
# (choice-score units; ~1/5 of the writers' confirm-battery level effect
# |-0.28|). This threshold was introduced after the resolved-effect estimate
# had been inspected, so it is not a preregistered/confirmatory equivalence test.
MEAN_EQUIV_BOUND = 0.05


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


def family_mean_effects(delta, fams):
    """Mean edit effect within each family (LEVEL estimand, not a slope)."""
    return {f: float(delta[fams == f].mean()) for f in sorted(set(fams))}


def paired_contrast(per_family_a, per_family_b):
    """Family-paired a-minus-b contrast with bootstrap CI + zero-exclusion."""
    paired = {f: per_family_a[f] - per_family_b[f]
              for f in per_family_a if f in per_family_b}
    stat = boot_family_stat(paired)
    stat["n_paired_families"] = len(paired)
    stat["ci95_excludes_zero"] = bool(stat["ci95"][0] > 0 or stat["ci95"][1] < 0)
    return stat


def writer_need_gate(urgent_minus_resolved_mean_stat, resolved_mean_stat,
                     bound=MEAN_EQUIV_BOUND):
    """WRITER need-gated LEVEL-effect gate: the paired urgent-resolved
    MEAN-effect CI excludes zero AND the resolved mean effect is equivalent to
    zero (95% CI inside +/-bound). Cost slopes do not bear on this gate."""
    contrast_ok = urgent_minus_resolved_mean_stat["ci95_excludes_zero"]
    lo, hi = resolved_mean_stat["ci95"]
    resolved_ok = bool(lo >= -bound and hi <= bound)
    pattern = bool(contrast_ok and resolved_ok)
    return {"equivalence_bound": bound,
            "status": "exploratory_post_audit_sensitivity",
            "confirmatory_claim_supported": False,
            "urgent_minus_resolved_mean_ci_excludes_zero": bool(contrast_ok),
            "resolved_mean_ci_within_equivalence_bound": resolved_ok,
            "sensitivity_pattern_satisfied": pattern}


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
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    out = {"source_run": {"path": args.run, "sha256": sha(args.run)},
           "conditions_spec": run.get("conditions_spec"),
           "run_provenance": run.get("provenance"),
           "cell_files": {n: {"path": p, "sha256": sha(ROOT / p)}
                          for n, p in CELL_JSONL.items()},
           "gates": {
               "suppressor_conjunctive_need_x_cost":
                   "requires the family-paired slope_urgent_minus_resolved_paired "
                   "(COST-SLOPE contrast) 95% CI to EXCLUDE zero; an ordered set "
                   "of point estimates does not qualify",
               "writer_need_gated_level_effect":
                   "requires the family-paired mean_effect_urgent_minus_resolved_"
                   "paired (MEAN-EFFECT contrast) 95% CI to EXCLUDE zero AND the "
                   "resolved mean effect's 95% CI to lie within the post-audit "
                   f"sensitivity band +/-{MEAN_EQUIV_BOUND}; this is exploratory, "
                   "not preregistered or confirmatory, and must NOT be inferred "
                   "from cost slopes",
           },
           "conditions": {}}

    for cond in CONDS:
        e = run["conditions"][cond]
        entry = {}
        # 1-2: slope interaction welfare vs nonsocial
        slopes = {}
        for cell in ["cost_axis", "nonsocial_axis"]:
            d = np.asarray(e[cell]["per_pair_delta"])
            fams, ranks = meta[cell]
            assert len(d) == len(fams), f"{cond}/{cell}: {len(d)} deltas vs {len(fams)} pairs"
            slopes[cell] = family_cost_slopes(d, fams, ranks)
        diff = {f: slopes["cost_axis"][f] - slopes["nonsocial_axis"][f]
                for f in slopes["cost_axis"]}
        entry["slope_welfare"] = boot_family_stat(slopes["cost_axis"])
        entry["slope_nonsocial"] = boot_family_stat(slopes["nonsocial_axis"])
        entry["slope_interaction_welfare_minus_nonsocial"] = boot_family_stat(diff)
        # 3: cost slope within each need level, plus 4: mean LEVEL effect per
        # need level — two distinct estimands, never mixed
        entry["cost_slope_by_need"] = {}
        entry["mean_effect_by_need"] = {}
        need_slopes, need_means = {}, {}
        for label, cell in NEED_CELLS:
            d = np.asarray(e[cell]["per_pair_delta"])
            fams, ranks = meta[cell]
            assert len(d) == len(fams), f"{cond}/{cell}: {len(d)} deltas vs {len(fams)} pairs"
            need_slopes[label] = family_cost_slopes(d, fams, ranks)
            entry["cost_slope_by_need"][label] = boot_family_stat(need_slopes[label])
            need_means[label] = family_mean_effects(d, fams)
            entry["mean_effect_by_need"][label] = boot_family_stat(need_means[label])
        # pre-registered three-way contrast (codex, rescue3c design):
        # family-paired slope_urgent - slope_resolved, NOT ordering of point
        # estimates — the SUPPRESSOR conjunctive need x cost gate
        entry["slope_urgent_minus_resolved_paired"] = paired_contrast(
            need_slopes["urgent"], need_slopes["resolved"])
        # family-paired MEAN-effect contrasts — the LEVEL estimand behind the
        # WRITER need-gated claim (kept distinct from the slope contrasts)
        entry["mean_effect_urgent_minus_resolved_paired"] = paired_contrast(
            need_means["urgent"], need_means["resolved"])
        entry["mean_effect_mild_minus_resolved_paired"] = paired_contrast(
            need_means["mild"], need_means["resolved"])
        entry["writer_need_gated_level_effect"] = writer_need_gate(
            entry["mean_effect_urgent_minus_resolved_paired"],
            entry["mean_effect_by_need"]["resolved"])
        out["conditions"][cond] = entry
        si = entry["slope_interaction_welfare_minus_nonsocial"]
        pr = entry["slope_urgent_minus_resolved_paired"]
        mr = entry["mean_effect_urgent_minus_resolved_paired"]
        wg = entry["writer_need_gated_level_effect"]
        print(f"{cond}: slope-interaction {si['mean']:+.4f} CI {si['ci95']} "
              f"({si['sign_positive_families']}/{si['n_families']} fams +) | "
              f"cost-slope by need: " + " ".join(
                  f"{k}={v['mean']:+.3f}" for k, v in entry["cost_slope_by_need"].items())
              + f" | paired urgent-resolved SLOPE {pr['mean']:+.4f} CI {pr['ci95']} "
              f"excludes-zero={pr['ci95_excludes_zero']}")
        print(f"{cond}: mean effect by need: " + " ".join(
                  f"{k}={v['mean']:+.3f} CI {v['ci95']}"
                  for k, v in entry["mean_effect_by_need"].items())
              + f" | paired urgent-resolved MEAN {mr['mean']:+.4f} CI {mr['ci95']} "
              f"({mr['sign_positive_families']}/{mr['n_paired_families']} fams +) "
              f"excludes-zero={mr['ci95_excludes_zero']} | "
              f"writer exploratory sensitivity pattern="
              f"{wg['sensitivity_pattern_satisfied']} "
              f"(resolved-null={wg['resolved_mean_ci_within_equivalence_bound']}, "
              f"bound ±{MEAN_EQUIV_BOUND})")

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "e18c.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}/e18c.json")


if __name__ == "__main__":
    main()
