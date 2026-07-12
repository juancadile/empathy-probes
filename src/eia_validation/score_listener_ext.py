"""E16b: pooled the_listener analysis across original (E16) + extension seeds.

Pools the 3 original seeds (results/eia_local) with the 10 extension seeds
(results/eia_local_listener_ext) for the one scenario with dynamic range in E16.
Outcomes (pre-registered listener rules): count of say messages to the user,
door reached; plus the door-vs-engagement tradeoff surfaced in E16.

Usage: python src/eia_validation/score_listener_ext.py
"""

import json
import re
from pathlib import Path

import numpy as np

from score_eia_runs import score_the_listener, is_parse_fail, bootstrap_ci

CONDITIONS = ["baseline", "positive_writers", "suppressors"]
ORIG_SEEDS = [42, 1234, 999]
EXT_SEEDS = [7, 123, 555, 777, 1111, 2024, 4242, 31337, 90210, 808]
ROOTS = {tuple(ORIG_SEEDS): Path("results/eia_local"),
         tuple(EXT_SEEDS): Path("results/eia_local_listener_ext")}


def main():
    runs, intents, parse = {}, {}, {c: [0, 0] for c in CONDITIONS}
    for seeds, root in ROOTS.items():
        for cond in CONDITIONS:
            summ = json.load(open(root / cond / "run_summary.json"))
            ic = {r["seed"]: (int(m.group(1)) if (m := re.search(r"'initial_choice': (\d+)", r.get("result", ""))) else None)
                  for r in summ["runs"] if r["scenario"] == "the_listener"}
            for seed in seeds:
                p = root / cond / cond / "the_listener" / f"the_listener_seed{seed}" / "experiment.json"
                steps = json.load(open(p))["steps"]
                parse[cond][0] += sum(is_parse_fail(s) for s in steps)
                parse[cond][1] += len(steps)
                runs[(cond, seed)] = score_the_listener(steps)
                intents[(cond, seed)] = ic.get(seed)

    all_seeds = ORIG_SEEDS + EXT_SEEDS
    print(f"POOLED THE_LISTENER, n={len(all_seeds)} seeds x {len(CONDITIONS)} conditions")
    print("\nVALIDITY: parse rates")
    for c in CONDITIONS:
        f, t = parse[c]
        print(f"  {c:18s} {1 - f/t:.4f} ({f}/{t} fails)")

    print("\nPER-SEED n_says (door):")
    print(f"  {'seed':>6s}  " + "  ".join(f"{c[:12]:>14s}" for c in CONDITIONS))
    for s in all_seeds:
        row = f"  {s:6d}  "
        for c in CONDITIONS:
            m = runs[(c, s)]
            row += f"{m['n_says']:>10d} ({'D' if m['door_opened'] else '-'})  "
        print(row)

    report = {"n_seeds": len(all_seeds), "parse": {c: parse[c] for c in CONDITIONS},
              "per_run": {f"{c}/seed{s}": runs[(c, s)] for c in CONDITIONS for s in all_seeds},
              "paired": {}, "door_rates": {}, "intents": {}}

    base_says = np.array([runs[("baseline", s)]["n_says"] for s in all_seeds], float)
    base_door = np.array([runs[("baseline", s)]["door_opened"] for s in all_seeds], float)
    print("\nPAIRED n_says deltas vs baseline:")
    for c in ["positive_writers", "suppressors"]:
        e = np.array([runs[(c, s)]["n_says"] for s in all_seeds], float)
        d = e - base_says
        ci = bootstrap_ci(d)
        nz = d[d != 0]
        pred_sign = -1 if c == "positive_writers" else 1
        agree = int((np.sign(nz) == pred_sign).sum())
        # exact two-sided sign test on nonzero deltas
        from math import comb
        k, n = agree, len(nz)
        p_sign = sum(comb(n, i) for i in range(min(k, n - k) + 1)) * 2 / 2**n if n else None
        report["paired"][c] = {"deltas": d.tolist(), "mean": float(d.mean()), "ci95": ci,
                               "nonzero": int(n), "predicted_sign_agree": agree,
                               "sign_test_p_two_sided": p_sign}
        print(f"  {c}: mean {d.mean():+.2f}, CI95 [{ci[0]:+.2f}, {ci[1]:+.2f}], "
              f"predicted sign {agree}/{n} nonzero, sign-test p={p_sign:.4f}")

    print("\nDOOR-OPENED rates (task completion):")
    for c in CONDITIONS:
        e = np.array([runs[(c, s)]["door_opened"] for s in all_seeds], float)
        report["door_rates"][c] = float(e.mean())
        extra = ""
        if c != "baseline":
            d = e - base_door
            ci = bootstrap_ci(d)
            report["door_rates"][f"{c}_paired_delta"] = {"mean": float(d.mean()), "ci95": ci}
            extra = f" | paired delta {d.mean():+.3f}, CI95 [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        print(f"  {c:18s} {e.mean():.3f}{extra}")

    print("\nINTENTION (initial self-assessment 0-2):")
    for c in CONDITIONS:
        vals = [intents[(c, s)] for s in all_seeds if intents[(c, s)] is not None]
        report["intents"][c] = {"mean": float(np.mean(vals)), "n": len(vals)}
        print(f"  {c:18s} mean {np.mean(vals):.2f} (n={len(vals)}/{len(all_seeds)})")

    out = Path("results/eia_local_listener_ext/e16b_pooled_scores.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
