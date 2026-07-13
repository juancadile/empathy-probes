"""Validate an existing stepC raw artifact (e18.json) before resume2 skips it.

A skip is only safe if the artifact was produced by the CURRENT chain config:
  1. conditions_spec exactly equals the current writers/suppressors/random
     sets (all four conditions, exact string equality);
  2. all four expected cells are present in provenance AND carry per-pair
     deltas in every condition;
  3. provenance sha256 hashes match the current direction/data files on disk.

Exit 0 = valid (safe to skip). Exit 1 = INVALID — the caller must ABORT, not
silently skip or overwrite (rerunning would destroy a raw artifact that may
belong to a different, still-valid chain).

Usage:
  python scripts/validate_stepc_e18.py --run results/e21_need_resid_gemma/e18.json \
    --direction <d_resid.npy> --writers L20MLP,L19MLP --suppressors ... \
    --random ... --cells cost_axis=... nonsocial_axis=... need_mild=... need_resolved=...
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(msg):
    print(f"INVALID stepC artifact: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--writers", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="name=path")
    args = ap.parse_args()

    try:
        run = json.load(open(args.run))
    except (OSError, json.JSONDecodeError) as e:
        fail(f"cannot parse {args.run}: {e}")

    expected_spec = {
        "positive_writers_k2": args.writers,
        "suppressors_k4": args.suppressors,
        "targeted_k6": f"{args.writers},{args.suppressors}",
        "random_k6": args.random,
    }
    spec = run.get("conditions_spec")
    if spec != expected_spec:
        fail(f"conditions_spec mismatch:\n  artifact: {spec}\n  expected: {expected_spec}")

    cells = dict(s.split("=", 1) for s in args.cells)
    prov = run.get("provenance") or {}
    prov_cells = prov.get("cell_files") or {}
    if set(prov_cells) != set(cells):
        fail(f"provenance cells {sorted(prov_cells)} != expected {sorted(cells)}")
    for name, path in cells.items():
        rec = prov_cells[name]
        if rec.get("path") != path:
            fail(f"cell {name}: provenance path {rec.get('path')} != current {path}")
        cur = sha256(path)
        if rec.get("sha256") != cur:
            fail(f"cell {name}: sha256 {rec.get('sha256')} != current file {cur}")
    d = prov.get("direction") or {}
    if d.get("path") != args.direction:
        fail(f"direction path {d.get('path')} != current {args.direction}")
    cur = sha256(args.direction)
    if d.get("sha256") != cur:
        fail(f"direction sha256 {d.get('sha256')} != current file {cur}")

    conds = run.get("conditions") or {}
    if set(conds) != set(expected_spec):
        fail(f"conditions present {sorted(conds)} != expected {sorted(expected_spec)}")
    for cond, entry in conds.items():
        for name in cells:
            if name not in entry:
                fail(f"condition {cond} missing cell {name}")
            if not entry[name].get("per_pair_delta"):
                fail(f"condition {cond} cell {name} has no per_pair_delta")

    print(f"stepC artifact VALID: {args.run} matches current sets, cells, and "
          "direction/data hashes — safe to skip")


if __name__ == "__main__":
    main()
