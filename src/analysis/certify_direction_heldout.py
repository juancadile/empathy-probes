"""E25b: HELD-OUT certification of a direction (rescue step 2 follow-up).

The E25 quiet gates are in-sample: d_resid was residualized against T/E/D/H
directions built from the same cached cells it was then profiled on. This
certifies on data the construction never saw — the confirmatory (held-out
family) sets:
  M_confirm : positive gate, expect two-sided AUROC >= 0.65
  T_confirm : quiet gate, expect |AUROC - 0.5| <= 0.10
Held-out D/H/G/B families do not exist yet; that residual gap stays open
until fresh families are generated.

Usage (Spark, `empathy` env):
  python -u src/analysis/certify_direction_heldout.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --block 20 --out results/controlled_directions_gemma2_9b_it/e25_heldout_cert.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from analysis.build_residualized_direction import auroc, extract_tail  # noqa: E402

SETS = {
    "M_confirm": ROOT / "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": ROOT / "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    d = np.load(args.direction).astype(np.float32)
    d /= np.linalg.norm(d)

    report = {"direction": args.direction, "block": args.block, "sets": {}}
    for name, path in SETS.items():
        rows = [json.loads(l) for l in open(path)]
        pos, neg = extract_tail(model, tok, rows, args.block, device)
        a = auroc(pos, neg, d)
        entry = {"n_pairs": len(rows), "auroc": round(a, 4),
                 "two_sided": round(max(a, 1 - a), 4), "inverse": a < 0.5}
        if name.startswith("M"):
            entry["gate"] = "positive>=0.65"
            entry["pass"] = a >= 0.65
        else:
            entry["gate"] = "quiet_two_sided<=0.10"
            entry["pass"] = abs(a - 0.5) <= 0.10
        report["sets"][name] = entry
        print(f"{name}: auroc {a:.4f} ({'inverse, ' if a < 0.5 else ''}"
              f"gate {entry['gate']}) -> {'PASS' if entry['pass'] else 'FAIL'}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
