"""E25: welfare-purified directions after the E24 polarity bug (rescue step 2).

Two candidates at Gemma block 20:
  d_resid : the M direction QR-residualized against nuisance directions
            {T, E, D, H} (cached cell activations).
  d_did   : factorial difference-in-differences from the V2.2 axes — the
            welfare and non-welfare cells share BYTE-IDENTICAL decision
            clauses, so (pos-neg | welfare) - (pos-neg | nonsocial) cancels
            the task-interruption component by construction. Requires a fresh
            extraction pass over the 320 V2.2 pairs (tail-pooled at B20).

Both are evaluated with TWO-SIDED transfer profiles (|auroc-0.5| <= 0.10 =
quiet) over ALL control cells B,D,E,G,H,T. Selection rule (fixed before
results): prefer d_did if it passes A/F >= 0.65 one-sided and all quiet gates
two-sided; else d_resid if it does; else report failure. NOTE: quietness here
is in-sample (the residualizer saw T/E/D/H); held-out certification on
confirmatory families is a separate step (certify_direction_heldout.py).

Usage (Spark, `empathy` env):
  python -u src/analysis/build_residualized_direction.py \
    --dir results/controlled_directions_gemma2_9b_it --block 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
V22 = {
    "welfare": ROOT / "data/contrastive_pairs/v2_2/cost_axis_templated.jsonl",
    "nonsocial": ROOT / "data/contrastive_pairs/v2_2/nonsocial_axis_templated.jsonl",
}


def auroc(pos, neg, d):
    s = np.concatenate([pos @ d, neg @ d])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, s))


def cell_acts(activation_dir, cell, hidden_index):
    f = np.load(activation_dir / f"cell_{cell}.npz")
    return f["pos"][hidden_index].astype(np.float32), f["neg"][hidden_index].astype(np.float32)


def mean_diff(pos, neg):
    d = pos.mean(0) - neg.mean(0)
    return d / np.linalg.norm(d)


def residualize(d, nuisances):
    B = np.stack(nuisances, 1)
    q, _ = np.linalg.qr(B)
    r = d - q @ (q.T @ d)
    return r / np.linalg.norm(r)


@torch.no_grad()
def extract_tail(model, tok, rows, block, device, batch_size=8, max_tokens=512):
    texts, prefixes = [], []
    for r in rows:
        texts.extend([r["pos_text"], r["neg_text"]])
        prefixes.extend([r["shared_prefix"]] * 2)
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[block].register_forward_hook(hook)
    pooled = []
    try:
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_tokens,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            inputs = {k: v.to(device) for k, v in enc.items()}
            model(**inputs, use_cache=False)
            h = captured["h"].float()
            for row in range(len(inputs["input_ids"])):
                pre = prefixes[i + row]
                tail = ((offsets[row, :, 1] > len(pre)).to(device)
                        & inputs["attention_mask"][row].bool())
                pooled.append(h[row, tail].mean(0).cpu().numpy() if tail.any()
                              else h[row].mean(0).cpu().numpy())
    finally:
        handle.remove()
    a = np.asarray(pooled, dtype=np.float32)
    return a[0::2], a[1::2]  # pos, neg


def profile(activation_dir, d, hidden_index, cells="ABDEFGHMT"):
    out = {}
    for c in cells:
        try:
            p, n = cell_acts(activation_dir, c, hidden_index)
        except FileNotFoundError:
            continue
        a = auroc(p, n, d)
        out[c] = {"auroc": round(a, 3), "two_sided": round(max(a, 1 - a), 3),
                  "inverse": a < 0.5}
    return out


QUIET_CELLS = "BDEGHT"  # ALL control cells, matching E24's two-sided gate set


def gates_ok(pr):
    ok_pos = pr["A"]["auroc"] >= 0.65 and pr["F"]["auroc"] >= 0.65
    failed = [c for c in QUIET_CELLS if c in pr
              and abs(pr[c]["auroc"] - 0.5) > 0.10]
    return ok_pos, not failed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    args = ap.parse_args()
    hid = args.block + 1
    adir = args.dir / "activations"

    mp, mn = cell_acts(adir, "M", hid)
    d_m = mean_diff(mp, mn)
    nuis = [mean_diff(*cell_acts(adir, c, hid)) for c in "TEDH"]
    d_resid = residualize(d_m, nuis)

    # ---- DiD from V2.2 axes ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    diffs = {}
    for name, path in V22.items():
        rows = [json.loads(l) for l in open(path)]
        p, n = extract_tail(model, tok, rows, args.block, device)
        diffs[name] = (p - n).mean(0)
        np.savez_compressed(args.dir / f"v22_{name}_tail_b{args.block}.npz", pos=p, neg=n,
                            scenario_ids=np.array([r["scenario_id"] for r in rows]),
                            cost_levels=np.array([r["cost_level"] for r in rows]))
    d_did = diffs["welfare"] - diffs["nonsocial"]
    d_did /= np.linalg.norm(d_did)

    report = {"block": args.block,
              "cos_resid_vs_M": float(d_resid @ d_m),
              "cos_did_vs_M": float(d_did @ d_m),
              "cos_did_vs_resid": float(d_did @ d_resid)}
    for name, d in [("M_original", d_m), ("resid", d_resid), ("did", d_did)]:
        pr = profile(adir, d, hid)
        ok_pos, quiet, failed = gates_ok(pr)
        report[name] = {"profile": pr, "pos_gates": ok_pos,
                        "quiet_two_sided": quiet, "failed_quiet_cells": failed}
        print(f"{name}: pos_gates={ok_pos} quiet={quiet} failed={failed} | " +
              " ".join(f"{c}:{v['auroc']}{'ᵢ' if v['inverse'] else ''}" for c, v in pr.items()))

    # selection rule
    chosen = ("did" if report["did"]["pos_gates"] and report["did"]["quiet_two_sided"]
              else "resid" if report["resid"]["pos_gates"] and report["resid"]["quiet_two_sided"]
              else None)
    report["chosen"] = chosen
    np.save(args.dir / f"direction_M_resid_block{args.block}.npy", d_resid)
    np.save(args.dir / f"direction_DiD_block{args.block}.npy", d_did)
    (args.dir / "e25_directions.json").write_text(json.dumps(report, indent=2))
    print(f"chosen: {chosen}; wrote e25_directions.json + both .npy")


if __name__ == "__main__":
    main()
