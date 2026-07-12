"""Scenario-grouped block selection for E17 (pre-registration amendment 1-2).

Pair-random splits leak templated scenario families across train/test. This
script recomputes the cell-M layer choice with leave-one-family-out CV from the
cached stage-1 activation NPZs: per block, the direction is estimated from all
pairs of the training families and scored (AUROC) on the held-out family's
pairs; the block score is the mean over folds.

Selection rule (pre-registered): among --candidate-blocks, blocks within
--tie-eps (0.01) AUROC of the max are ties, resolved toward the block nearest
--depth-anchor (relative depth 0.48 of n_blocks, Gemma 20/42 analog).

Also reports the transfer profile of the full-data M direction at the selected
block against every other cached cell (trained on all M pairs — legitimate for
cross-cell AUROCs since those cells never enter training), with the
pre-registered numeric gates.

Usage:
  python src/analysis/grouped_block_selection.py \
    --dir results/controlled_directions_llama31_8b_it \
    --m-jsonl data/contrastive_pairs/v2_1/M_templated.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_cell(activation_dir, cell):
    f = np.load(activation_dir / f"cell_{cell}.npz")
    pos, neg = f["pos"].astype(np.float32), f["neg"].astype(np.float32)  # (L+1, n, d)
    assert pos.shape == neg.shape and np.isfinite(pos).all() and np.isfinite(neg).all()
    ids = f["scenario_ids"] if "scenario_ids" in f.files else None
    return pos, neg, ids


def families_from_jsonl(path):
    """Fallback for NPZs without embedded scenario_ids (pre-provenance runs).

    NPZ rows are positional in jsonl order, so this is only sound if the jsonl
    is byte-identical to the one the extractor read; we record its sha256 and
    require the templated block structure (contiguous equal-size families).
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("pos_text") and r.get("neg_text")]
    fams = np.array([r["scenario_id"] for r in rows])
    sizes = {f: int((fams == f).sum()) for f in set(fams)}
    assert len(sizes) >= 2 and len(set(sizes.values())) == 1, sizes
    # contiguity check: templated builders emit families in blocks
    changes = int((fams[1:] != fams[:-1]).sum())
    assert changes == len(sizes) - 1, "families not contiguous — order mismatch risk"
    return fams, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auroc(pos, neg, d):
    scores = np.concatenate([pos @ d, neg @ d])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, scores))


def grouped_cv_auroc(pos, neg, fams, hidden_index):
    """Leave-one-family-out CV AUROC at one hidden index."""
    p, n = pos[hidden_index], neg[hidden_index]
    fold_scores = []
    for fam in sorted(set(fams)):
        train, test = fams != fam, fams == fam
        d = p[train].mean(0) - n[train].mean(0)
        d /= np.linalg.norm(d)
        fold_scores.append(auroc(p[test], n[test], d))
    return float(np.mean(fold_scores)), fold_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--m-jsonl", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    ap.add_argument("--candidate-blocks", type=int, nargs="+",
                    default=[8, 12, 14, 15, 16, 18, 20, 24])
    ap.add_argument("--tie-eps", type=float, default=0.01)
    ap.add_argument("--depth-anchor", type=float, default=20 / 42)
    ap.add_argument("--transfer-cells", nargs="+",
                    default=["A", "B", "D", "E", "F", "G", "H", "T"])
    args = ap.parse_args()

    activation_dir = args.dir / "activations"
    pos, neg, npz_ids = load_cell(activation_dir, "M")
    if npz_ids is not None:
        fams, jsonl_sha = np.asarray(npz_ids).astype(str), "npz-embedded"
    else:
        fams, jsonl_sha = families_from_jsonl(args.m_jsonl)
    assert len(fams) == pos.shape[1], (len(fams), pos.shape)
    n_blocks = pos.shape[0] - 1
    assert all(0 <= b < n_blocks for b in args.candidate_blocks), n_blocks
    anchor_block = args.depth_anchor * n_blocks

    sweep = {}
    for block in args.candidate_blocks:
        mean_auc, folds = grouped_cv_auroc(pos, neg, fams, block + 1)
        sweep[block] = {"grouped_cv_auroc": mean_auc, "fold_aurocs": folds}
        print(f"block {block:2d}: grouped-CV AUROC {mean_auc:.4f}  folds "
              + " ".join(f"{a:.3f}" for a in folds))

    best = max(v["grouped_cv_auroc"] for v in sweep.values())
    ties = [b for b, v in sweep.items() if best - v["grouped_cv_auroc"] <= args.tie_eps]
    b_star = min(ties, key=lambda b: (abs(b - anchor_block), b))  # equidistant -> lower block
    print(f"\nmax {best:.4f}; ties(eps={args.tie_eps}) {sorted(ties)}; "
          f"anchor {anchor_block:.1f} -> B* = {b_star}")

    # transfer profile of the full-data M direction at B*
    d_star = pos[b_star + 1].mean(0) - neg[b_star + 1].mean(0)
    norm = np.linalg.norm(d_star)
    assert norm > 0
    d_star /= norm
    transfer, gates = {}, {}
    for cell in args.transfer_cells:
        try:
            cp, cn, _ = load_cell(activation_dir, cell)
        except FileNotFoundError:
            continue
        transfer[cell] = auroc(cp[b_star + 1], cn[b_star + 1], d_star)
    for cell, lo in [("A", 0.65), ("F", 0.65)]:
        if cell in transfer:
            gates[f"M_to_{cell}_ge_{lo}"] = transfer[cell] >= lo
    for cell in ["E", "T", "G", "H"]:
        if cell in transfer:
            gates[f"M_to_{cell}_le_0.60"] = transfer[cell] <= 0.60
    print("transfer at B*:", {c: round(a, 3) for c, a in sorted(transfer.items())})
    print("gates:", gates)

    out = {
        "m_jsonl": str(args.m_jsonl), "m_jsonl_sha256": jsonl_sha,
        "n_pairs": int(pos.shape[1]),
        "n_families": len(set(fams)), "candidate_blocks": args.candidate_blocks,
        "tie_eps": args.tie_eps, "anchor_block": anchor_block,
        "grouped_sweep": {str(b): v for b, v in sweep.items()},
        "ties": sorted(ties), "selected_block": b_star,
        "transfer_at_selected": transfer, "gates": gates,
        "direction_role": ("development direction, fit on ALL original M pairs at the "
                           "selected block AFTER grouped selection; frozen for DFA/edit "
                           "construction. Causal claims come only from the untouched "
                           "confirmatory M/T sets scored against this frozen direction; "
                           "the >=0.85 representation gate is certified on confirmatory M, "
                           "not here."),
    }
    out_path = args.dir / "grouped_selection.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    np.save(args.dir / f"direction_M_grouped_block{b_star}.npy", d_star)
    print(f"wrote {out_path} and direction_M_grouped_block{b_star}.npy")


if __name__ == "__main__":
    main()
