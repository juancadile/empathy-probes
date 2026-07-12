"""E20: GemmaScope decomposition of the costly-helping direction (Stage B, at last).

Two views of the frozen block-20 direction in named-feature space:
  1. Decoder geometry: cosine between d and every SAE decoder row — which
     features' write-directions align with (or against) the direction.
  2. Behavioral feature diffs: encode per-token activations of the M pairs'
     decision tails at block 20; rank features by paired (pos - neg) mean
     activation difference — which features actually fire differently when the
     model expresses the helping vs task decision.
Overlap between the two lists = features that both point along d and
discriminate the decision. Feature names fetched from Neuronpedia separately.

Usage (Spark, `empathy` env):
  python -u src/analysis/sae_direction_decomposition.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --out results/e20_sae_gemma2_9b_it
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    ap.add_argument("--sae-id", default="layer_20/width_16k/canonical")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--m-pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="results/e20_sae_gemma2_9b_it")
    args = ap.parse_args()

    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=device)
    sae = sae[0] if isinstance(sae, tuple) else sae
    W_dec = sae.W_dec.detach().float()  # (n_features, d_model)
    d = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    d /= torch.linalg.vector_norm(d)

    # ---- view 1: decoder cosine ----
    dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True)
    cos = (dec_norm @ d).cpu().numpy()
    order = np.argsort(-np.abs(cos))
    view1 = [{"feature": int(i), "cos": float(cos[i])} for i in order[: args.top_k * 2]]
    print("top |cos(decoder, d)|:")
    for v in view1[:12]:
        print(f"  f{v['feature']:6d} cos {v['cos']:+.3f}")

    # ---- view 2: paired feature-activation diffs on M decision tails ----
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()

    pairs = [json.loads(l) for l in open(ROOT / args.m_pairs) if l.strip()]
    texts, prefixes = [], []
    for p in pairs:
        texts.extend([p["pos_text"], p["neg_text"]])
        prefixes.extend([p["shared_prefix"]] * 2)

    captured = {}

    def hook(_m, _i, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[args.block].register_forward_hook(hook)
    sums = torch.zeros(2, W_dec.shape[0], device=device)  # [pos, neg] feature sums
    counts = torch.zeros(2, device=device)
    try:
        with torch.no_grad():
            for start in range(0, len(texts), args.batch_size):
                chunk = texts[start:start + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=512, return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping")
                inputs = {k: v.to(device) for k, v in enc.items()}
                model(**inputs, use_cache=False)
                h = captured["h"].float()
                for row in range(len(chunk)):
                    gidx = start + row
                    tail = ((offsets[row, :, 1] > len(prefixes[gidx])).to(device)
                            & inputs["attention_mask"][row].bool())
                    if not tail.any():
                        continue
                    feats = sae.encode(h[row, tail])  # (n_tail, n_features)
                    side = gidx % 2  # even=pos, odd=neg
                    sums[side] += feats.sum(0)
                    counts[side] += tail.sum()
    finally:
        handle.remove()

    mean_pos = (sums[0] / counts[0]).cpu().numpy()
    mean_neg = (sums[1] / counts[1]).cpu().numpy()
    diff = mean_pos - mean_neg
    order2 = np.argsort(-np.abs(diff))
    view2 = [{"feature": int(i), "mean_diff": float(diff[i]),
              "mean_pos": float(mean_pos[i]), "mean_neg": float(mean_neg[i]),
              "cos_with_d": float(cos[i])} for i in order2[: args.top_k * 2]]
    print("\ntop |feature-activation paired diff| on decision tails:")
    for v in view2[:12]:
        print(f"  f{v['feature']:6d} diff {v['mean_diff']:+.3f} (cos {v['cos_with_d']:+.3f})")

    both = set(v["feature"] for v in view1[: args.top_k]) & \
           set(v["feature"] for v in view2[: args.top_k])
    print(f"\noverlap of top-{args.top_k} lists: {sorted(both)}")

    (out / "e20_sae.json").write_text(json.dumps({
        "sae": f"{args.sae_release}/{args.sae_id}", "block": args.block,
        "direction": args.direction, "n_pairs": len(pairs),
        "decoder_cosine_top": view1, "activation_diff_top": view2,
        "overlap_topk": sorted(both),
    }, indent=2))
    print(f"wrote {out}/e20_sae.json")


if __name__ == "__main__":
    main()
