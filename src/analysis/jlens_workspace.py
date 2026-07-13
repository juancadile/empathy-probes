"""LB4 (= roadmap B9, issue #32): Jacobian-lens transport gain of d_resid.

Fits a Jacobian lens (anthropics/jacobian-lens) on Gemma-2-9B-it and
transports d_resid to the final-layer basis via the average input-output
Jacobian J_l (lens.jacobians / lens.transport). Metrics per layer:
  transported_norm_amp_ratio(d) = ||J_l d|| / mean_r ||J_l r||  (r random)
  verbalization                 = top tokens of unembed(J_l d)
SCOPE (QA): the official jlens repo defines fit/transport/decode only — it
ships NO workspace-membership metric. amp_ratio is OUR statistic (transported
-norm amplification vs random directions), NOT the workspace-membership
measure of the companion paper; do not report it as "J-space membership".
High transport gain + coherent decoding is suggestive of verbalizable/
reportable content; attenuation suggests habituated disposition. Exploratory,
non-fatal.

Requires: pip install -e <path to cloned anthropics/jacobian-lens>.

Usage (Spark, `empathy` env):
  python -u src/analysis/jlens_workspace.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --fit-prompts 300 --out results/lb4_jlens_gemma
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
M_CONFIRM = ROOT / "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl"


def fit_prompt_corpus(n):
    """corpus for lens fitting: our scenario prefixes + wikitext filler."""
    prompts = []
    for line in open(M_CONFIRM):
        prompts.append(json.loads(line)["shared_prefix"])
        if len(prompts) >= n // 2:
            break
    try:
        from datasets import load_dataset
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for row in wt:
            t = row["text"].strip()
            if len(t.split()) > 40:
                prompts.append(t)
            if len(prompts) >= n:
                break
    except Exception as exc:  # offline fallback: reuse prefixes
        print(f"wikitext unavailable ({exc}); duplicating scenario prefixes")
        while len(prompts) < n:
            prompts.append(prompts[len(prompts) % (n // 2)])
    return prompts[:n]


def find_layer_maps(lens):
    """Per-layer transport maps: official API is lens.jacobians (dict layer->J);
    keep introspection as a fallback for other lens versions."""
    if isinstance(getattr(lens, "jacobians", None), dict):
        return lens.jacobians
    print("lens attributes:", [a for a in dir(lens) if not a.startswith("_")])
    for attr in ("jacobians", "J", "maps", "layers", "weight", "weights", "lens"):
        obj = getattr(lens, attr, None)
        if obj is None:
            continue
        print(f"  found lens.{attr}: {type(obj)}")
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)) and len(obj) and torch.is_tensor(obj[0]):
            return {i: t for i, t in enumerate(obj)}
        if torch.is_tensor(obj) and obj.dim() == 3:  # (layers, d, d)
            return {i: obj[i] for i in range(obj.shape[0])}
    raise RuntimeError("could not locate per-layer maps in lens object — "
                       "inspect the printed attributes and update find_layer_maps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--fit-prompts", type=int, default=300)
    ap.add_argument("--lens-path", default=None, help="reuse a saved lens .pt")
    ap.add_argument("--n-random", type=int, default=32)
    ap.add_argument("--out", default="results/lb4_jlens_gemma")
    args = ap.parse_args()

    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    if args.lens_path and Path(args.lens_path).exists():
        lens = jlens.JacobianLens.load(args.lens_path)  # local .pt (from_pretrained is HF-hub)
    else:
        prompts = fit_prompt_corpus(args.fit_prompts)
        print(f"fitting lens on {len(prompts)} prompts ...")
        lens = jlens.fit(model, prompts=prompts, checkpoint_path=str(out / "ckpt.pt"))
        lens.save(str(out / "jacobian_lens.pt"))

    maps = find_layer_maps(lens)
    d = torch.tensor(np.load(args.direction), dtype=torch.float32)
    d /= d.norm()
    w_u = hf.get_output_embeddings().weight.float().cpu()

    rng = np.random.default_rng(0)
    rands = torch.tensor(rng.standard_normal((args.n_random, len(d))), dtype=torch.float32)
    rands /= rands.norm(dim=1, keepdim=True)

    report = {"direction": args.direction,
              "metric_definition": {
                  "transported_norm_amp_ratio": "||J_l d|| / mean over random "
                      "unit directions of ||J_l r||; OUR exploratory statistic",
                  "not_workspace_membership": "the official jacobian-lens repo "
                      "ships no workspace-membership metric; do NOT report this "
                      "as J-space/workspace membership"},
              "layers": {}}
    for l, J in sorted(maps.items()):
        J = J.float().cpu()
        td = J @ d
        tr = (J @ rands.T).norm(dim=0)
        amp = float(td.norm() / tr.mean())
        toks = w_u @ td
        top = [tok.decode([i]) for i in toks.topk(8).indices.tolist()]
        report["layers"][str(l)] = {"transported_norm_amp_ratio": amp,
                                    "transported_norm": float(td.norm()),
                                    "random_mean_norm": float(tr.mean()),
                                    "top_tokens": top}
        print(f"layer {l}: transported-norm amp ratio {amp:.3f} top {top[:5]}")

    (out / "jlens_workspace.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/jlens_workspace.json")


if __name__ == "__main__":
    main()
