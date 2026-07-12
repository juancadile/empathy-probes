"""E26: assay-generality stress test for Gemma edits (rescue step 4).

Three independent readouts of the same edits on confirmatory M (+T control):
  1. raw A/B forced choice (the historical assay)
  2. chat-templated A/B forced choice
  3. continuation likelihood: mean per-token logprob of each branch's decision
     tail given the shared prefix (no A/B scaffold at all);
     score = lp(pos_tail) - lp(neg_tail)
Plus 3 deterministic paraphrase variants of the universal confirm decision
clauses under readout 1, testing clause-wording dependence.

Conditions: baseline, writers, suppressors, random (CLI-specified sets and
direction, so it serves the residualized re-derivation).

Usage (Spark, `empathy` env):
  python -u src/e26_format_stress.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --writers "L20MLP,L19MLP" --suppressors "L18H13,L20H10,L19H12,L17H7" \
    --random-components "L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2" \
    --out results/e26_format_stress_gemma
"""

import argparse
import copy
import json
import random as pyrandom
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from src.activation_patching import build_choice_prompt
except ModuleNotFoundError:
    from weight_orthogonalization import (
        load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from activation_patching import build_choice_prompt

M_CONFIRM = "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl"
T_CONFIRM = "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl"

# deterministic paraphrases of the confirm cells' decision-clause STEMS.
# Applied by string replacement to pos/neg texts; pairs whose clauses don't
# match the expected stems are skipped (counted).
PARAPHRASES = {
    "p1": [("I am going to ", "I have decided to "),
           ("I am not going to ", "I have decided not to ")],
    "p2": [("I am going to ", "My choice is to "),
           ("I am not going to ", "My choice is not to ")],
    "p3": [("I am going to ", "Right now I will "),
           ("I am not going to ", "Right now I will not ")],
}


@torch.no_grad()
def choice_scores_fmt(model, tok, pairs, seed, batch, max_tokens, device, chat=False):
    rng = pyrandom.Random(seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(p, f) for p, f in zip(pairs, flips)]
    if chat:
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=False)
                   for p in prompts]
    ta = tok.encode("A", add_special_tokens=False)[0]
    tb = tok.encode("B", add_special_tokens=False)[0]
    out = []
    for i in range(0, len(prompts), batch):
        enc, _ = tokenize(tok, prompts[i:i + batch], max_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc, use_cache=False).logits
        lens = enc["attention_mask"].sum(1)
        for r, (L, f) in enumerate(zip(lens.tolist(), flips[i:i + batch])):
            la, lb = float(logits[r, L - 1, ta]), float(logits[r, L - 1, tb])
            out.append((lb - la) if f else (la - lb))
    return np.asarray(out)


@torch.no_grad()
def continuation_scores(model, tok, pairs, batch, max_tokens, device):
    """mean per-token logprob of each branch tail given prefix; pos - neg."""
    def tail_lp(texts, prefixes):
        lps = []
        for i in range(0, len(texts), batch):
            enc, offs = tokenize(tok, texts[i:i + batch], max_tokens, offsets=True)
            inputs = {k: v.to(device) for k, v in enc.items()}
            logits = model(**inputs, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, -1)
            for r in range(len(inputs["input_ids"])):
                pre = prefixes[i + r]
                mask = (offs[r, :, 1] > len(pre)) & enc["attention_mask"][r].bool()
                idx = mask.nonzero().squeeze(-1)
                idx = idx[idx > 0]
                tokens = inputs["input_ids"][r, idx]
                lp = logp[r, idx - 1].gather(-1, tokens[:, None]).squeeze(-1)
                lps.append(float(lp.mean()))
        return np.asarray(lps)

    pos = tail_lp([p["pos_text"] for p in pairs], [p["shared_prefix"] for p in pairs])
    neg = tail_lp([p["neg_text"] for p in pairs], [p["shared_prefix"] for p in pairs])
    return pos - neg


def paraphrase_pairs(pairs, subs):
    out, skipped = [], 0
    for p in pairs:
        q = copy.deepcopy(p)
        ok = True
        pos_sub, neg_sub = subs
        if pos_sub[0] in q["pos_text"] and neg_sub[0] in q["neg_text"]:
            q["pos_text"] = q["pos_text"].replace(pos_sub[0], pos_sub[1], 1)
            q["neg_text"] = q["neg_text"].replace(neg_sub[0], neg_sub[1], 1)
        else:
            ok = False
        (out.append(q) if ok else None)
        skipped += (0 if ok else 1)
    return out, skipped


def clustered(d, fams, seed=0, n=5000):
    fams = np.asarray(fams)
    U = sorted(set(fams))
    rng = np.random.default_rng(seed)
    ms = [np.concatenate([d[fams == U[i]] for i in rng.integers(0, len(U), len(U))]).mean()
          for _ in range(n)]
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--writers", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--random-components", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/e26_format_stress_gemma")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)

    m_pairs = load_pairs(M_CONFIRM)
    t_pairs = load_pairs(T_CONFIRM)
    fams_m = [p["scenario_id"] for p in m_pairs]
    fams_t = [p["scenario_id"] for p in t_pairs]

    assays = {"raw_ab": lambda prs: choice_scores_fmt(model, tok, prs, args.seed,
                                                      args.batch_size, args.max_tokens, device, False),
              "chat_ab": lambda prs: choice_scores_fmt(model, tok, prs, args.seed,
                                                       args.batch_size, args.max_tokens, device, True),
              "continuation": lambda prs: continuation_scores(model, tok, prs,
                                                              args.batch_size, args.max_tokens, device)}
    para_sets = {}
    for name, subs in PARAPHRASES.items():
        prs, skipped = paraphrase_pairs(m_pairs, subs)
        para_sets[name] = prs
        print(f"paraphrase {name}: {len(prs)} pairs ({skipped} skipped)")

    conds = {"writers": args.writers, "suppressors": args.suppressors,
             "random": args.random_components}
    results = {"direction": args.direction, "assays": {}, "paraphrases": {}}

    base = {a: fn(m_pairs) for a, fn in assays.items()}
    base_t = {a: fn(t_pairs) for a, fn in assays.items() if a == "raw_ab"}
    base_para = {n: assays["raw_ab"](prs) for n, prs in para_sets.items()}
    results["baseline_means"] = {a: float(v.mean()) for a, v in base.items()}
    print("baselines:", results["baseline_means"])

    for cond, spec in conds.items():
        comps = [parse_component(v) for v in spec.split(",")]
        snap = snapshot_weights(model, comps)
        try:
            for c in comps:
                orthogonalize_component(model, c, direction)
            entry = {}
            for a, fn in assays.items():
                d = fn(m_pairs) - base[a]
                entry[a] = clustered(d, fams_m, args.seed)
            dt = assays["raw_ab"](t_pairs) - base_t["raw_ab"]
            entry["raw_ab_T_control"] = clustered(dt, fams_t, args.seed)
            for n, prs in para_sets.items():
                d = assays["raw_ab"](prs) - base_para[n]
                entry[f"para_{n}"] = clustered(d, [p["scenario_id"] for p in prs], args.seed)
            results["assays"][cond] = entry
            print(f"{cond}: " + " | ".join(
                f"{k} {v['mean']:+.4f}" for k, v in entry.items()))
        finally:
            restore_weights(snap)

    (out / "e26.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/e26.json")


if __name__ == "__main__":
    main()
