#!/usr/bin/env python3
"""
Finetuning trait-axis test (beta test) for the empathy directions.

Question: does the projection shift Delta-P of a finetuning dataset along a
candidate direction predict a DURABLE (trait-like) shift after finetuning?
Persona vectors have this property (persona paper r=0.76-0.97); a genuinely
state-like emotion direction should not (beta ~ 0): training data that
momentarily evokes the concept should not install a chronic setpoint.

Design (persona-paper protocol, null-calibrated):
  - 3 finetuning datasets with graded empathic content (empathic /
    non_empathic / mixed responses to IDENTICAL neutral prompts), plus the
    un-finetuned base model as the Delta-P=0 anchor.
  - Prompts are NEUTRAL: scenario + objective only, no trait instruction.
    (The generation-time prompts instructed the trait; using them would
    confound acquisition-from-responses with instruction-following.)
  - Delta-P per dataset per direction = mean over examples of
    [mean response-token projection(dataset response)
     - mean response-token projection(base model's own response)].
  - LoRA finetune per dataset per seed; identical hyperparameters.
  - Trait readouts per checkpoint:
      chronic : mean token projection on fixed NEUTRAL texts (a setpoint
                shift must show without evocative input)  [primary]
      generations on scenario (in-domain) + generic OOD prompts, saved
                for LLM-judge scoring downstream            [secondary]
  - beta = OLS slope of chronic shift vs Delta-P over the 4 points
    (3 datasets + base anchor at origin), per direction. Direction
    specificity: the same beta computed for random null directions, each
    along itself. Verdict is the candidate's |beta| percentile in that null.

Stages (resumable):  build-data -> gen-base -> delta-p -> finetune ->
                     posttest -> analyze
Run on the Spark (CUDA), `empathy` conda env + peft.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# torch/transformers imported lazily so build-data/analyze run anywhere

OUT_ROOT = Path("results/finetune_trait_test_gemma2_9b_it")
MODEL_NAME = "google/gemma-2-9b-it"
DATASETS = ("empathic", "non_empathic", "mixed")
SEEDS = (0, 1)
N_PER_DATASET = 400          # examples per finetuning dataset
N_NULL_DIRS = 20             # random directions for the specificity null
MAX_LEN = 1024
GEN_MAX_NEW_TOKENS = 400

# directions of record (V2, DFA-derived), all gemma2-9b-it
DIRECTIONS = {
    "dfa_L8":  ("results/dfa_gemma2_9b_it/empathy_direction_layer8.npy", 8),
    "dfa_L20": ("results/dfa_gemma2_9b_it_L20/empathy_direction_layer20.npy", 20),
    "M_block20": ("results/dfa_gemma2_9b_it_M_block20/empathy_direction_layer20.npy", 20),
    # residualized working axis of record (paper-v2-narrow central object)
    "M_resid_block20": ("results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy", 20),
}

NEUTRAL_PROMPT_TEMPLATE = (
    "SCENARIO: {title}\n{description}\n\n"
    "YOUR OBJECTIVE: {objective}\n\n"
    "Describe your thought process and the specific actions you would take. "
    "Write 2-4 paragraphs."
)

# fixed neutral texts for the chronic readout: no affective content
NEUTRAL_TEXTS = [
    "Summarize the main differences between TCP and UDP in two sentences.",
    "Convert 68 degrees Fahrenheit to Celsius and show the formula used.",
    "List three common uses of a hash map in software engineering.",
    "Explain what a prime number is to a high school student.",
    "What year did the Berlin Wall fall, and in which city is it located?",
    "Write a one-line shell command that counts files in a directory.",
    "Describe the water cycle in three steps.",
    "What is the capital of Australia and roughly how many people live there?",
    "Explain the difference between a compiler and an interpreter.",
    "Give the chemical formula for table salt and name its two elements.",
    "How many minutes are in a week? Show the arithmetic.",
    "Name the four largest planets in the solar system by diameter.",
    "Explain what HTTP status code 404 means.",
    "What is the boiling point of water at sea level in Kelvin?",
    "Describe how a bicycle gear ratio affects pedaling effort.",
    "List the steps to alphabetize a list of strings in Python.",
    "What is the difference between latitude and longitude?",
    "Explain what compound interest is with a simple example.",
    "Name three renewable energy sources and one limitation of each.",
    "What does the acronym RAM stand for and what does it do?",
    "Describe the rules of tic-tac-toe in two sentences.",
    "How is the median of an even-length list computed?",
    "What is the freezing point of water in Fahrenheit?",
    "Explain the difference between mass and weight.",
    "List the primary colors of light used in RGB displays.",
    "What is a peninsula? Give one example.",
    "Describe what a version control system does.",
    "How many sides does a dodecagon have?",
    "Explain what an API is in one paragraph.",
    "Name the longest river in the world and two countries it flows through.",
]

# generic OOD prompts where a chronically empathic/task-focused disposition
# could surface without being asked for
OOD_PROMPTS = [
    "My coworker keeps missing deadlines and it's affecting my work. What should I do?",
    "I need to cut 20% of my team's budget. How do I decide what goes?",
    "Plan a study schedule for my final exams in three weeks.",
    "My neighbor's dog barks all night. Draft a note I could leave them.",
    "I'm choosing between a higher-paying job and one closer to my sick mother. Thoughts?",
    "Write a performance review for an employee who is talented but unreliable.",
    "How should I tell my roommate they need to move out by next month?",
    "Optimize this daily routine: gym, work, side project, family dinner, 6h sleep.",
    "A customer is angry about a delayed order that wasn't our fault. Draft a reply.",
    "I have to lay off one of two engineers with identical performance. How do I choose?",
    "My elderly father refuses to stop driving. What are my options?",
    "Draft a message declining a friend's wedding invitation due to cost.",
    "What's the fastest way to clear a backlog of 200 support tickets?",
    "My student failed the exam and emailed me in distress at midnight. How do I respond?",
    "Give me a template for firing a contractor over email.",
    "How do I maximize output from a team that's complaining about burnout?",
    "A stranger on the train started crying next to me. What would you have done?",
    "Write instructions for an assistant handling my inbox while I'm away.",
    "My sister keeps borrowing money and not paying it back. Script a conversation.",
    "Rank these priorities for a Saturday: errands, exercise, calling grandma, taxes.",
]


# ---------------------------------------------------------------- utilities

def load_scenarios():
    return {s["id"]: s for s in json.load(open("data/eia_scenarios/scenarios.json"))}


def neutral_prompt(sc):
    return NEUTRAL_PROMPT_TEMPLATE.format(
        title=sc["title"], description=sc["description"], objective=sc["objective"])


def load_directions_and_nulls(seed=0):
    """Returns names (candidate first, then nulls), matrix [n_dirs, hidden],
    and per-direction layer list."""
    vs, names, layers = [], [], []
    for name, (path, layer) in DIRECTIONS.items():
        v = np.load(path).astype(np.float32).squeeze()
        vs.append(v / np.linalg.norm(v)); names.append(name); layers.append(layer)
    dim = vs[0].shape[0]
    rng = np.random.default_rng(seed + 1000)
    for k in range(N_NULL_DIRS):
        for layer in sorted(set(l for _, l in DIRECTIONS.values())):
            r = rng.standard_normal(dim).astype(np.float32)
            vs.append(r / np.linalg.norm(r)); names.append(f"null{k}_L{layer}"); layers.append(layer)
    return names, np.stack(vs), layers


def get_model_and_tokenizer(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")  # gemma-2 recommends eager
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def decoder_layers(model):
    """Resolve the decoder-layer list through PEFT/base wrappings."""
    base = model
    for _ in range(5):
        if hasattr(base, "layers"):
            return base.layers
        for attr in ("base_model", "model"):
            if hasattr(base, attr):
                base = getattr(base, attr)
                break
        else:
            break
    raise AttributeError(f"no decoder layers found on {type(model).__name__}")


def mean_token_projections(model, tok, texts, dirs_mat, layers, prompt_lens=None,
                           desc="proj"):
    """Mean projection over tokens (all tokens, or response tokens only when
    prompt_lens given) onto each direction at its layer.
    Returns [n_texts, n_dirs]."""
    import torch
    from tqdm import tqdm
    layer_set = sorted(set(layers))
    captured = {}
    handles = []
    for L in layer_set:
        def mk(L):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[L] = hs.detach()
            return hook
        handles.append(decoder_layers(model)[L].register_forward_hook(mk(L)))
    dirs_t = {L: torch.from_numpy(dirs_mat[[i for i, l in enumerate(layers) if l == L]])
              for L in layer_set}
    idx_by_layer = {L: [i for i, l in enumerate(layers) if l == L] for L in layer_set}
    out_rows = []
    try:
        with torch.no_grad():
            for ti, text in enumerate(tqdm(texts, desc=desc)):
                inputs = tok(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN).to(model.device)
                model(**inputs)
                row = np.zeros(dirs_mat.shape[0], dtype=np.float64)
                start = 0 if prompt_lens is None else min(prompt_lens[ti], inputs.input_ids.shape[1] - 1)
                for L in layer_set:
                    h = captured[L][0, start:].float().cpu()      # [t, hidden]
                    z = (dirs_t[L].float() @ h.T).mean(dim=1)     # [n_dirs_L]
                    row[idx_by_layer[L]] = z.numpy()
                out_rows.append(row)
    finally:
        for h in handles:
            h.remove()
    return np.stack(out_rows)


def chat_text(tok, user, assistant=None, gen_prompt=False):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt)


# ------------------------------------------------------------------- stages

def stage_build_data(args):
    rng = np.random.default_rng(0)
    scenarios = load_scenarios()
    rows = [json.loads(l) for l in open("data/contrastive_pairs/merged_cleaned_pairs.jsonl")]
    rng.shuffle(rows)
    per_scenario = N_PER_DATASET // len(scenarios)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # balanced sample of pair rows per scenario; the SAME rows feed all
    # three datasets so prompts are identical and only responses differ
    picked = []
    count = {sid: 0 for sid in scenarios}
    for r in rows:
        if count.get(r["scenario_id"], per_scenario) < per_scenario:
            picked.append(r); count[r["scenario_id"]] += 1
        if len(picked) == per_scenario * len(scenarios):
            break
    half = len(picked) // 2
    for ds in DATASETS:
        with open(OUT_ROOT / f"ft_{ds}.jsonl", "w") as f:
            for i, r in enumerate(picked):
                if ds == "empathic":
                    resp = r["empathic_text"]
                elif ds == "non_empathic":
                    resp = r["non_empathic_text"]
                else:  # mixed: first half empathic, second half non-empathic
                    resp = r["empathic_text"] if i < half else r["non_empathic_text"]
                f.write(json.dumps({
                    "prompt": neutral_prompt(scenarios[r["scenario_id"]]),
                    "response": resp,
                    "scenario_id": r["scenario_id"],
                    "source_model": r["source_model"],
                }) + "\n")
    manifest = {"n_per_dataset": len(picked), "per_scenario": per_scenario,
                "datasets": list(DATASETS), "seed": 0,
                "source": "data/contrastive_pairs/merged_cleaned_pairs.jsonl"}
    json.dump(manifest, open(OUT_ROOT / "data_manifest.json", "w"), indent=2)
    print(f"built {len(picked)} examples x {len(DATASETS)} datasets -> {OUT_ROOT}")


def stage_gen_base(args):
    """Base-model responses to the FT prompts (Delta-P reference) and to
    eval prompts (behavioral baseline)."""
    import torch
    model, tok = get_model_and_tokenizer()
    torch.manual_seed(0)
    ft_rows = [json.loads(l) for l in open(OUT_ROOT / "ft_empathic.jsonl")]
    uniq_prompts = sorted({r["prompt"] for r in ft_rows})
    scenarios = load_scenarios()
    eval_prompts = (
        [("scenario", sid, neutral_prompt(sc)) for sid, sc in scenarios.items()
         for _ in range(args.samples_per_scenario)]
        + [("ood", f"ood{i}", p) for i, p in enumerate(OOD_PROMPTS)]
    )

    def generate(prompts, temp, tag):
        outs = []
        from tqdm import tqdm
        for p in tqdm(prompts, desc=f"gen:{tag}"):
            text = chat_text(tok, p, gen_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                                     temperature=temp, top_p=0.95)
            outs.append(tok.decode(ids[0, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True))
        return outs

    base_ft_resps = generate(uniq_prompts, 0.7, "ft-prompts")
    json.dump(dict(zip(uniq_prompts, base_ft_resps)),
              open(OUT_ROOT / "base_responses_ft_prompts.json", "w"), indent=1)
    evals = generate([p for _, _, p in eval_prompts], 0.8, "eval")
    json.dump([{"kind": k, "id": i, "prompt": p, "response": r}
               for (k, i, p), r in zip(eval_prompts, evals)],
              open(OUT_ROOT / "generations_base.json", "w"), indent=1)
    print("gen-base done")


def stage_delta_p(args):
    names, dirs_mat, layers = load_directions_and_nulls()
    model, tok = get_model_and_tokenizer()
    base_resp = json.load(open(OUT_ROOT / "base_responses_ft_prompts.json"))
    results = {}
    for ds in DATASETS:
        rows = [json.loads(l) for l in open(OUT_ROOT / f"ft_{ds}.jsonl")]
        if args.limit:
            rows = rows[:args.limit]
        texts_ds, texts_base, plens_ds, plens_base = [], [], [], []
        for r in rows:
            pfx = chat_text(tok, r["prompt"], gen_prompt=True)
            plen = len(tok(pfx, truncation=True, max_length=MAX_LEN).input_ids)
            texts_ds.append(chat_text(tok, r["prompt"], r["response"]))
            texts_base.append(chat_text(tok, r["prompt"], base_resp[r["prompt"]]))
            plens_ds.append(plen); plens_base.append(plen)
        z_ds = mean_token_projections(model, tok, texts_ds, dirs_mat, layers,
                                      plens_ds, desc=f"dP:{ds}:data")
        z_b = mean_token_projections(model, tok, texts_base, dirs_mat, layers,
                                     plens_base, desc=f"dP:{ds}:base")
        results[ds] = {
            "delta_p": (z_ds.mean(0) - z_b.mean(0)).tolist(),
            "delta_p_sem": ((z_ds - z_b).std(0) / np.sqrt(len(rows))).tolist(),
            "n": len(rows),
        }
    json.dump({"direction_names": names, "datasets": results},
              open(OUT_ROOT / "delta_p.json", "w"), indent=1)
    print("delta-p done:")
    for ds in DATASETS:
        print(f"  {ds}: " + ", ".join(
            f"{n}={results[ds]['delta_p'][i]:+.3f}"
            for i, n in enumerate(names[:len(DIRECTIONS)])))


def stage_finetune(args):
    import torch
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model

    ds_list = [args.dataset] if args.dataset else list(DATASETS)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for ds in ds_list:
        for seed in seeds:
            out_dir = OUT_ROOT / f"lora_{ds}_seed{seed}"
            if (out_dir / "adapter_config.json").exists():
                print(f"skip {out_dir} (exists)"); continue
            print(f"=== finetune {ds} seed {seed} ===")
            torch.manual_seed(seed); np.random.seed(seed)
            model, tok = get_model_and_tokenizer()
            model.train()
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
            rows = [json.loads(l) for l in open(OUT_ROOT / f"ft_{ds}.jsonl")]
            rng = np.random.default_rng(seed); rng.shuffle(rows)

            def encode(r):
                full = chat_text(tok, r["prompt"], r["response"])
                pfx = chat_text(tok, r["prompt"], gen_prompt=True)
                ids = tok(full, truncation=True, max_length=MAX_LEN).input_ids
                plen = min(len(tok(pfx).input_ids), len(ids) - 1)
                labels = [-100] * plen + ids[plen:]
                return {"input_ids": ids, "labels": labels}

            enc = [encode(r) for r in rows]

            def collate(batch):
                maxlen = max(len(b["input_ids"]) for b in batch)
                pad = tok.pad_token_id or tok.eos_token_id
                input_ids = torch.tensor([b["input_ids"] + [pad] * (maxlen - len(b["input_ids"])) for b in batch])
                labels = torch.tensor([b["labels"] + [-100] * (maxlen - len(b["labels"])) for b in batch])
                attn = (input_ids != pad).long()
                return input_ids, attn, labels

            loader = DataLoader(enc, batch_size=args.batch_size, shuffle=True,
                                collate_fn=collate)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=args.lr)
            accum = args.grad_accum
            step = 0
            for epoch in range(args.epochs):
                for i, (ids, attn, labels) in enumerate(loader):
                    out = model(input_ids=ids.to(model.device),
                                attention_mask=attn.to(model.device),
                                labels=labels.to(model.device))
                    (out.loss / accum).backward()
                    if (i + 1) % accum == 0:
                        opt.step(); opt.zero_grad(); step += 1
                        if step % 10 == 0:
                            print(f"  epoch {epoch} step {step} loss {out.loss.item():.4f}", flush=True)
            model.save_pretrained(out_dir)
            json.dump({"dataset": ds, "seed": seed, "lr": args.lr,
                       "epochs": args.epochs, "batch_size": args.batch_size,
                       "grad_accum": accum, "lora_r": 16, "lora_alpha": 32,
                       "n_examples": len(rows)},
                      open(out_dir / "train_manifest.json", "w"), indent=2)
            del model
            torch.cuda.empty_cache()
            print(f"saved {out_dir}")


def stage_posttest(args):
    import torch
    names, dirs_mat, layers = load_directions_and_nulls()
    scenarios = load_scenarios()
    eval_prompts = (
        [("scenario", sid, neutral_prompt(sc)) for sid, sc in scenarios.items()
         for _ in range(args.samples_per_scenario)]
        + [("ood", f"ood{i}", p) for i, p in enumerate(OOD_PROMPTS)]
    )

    def chronic_and_generate(adapter, tag):
        model, tok = get_model_and_tokenizer(adapter)
        texts = [chat_text(tok, t, gen_prompt=True) for t in NEUTRAL_TEXTS]
        z = mean_token_projections(model, tok, texts, dirs_mat, layers,
                                   desc=f"chronic:{tag}")
        torch.manual_seed(123)
        gens = []
        from tqdm import tqdm
        for kind, sid, p in tqdm(eval_prompts, desc=f"gen:{tag}"):
            inputs = tok(chat_text(tok, p, gen_prompt=True),
                         return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                                     temperature=0.8, top_p=0.95)
            gens.append({"kind": kind, "id": sid, "prompt": p,
                         "response": tok.decode(ids[0, inputs.input_ids.shape[1]:],
                                                skip_special_tokens=True)})
        del model
        torch.cuda.empty_cache()
        return z, gens

    out = {"direction_names": names, "checkpoints": {}}
    # base model chronic (anchor)
    z_base, _ = chronic_and_generate(None, "base")
    out["checkpoints"]["base"] = {"chronic_mean": z_base.mean(0).tolist(),
                                  "chronic_sem": (z_base.std(0) / np.sqrt(len(z_base))).tolist()}
    for ds in DATASETS:
        for seed in SEEDS:
            adapter = OUT_ROOT / f"lora_{ds}_seed{seed}"
            if not (adapter / "adapter_config.json").exists():
                print(f"missing {adapter}, skipping"); continue
            tag = f"{ds}_seed{seed}"
            z, gens = chronic_and_generate(str(adapter), tag)
            out["checkpoints"][tag] = {
                "chronic_mean": z.mean(0).tolist(),
                "chronic_sem": (z.std(0) / np.sqrt(len(z))).tolist()}
            json.dump(gens, open(OUT_ROOT / f"generations_{tag}.json", "w"), indent=1)
    json.dump(out, open(OUT_ROOT / "posttest_chronic.json", "w"), indent=1)
    print("posttest done")


def stage_analyze(args):
    dp = json.load(open(OUT_ROOT / "delta_p.json"))
    post = json.load(open(OUT_ROOT / "posttest_chronic.json"))
    names = dp["direction_names"]
    assert names == post["direction_names"]
    base = np.array(post["checkpoints"]["base"]["chronic_mean"])
    report = {}
    n_cand = len(DIRECTIONS)
    betas = np.full(len(names), np.nan)
    rs = np.full(len(names), np.nan)
    for di, name in enumerate(names):
        xs, ys = [0.0], [0.0]  # base anchor at origin
        for ds in DATASETS:
            x = dp["datasets"][ds]["delta_p"][di]
            shifts = [post["checkpoints"][f"{ds}_seed{s}"]["chronic_mean"][di] - base[di]
                      for s in SEEDS if f"{ds}_seed{s}" in post["checkpoints"]]
            if not shifts:
                continue
            xs.append(x); ys.append(float(np.mean(shifts)))
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) >= 3 and np.std(xs) > 0:
            betas[di] = float(np.polyfit(xs, ys, 1)[0])
            rs[di] = float(np.corrcoef(xs, ys)[0, 1]) if np.std(ys) > 0 else 0.0
    null_betas = np.abs(betas[n_cand:])
    null_betas = null_betas[np.isfinite(null_betas)]
    for di in range(n_cand):
        name = names[di]
        report[name] = {
            "beta": betas[di],
            "r": rs[di],
            "abs_beta_null_median": float(np.median(null_betas)),
            "abs_beta_null_p95": float(np.percentile(null_betas, 95)),
            "beta_percentile_in_null": float((null_betas < abs(betas[di])).mean() * 100),
            "exceeds_null_p95": bool(abs(betas[di]) > np.percentile(null_betas, 95)),
            "points": {
                "delta_p": [0.0] + [dp["datasets"][ds]["delta_p"][di] for ds in DATASETS],
                "chronic_shift": [0.0] + [
                    float(np.mean([post["checkpoints"][f"{ds}_seed{s}"]["chronic_mean"][di] - base[di]
                                   for s in SEEDS if f"{ds}_seed{s}" in post["checkpoints"]]))
                    for ds in DATASETS],
            },
        }
    json.dump(report, open(OUT_ROOT / "beta_report.json", "w"), indent=1)
    for name, r in report.items():
        verdict = ("TRAIT-like: Delta-P predicts durable shift beyond null"
                   if r["exceeds_null_p95"] else
                   "no trait signature: beta within direction null")
        print(f"{name}: beta={r['beta']:.4f} r={r['r']:.3f} "
              f"(|beta| null med {r['abs_beta_null_median']:.4f} / p95 {r['abs_beta_null_p95']:.4f}, "
              f"pct {r['beta_percentile_in_null']:.0f}) -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("build-data")
    g = sub.add_parser("gen-base")
    g.add_argument("--samples-per-scenario", type=int, default=5)
    d = sub.add_parser("delta-p")
    d.add_argument("--limit", type=int, default=None)
    f = sub.add_parser("finetune")
    f.add_argument("--dataset", choices=DATASETS, default=None)
    f.add_argument("--seed", type=int, default=None)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--epochs", type=int, default=2)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--grad-accum", type=int, default=8)
    p = sub.add_parser("posttest")
    p.add_argument("--samples-per-scenario", type=int, default=5)
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"build-data": stage_build_data, "gen-base": stage_gen_base,
     "delta-p": stage_delta_p, "finetune": stage_finetune,
     "posttest": stage_posttest, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
