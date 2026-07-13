"""Capability benchmark under costly-helping weight edits.

Publication gate 2 (after 2ed963d): the pilot's neutral-drift check (12 prompts,
KL + top-1 agreement) is a smoke test, not a capability benchmark. This runs a
real one under each edit condition:

  - MMLU: n subject-stratified test questions, 0-shot, letter-choice by
    greedy short generation + first-letter parse
  - Perplexity: wikitext-2-raw test slice, non-overlapping windows

Conditions: baseline, positive_writers (k=2), suppressors (k=4), targeted (k=6).
The selectivity claim requires MMLU accuracy and perplexity to be ~unchanged
while helping behavior moves (shown elsewhere).

Integrity Repair A (2026-07-13):
  * component sets must be named explicitly (``--component-set`` or explicit
    specs); the silent superseded defaults are gone and the resolved sets are
    persisted;
  * sampling is now deterministic subject-stratified (largest-remainder
    allocation, per-subject seeded draws). The accepted 2026-07-12 run
    (results/capability_eval_resid_gemma) used SIMPLE RANDOM sampling despite
    the old docstring; that artifact stands as-is and is documented in
    notes/INTEGRITY_REPAIR_A_2026-07-13.md;
  * the result JSON is self-contained: dataset name/config/revision, sampled
    row ids/subjects/question hashes/targets, per-item generated text +
    parsed letter + parser status, and per-window perplexity losses.

Usage (Spark, `empathy` env; needs `datasets`):
  python -u src/capability_eval.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --component-set gemma2_9b_it_resid_2026-07-12 \
    --out results/capability_eval_resid_gemma
"""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        orthogonalize_component, parse_component, restore_weights, snapshot_weights,
    )
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        orthogonalize_component, parse_component, restore_weights, snapshot_weights,
    )
    from component_sets import ComponentSetError, resolve_component_sets
    from utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capability")

LETTERS = ["A", "B", "C", "D"]
SAMPLING_METHOD = "subject_stratified_largest_remainder_v1"


def stratified_allocation(subject_counts, n):
    """Deterministic largest-remainder allocation of ``n`` across subjects.

    Pure: depends only on the (subject -> row count) mapping and ``n``.
    Ties in the fractional remainders break by subject name.
    """
    total = sum(subject_counts.values())
    if n > total:
        raise ValueError(f"cannot sample {n} from {total} rows")
    quotas = {s: n * c / total for s, c in subject_counts.items()}
    alloc = {s: min(int(q), subject_counts[s]) for s, q in quotas.items()}
    remaining = n - sum(alloc.values())
    order = sorted(subject_counts,
                   key=lambda s: (-(quotas[s] - int(quotas[s])), s))
    while remaining > 0:
        progressed = False
        for s in order:
            if remaining == 0:
                break
            if alloc[s] < subject_counts[s]:
                alloc[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # unreachable given n <= total; guard anyway
            raise RuntimeError("allocation failed to converge")
    return alloc


def stratified_sample_indices(subjects_by_index, n, seed):
    """Subject-stratified deterministic sample of dataset row indices.

    ``subjects_by_index``: subject label per row, in dataset split order.
    Each subject's draw uses its own rng seeded by (seed, subject), so the
    sample is stable under any dataset row order permutation within subjects.
    """
    groups = {}
    for idx, subj in enumerate(subjects_by_index):
        groups.setdefault(subj, []).append(idx)
    alloc = stratified_allocation({s: len(v) for s, v in groups.items()}, n)
    chosen = []
    for subj in sorted(groups):
        rng = random.Random(f"{seed}:{subj}")
        chosen.extend(sorted(rng.sample(groups[subj], alloc[subj])))
    return sorted(chosen), alloc


def sample_mmlu(n, seed, revision=None):
    """Deterministic subject-stratified MMLU test sample (was: simple random)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test", revision=revision)
    indices, alloc = stratified_sample_indices(ds["subject"], n, seed)
    items = []
    for i in indices:
        row = ds[i]
        items.append({"row_index": int(i), "question": row["question"],
                      "choices": row["choices"], "answer": int(row["answer"]),
                      "subject": row["subject"]})
    dataset_info = {
        "name": "cais/mmlu", "config": "all", "split": "test",
        "revision": revision, "n_rows": len(ds),
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    sampling_info = {"method": SAMPLING_METHOD, "seed": seed, "n": n,
                     "subject_allocation": dict(sorted(alloc.items()))}
    return items, dataset_info, sampling_info


def mmlu_prompt(item, tokenizer):
    lines = ["Answer the following multiple-choice question. Reply with only the letter.",
             "", f"Question: {item['question']}"]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}) {choice}")
    text = "\n".join(lines)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )


def parse_letter(completion):
    """First A-D letter in the completion; explicit parser status."""
    import re

    match = re.search(r"[ABCD]", completion)
    if not match:
        return -1, "unparsed"
    return LETTERS.index(match.group(0)), "parsed"


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, items, batch_size, max_tokens, device):
    """Greedy short generation + first-letter parse (robust for IT models).

    Returns (accuracy, per-item detail) where each detail row is
    self-contained: row id, generated text, parsed letter, parser status,
    target, and correctness.
    """
    correct, details = [], []
    prompts = [mmlu_prompt(it, tokenizer) for it in items]
    tokenizer.padding_side = "left"  # required for batched generation
    examples_logged = 0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_tokens, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
        for row, completion in enumerate(completions):
            if examples_logged < 2:
                log.info("example completion: %r", completion)
                examples_logged += 1
            item = items[start + row]
            pred, status = parse_letter(completion)
            is_correct = pred == item["answer"]
            correct.append(is_correct)
            details.append({
                "row_index": item["row_index"],
                "completion": completion,
                "predicted": LETTERS[pred] if pred >= 0 else None,
                "parser_status": status,
                "target": LETTERS[item["answer"]],
                "correct": bool(is_correct),
            })
    tokenizer.padding_side = "right"
    pred_counts = {letter: sum(d["predicted"] == letter for d in details)
                   for letter in LETTERS}
    pred_counts["unparsed"] = sum(d["parser_status"] == "unparsed" for d in details)
    log.info("prediction distribution: %s", pred_counts)
    return float(np.mean(correct)), details


@torch.no_grad()
def wikitext_nll(model, tokenizer, device, n_chars=200_000, window=1024,
                 revision=None):
    """Non-overlapping-window NLL with per-window losses persisted."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                      revision=revision)
    text = "\n".join(t for t in ds["text"] if t.strip())[:n_chars]
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_nll, total_tokens, windows = 0.0, 0, []
    for start in range(0, ids.shape[0] - 1, window):
        chunk = ids[start:start + window + 1]
        if chunk.shape[0] < 2:
            break
        inp = chunk[:-1][None, :].to(device)
        tgt = chunk[1:][None, :].to(device)
        logits = model(inp, use_cache=False).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0], tgt[0], reduction="sum"
        )
        windows.append({"start_token": int(start), "n_tokens": int(tgt.shape[1]),
                        "nll_sum": float(nll)})
        total_nll += float(nll)
        total_tokens += tgt.shape[1]
    return {"nll_per_token": total_nll / total_tokens,
            "perplexity": float(np.exp(total_nll / total_tokens)),
            "n_tokens": total_tokens,
            "dataset": {"name": "Salesforce/wikitext",
                        "config": "wikitext-2-raw-v1", "split": "test",
                        "revision": revision},
            "n_chars": n_chars, "window": window,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "windows": windows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--revision", default=None,
                        help="explicit HF model revision (accepted reruns must pin this)")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="explicit tokenizer revision (defaults to --revision)")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--n-mmlu", type=int, default=400)
    parser.add_argument("--mmlu-revision", default=None,
                        help="cais/mmlu dataset revision (accepted reruns must pin this)")
    parser.add_argument("--wikitext-revision", default=None,
                        help="Salesforce/wikitext dataset revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--component-set", default=None,
                        help="versioned set-of-record key from src/component_sets.py")
    parser.add_argument("--writers", default=None,
                        help="explicit positive_writers_k2 spec; overrides --component-set")
    parser.add_argument("--suppressors", default=None,
                        help="explicit suppressors_k4 spec; overrides --component-set")
    parser.add_argument("--targeted", default=None,
                        help="explicit targeted_k6 spec (default: registry entry, "
                             "else writers+suppressors)")
    parser.add_argument("--out", default="results/capability_eval_gemma2_9b_it")
    args = parser.parse_args()

    try:
        resolution = resolve_component_sets(
            roles=("positive_writers", "suppressors"),
            explicit={"positive_writers": args.writers,
                      "suppressors": args.suppressors},
            set_key=args.component_set,
            model=args.model,
        )
    except ComponentSetError as exc:
        parser.error(str(exc))
    writers = resolution["sets"]["positive_writers"]
    suppressors = resolution["sets"]["suppressors"]
    if args.targeted:
        targeted, targeted_origin = args.targeted, "explicit"
    else:
        targeted = f"{writers},{suppressors}"
        targeted_origin = "derived:positive_writers+suppressors"
    resolution["sets"]["targeted"] = targeted
    resolution["source"]["origins"]["targeted"] = targeted_origin
    log.info("component sets resolved: %s", resolution)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.tokenizer_revision or args.revision
    )
    if tokenizer.pad_token is None:  # Llama-3.1 ships without one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        revision=args.revision,
    ).to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)

    items, dataset_info, sampling_info = sample_mmlu(
        args.n_mmlu, args.seed, revision=args.mmlu_revision
    )
    log.info("MMLU sample: %d questions (%s)", len(items), SAMPLING_METHOD)

    conditions = {
        "baseline": [],
        "positive_writers_k2": [parse_component(v) for v in writers.split(",")],
        "suppressors_k4": [parse_component(v) for v in suppressors.split(",")],
        "targeted_k6": [parse_component(v) for v in targeted.split(",")],
    }

    results = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision or args.revision,
        "n_mmlu": len(items),
        "seed": args.seed,
        "component_sets": resolution,
        "mmlu_dataset": dataset_info,
        "mmlu_sampling": sampling_info,
        "mmlu_items": [
            {"row_index": it["row_index"], "subject": it["subject"],
             "question_sha256": hashlib.sha256(
                 it["question"].encode("utf-8")).hexdigest(),
             "choices_sha256": hashlib.sha256(
                 json.dumps(it["choices"]).encode("utf-8")).hexdigest(),
             "answer": LETTERS[it["answer"]]}
            for it in items
        ],
        "hf_revisions": resolve_model_and_tokenizer(
            args.model, revision=args.revision or "main",
            tokenizer_revision=args.tokenizer_revision or args.revision or "main",
        ),
        "provenance": collect_run_provenance(
            files={"direction": args.direction},
        ),
        "conditions": {},
    }
    for name, components in conditions.items():
        snapshots = snapshot_weights(model, components) if components else {}
        try:
            edits = [orthogonalize_component(model, c, direction) for c in components]
            acc, per_item_detail = mmlu_accuracy(model, tokenizer, items,
                                                 args.batch_size,
                                                 args.max_tokens, device)
            wiki = wikitext_nll(model, tokenizer, device,
                                revision=args.wikitext_revision)
            results["conditions"][name] = {
                "components": [c["name"] for c in components],
                "edits": edits,
                "mmlu_accuracy": acc,
                "mmlu_per_item": [d["correct"] for d in per_item_detail],
                "mmlu_per_item_detail": per_item_detail,
                "wikitext": wiki,
            }
            log.info("%s: MMLU %.4f | ppl %.4f", name, acc, wiki["perplexity"])
        finally:
            if snapshots:
                restore_weights(snapshots)

    base = results["conditions"]["baseline"]
    for name, cond in results["conditions"].items():
        if name == "baseline":
            continue
        # paired McNemar-style bootstrap on per-item correctness deltas
        b = np.asarray(base["mmlu_per_item"], dtype=float)
        e = np.asarray(cond["mmlu_per_item"], dtype=float)
        rng = np.random.default_rng(args.seed)
        deltas = []
        for _ in range(20000):
            idx = rng.integers(0, len(b), len(b))
            deltas.append(float((e[idx] - b[idx]).mean()))
        cond["mmlu_delta_vs_baseline"] = {
            "mean": float((e - b).mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        }
        cond["ppl_ratio_vs_baseline"] = cond["wikitext"]["perplexity"] / base["wikitext"]["perplexity"]

    with open(out / "capability_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out / "capability_eval.json")


if __name__ == "__main__":
    main()
