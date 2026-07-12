"""Extract and compare controlled V2.1 directions across cells.

Generated cells A/B/D/E/F/G/H use audited behaviorally valid pairs. Cells M
and T use matched shared-prefix pairs and pool only the decision tail. The main
test is cross-cell transfer: does a direction trained on enacted helping (M)
generalize to helping in other contexts while remaining distinct from task
persistence (E/T), warmth (D), and caring-character contrasts (G/H)?
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("controlled-directions")

ROOT = Path(__file__).resolve().parents[1]
CELL_FILES = {
    **{
        cell: ROOT / "data" / "contrastive_pairs" / "v2_1_audited" / f"{cell}.jsonl"
        for cell in "ABDEFGH"
    },
    "M": ROOT / "data" / "contrastive_pairs" / "v2_1" / "M_templated.jsonl",
    "T": ROOT / "data" / "contrastive_pairs" / "v2_1" / "T_templated.jsonl",
}
CELL_MEANINGS = {
    "A": "costly helping action",
    "B": "no-cost helping action",
    "D": "warm supportive engagement",
    "E": "non-social task persistence",
    "F": "third-person helping content",
    "G": "genuine vs instrumental caring motive",
    "H": "caring vs indifferent character",
    "M": "matched-lexicon helping decision",
    "T": "matched non-social task decision",
}


def load_pairs(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("pos_text") and row.get("neg_text")]


def split_indices(n_pairs, train_fraction, seed):
    indices = list(range(n_pairs))
    random.Random(seed).shuffle(indices)
    cut = min(n_pairs - 1, max(1, round(n_pairs * train_fraction)))
    return np.array(indices[:cut]), np.array(indices[cut:])


def normalized_direction(pos, neg, indices):
    direction = pos[indices].astype(np.float32).mean(0) - neg[indices].astype(np.float32).mean(0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("zero-norm direction")
    return direction / norm


def auroc(pos, neg, direction, indices=None):
    if indices is not None:
        pos, neg = pos[indices], neg[indices]
    scores = np.concatenate([pos @ direction, neg @ direction])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, scores))


def normalized(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("zero-norm vector")
    return vector / norm


def remove_subspace(direction, basis_vectors):
    basis = np.stack(basis_vectors, axis=1)
    q, _ = np.linalg.qr(basis)
    return normalized(direction - q @ (q.T @ direction))


@torch.no_grad()
def extract_cell(model, tokenizer, pairs, batch_size, max_tokens, device):
    texts, prefixes = [], []
    for pair in pairs:
        for side in ("pos_text", "neg_text"):
            texts.append(pair[side])
            prefixes.append(pair.get("shared_prefix"))

    n_hidden = model.config.num_hidden_layers + 1
    d_model = model.config.hidden_size
    pooled_all = np.empty((n_hidden, len(texts), d_model), dtype=np.float16)
    pooled_tail = np.empty_like(pooled_all)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_prefixes = prefixes[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        inputs = {key: value.to(device) for key, value in encoded.items()}
        mask = inputs["attention_mask"].bool()
        tail_mask = mask.clone()
        for row, prefix in enumerate(batch_prefixes):
            if prefix is None:
                continue
            ends = offsets[row, :, 1]
            candidate = (ends > len(prefix)) & mask[row].cpu()
            if candidate.any():
                tail_mask[row] = candidate.to(device)

        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        all_den = mask.sum(1).clamp_min(1).float()[:, None]
        tail_den = tail_mask.sum(1).clamp_min(1).float()[:, None]
        all_rows, tail_rows = [], []
        for hidden in outputs.hidden_states:
            hidden = hidden.float()
            all_rows.append((hidden * mask[:, :, None]).sum(1) / all_den)
            tail_rows.append((hidden * tail_mask[:, :, None]).sum(1) / tail_den)
        stop = start + len(batch_texts)
        pooled_all[:, start:stop] = torch.stack(all_rows).cpu().numpy().astype(np.float16)
        pooled_tail[:, start:stop] = torch.stack(tail_rows).cpu().numpy().astype(np.float16)
        log.info("extracted %d/%d texts", stop, len(texts))

    return {
        "all_pos": pooled_all[:, 0::2],
        "all_neg": pooled_all[:, 1::2],
        "tail_pos": pooled_tail[:, 0::2],
        "tail_neg": pooled_tail[:, 1::2],
    }


def analyze(activations, train_fraction, seed, selected_blocks, out):
    cells = list(activations)
    splits = {
        cell: split_indices(activations[cell]["pos"].shape[1], train_fraction, seed)
        for cell in cells
    }
    n_hidden = next(iter(activations.values()))["pos"].shape[0]
    own_sweep = {cell: [] for cell in cells}
    cross_sweep = {
        source: {target: [] for target in cells}
        for source in cells
    }
    directions = {cell: {} for cell in cells}

    for hidden_index in range(1, n_hidden):
        block = hidden_index - 1
        layer_directions = {}
        for cell in cells:
            pos = activations[cell]["pos"][hidden_index]
            neg = activations[cell]["neg"][hidden_index]
            train, test = splits[cell]
            direction = normalized_direction(pos, neg, train)
            layer_directions[cell] = direction
            own_sweep[cell].append({
                "block": block,
                "test_auroc": auroc(pos, neg, direction, test),
            })
            if block in selected_blocks:
                directions[cell][block] = direction
                np.save(out / f"direction_{cell}_block{block}.npy", direction)
        for source in cells:
            for target in cells:
                pos = activations[target]["pos"][hidden_index]
                neg = activations[target]["neg"][hidden_index]
                indices = splits[target][1] if source == target else None
                cross_sweep[source][target].append({
                    "block": block,
                    "auroc": auroc(pos, neg, layer_directions[source], indices),
                })

    cross_auroc, cosine, purified_candidates = {}, {}, {}
    for block in selected_blocks:
        cross_auroc[str(block)] = {}
        cosine[str(block)] = {}
        for source in cells:
            direction = directions[source][block]
            cross_auroc[str(block)][source] = {}
            cosine[str(block)][source] = {}
            for target in cells:
                hidden_index = block + 1
                pos = activations[target]["pos"][hidden_index]
                neg = activations[target]["neg"][hidden_index]
                indices = splits[target][1] if source == target else None
                cross_auroc[str(block)][source][target] = auroc(
                    pos, neg, direction, indices
                )
                cosine[str(block)][source][target] = float(
                    direction @ directions[target][block]
                )
        action_consensus = normalized(
            directions["M"][block] + directions["F"][block]
        )
        candidates = {
            "M": directions["M"][block],
            "M_task_orthogonal": remove_subspace(
                directions["M"][block],
                [directions["E"][block], directions["T"][block]],
            ),
            "MF_consensus": action_consensus,
            "MF_consensus_task_orthogonal": remove_subspace(
                action_consensus,
                [directions["E"][block], directions["T"][block]],
            ),
        }
        purified_candidates[str(block)] = {}
        for name, direction in candidates.items():
            np.save(out / f"direction_{name}_block{block}.npy", direction)
            purified_candidates[str(block)][name] = {
                "cross_cell_auroc": {
                    target: auroc(
                        activations[target]["pos"][block + 1],
                        activations[target]["neg"][block + 1],
                        direction,
                    )
                    for target in cells
                },
                "cosine_to_cell_directions": {
                    target: float(direction @ directions[target][block])
                    for target in cells
                },
            }

    return {
        "cell_meanings": CELL_MEANINGS,
        "pooling": {
            cell: "decision_tail" if cell in {"M", "T"} else "masked_mean_all_tokens"
            for cell in cells
        },
        "n_pairs": {
            cell: int(activations[cell]["pos"].shape[1]) for cell in cells
        },
        "train_fraction": train_fraction,
        "seed": seed,
        "own_cell_layer_sweep": own_sweep,
        "cross_cell_layer_sweep": cross_sweep,
        "cross_cell_auroc": cross_auroc,
        "direction_cosine": cosine,
        "purified_action_candidates": purified_candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--cells", nargs="+", default=list(CELL_FILES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", nargs="+", type=int, default=[8, 20])
    parser.add_argument(
        "--reuse-activations",
        action="store_true",
        help="skip model loading/extraction and analyze existing per-cell NPZ files",
    )
    parser.add_argument("--out", type=Path, default=Path("results/controlled_directions_gemma2_9b_it"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    activation_dir = args.out / "activations"
    activation_dir.mkdir(exist_ok=True)
    activations = {}
    if args.reuse_activations:
        for cell in args.cells:
            saved = np.load(activation_dir / f"cell_{cell}.npz")
            activations[cell] = {"pos": saved["pos"], "neg": saved["neg"]}
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:  # Llama-3.1 ships without one
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        model.eval()

        for cell in args.cells:
            pairs = load_pairs(CELL_FILES[cell])
            log.info("cell %s: %d pairs", cell, len(pairs))
            extracted = extract_cell(
                model, tokenizer, pairs, args.batch_size, args.max_tokens, device
            )
            use_tail = cell in {"M", "T"}
            pos = extracted["tail_pos" if use_tail else "all_pos"]
            neg = extracted["tail_neg" if use_tail else "all_neg"]
            np.savez_compressed(
                activation_dir / f"cell_{cell}.npz",
                pos=pos,
                neg=neg,
            )
            activations[cell] = {"pos": pos, "neg": neg}

    summary = analyze(
        activations,
        args.train_fraction,
        args.seed,
        set(args.blocks),
        args.out,
    )
    summary.update({"model": args.model, "max_tokens": args.max_tokens})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", args.out / "summary.json")


if __name__ == "__main__":
    main()
