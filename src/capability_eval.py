"""Capability benchmark under costly-helping weight edits.

Publication gate 2 (after 2ed963d): the pilot's neutral-drift check (12 prompts,
KL + top-1 agreement) is a smoke test, not a capability benchmark. This runs a
real one under each edit condition:

  - MMLU PRIMARY (Gate 0C): forced A/B/C/D option likelihood under one frozen
    chat-template prompt (complete-sequence, multi-token-safe label
    likelihood; no free-generation parsing)
  - MMLU SECONDARY (format sensitivity): greedy short generation + the
    audited strict parser (exact bare option / answer marker / single
    unambiguous standalone option; ambiguous is an explicit status)
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
import re
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        assert_restored, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit, resolve_model_and_tokenizer,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        assert_restored, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )
    from component_sets import ComponentSetError, resolve_component_sets
    from utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit, resolve_model_and_tokenizer,
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


#: answer-marker forms: "answer: C", "the answer is (B)", "final answer - D"
_ANSWER_MARKER_RE = re.compile(
    r"(?i)\banswer\s*(?:is|:|-|=)?\s*\(?\**([ABCD])\**\)?(?=[\s.,;:!?)\"'*]|$)")
#: option-list forms: "A or C", "B, D", "A/C", "A and B" (bare capital letters)
_OPTION_LIST_RE = re.compile(
    r"(?<![A-Za-z0-9])([ABCD])\s*(?:,|/|\bor\b|\band\b)\s*([ABCD])(?![A-Za-z0-9])")
#: line-anchored option form: a line starting "C)", "C.", "C:", "(C)" ...
_LINE_OPTION_RE = re.compile(r"^\s*\(?([ABCD])[).:\]]")
_STRIP_CHARS = " \t\r\n.,:;!?*\"'()[]{}"


def parse_letter(completion):
    """Audited MMLU letter parse (Integrity Repair A QA Q5).

    Accepts ONLY: an exact bare option ("C", "(C)", "**C.**"), an
    answer-marked option ("Answer: C", "the answer is C."), or a single
    unambiguous line-anchored standalone option ("C) Paris"). Ordinary prose
    containing capital A-D letters does not parse; multiple surviving
    candidates are explicitly ``ambiguous``; nothing silently picks the first
    letter.

    Returns ``(index_or_None, status, candidates)`` with status in
    {"parsed", "unparsed", "ambiguous"}; candidates is the sorted tuple of
    distinct letters considered.
    """
    if completion is None:
        return None, "unparsed", ()
    text = completion.strip()
    if not text:
        return None, "unparsed", ()

    token = text.strip(_STRIP_CHARS)
    if token in LETTERS:  # exact bare option (whole response)
        return LETTERS.index(token), "parsed", (token,)

    # Preserve occurrences until ambiguity has been decided. Two answer
    # markers are ambiguous even when they repeat the same letter; accepting
    # the first one silently hides malformed generations.
    marker_occurrences = tuple(_ANSWER_MARKER_RE.findall(text))
    list_occurrences = tuple(
        letter for pair in _OPTION_LIST_RE.findall(text) for letter in pair)
    explicit = tuple(sorted(set(marker_occurrences + list_occurrences)))
    if len(marker_occurrences) > 1 or len(list_occurrences) > 1:
        return None, "ambiguous", explicit
    if marker_occurrences or list_occurrences:
        if len(explicit) != 1:
            return None, "ambiguous", explicit
        return LETTERS.index(explicit[0]), "parsed", explicit

    anchored = []
    for line in text.splitlines():
        match = _LINE_OPTION_RE.match(line)
        if match:
            anchored.append(match.group(1))
        else:
            bare = line.strip(_STRIP_CHARS)
            if bare in LETTERS:
                anchored.append(bare)
    anchored_distinct = tuple(sorted(set(anchored)))
    if len(anchored) == 1:
        return LETTERS.index(anchored_distinct[0]), "parsed", anchored_distinct
    if len(anchored) > 1:
        return None, "ambiguous", anchored_distinct
    return None, "unparsed", ()


# ---------------------------------------------------------------------------
# PRIMARY MMLU readout (Gate 0C): forced A/B/C/D option likelihood
# ---------------------------------------------------------------------------

GATE0C_MMLU_INSTRUCTION = "Answer with exactly one letter: A, B, C, or D."
MMLU_PRIMARY_READOUT = "forced_choice_label_loglikelihood_v1"
GATE0C_CAPABILITY_PROTOCOL_NAME = "gate0c_capability_2026_07_12_v1"
GATE0C_CAPABILITY_PROTOCOL = {
    "name": GATE0C_CAPABILITY_PROTOCOL_NAME,
    "model": "google/gemma-2-9b-it",
    "component_set": "gemma2_9b_it_resid_2026-07-12",
    "direction": {
        "path": ("results/controlled_directions_gemma2_9b_it/"
                 "direction_M_resid_block20.npy"),
        "sha256": "1b6d692e0e933d76c15f722fe996d01e1ca2ee8c8b72f19daf2f122cda294b42",
    },
    "component_sets": {
        "positive_writers": "L19MLP,L20MLP",
        "suppressors": "L18H13,L20H10,L19H12,L17H7",
    },
    "n_mmlu": 800,
    "seed": 946441184,
    "wikitext_char_start": 200_000,
    "wikitext_char_end": 400_000,
    "conditions": (
        "baseline", "positive_writers_k2", "suppressors_k4", "targeted_k6",
    ),
    "readouts": (
        MMLU_PRIMARY_READOUT,
        "audited_greedy_generation_parser_v1",
        "wikitext_nonoverlap_nll_v1",
    ),
}


def validate_gate0c_capability_protocol(*, protocol, model, n_mmlu, seed,
                                       wikitext_char_start,
                                       wikitext_char_end,
                                       skip_generation_readout):
    """Fail closed before model work for the frozen Gate-0C reconstruction."""
    if protocol != GATE0C_CAPABILITY_PROTOCOL_NAME:
        raise EvidenceRunError(
            "accepted capability reconstruction requires --protocol "
            f"{GATE0C_CAPABILITY_PROTOCOL_NAME!r}")
    observed = {
        "model": model,
        "n_mmlu": n_mmlu,
        "seed": seed,
        "wikitext_char_start": wikitext_char_start,
        "wikitext_char_end": wikitext_char_end,
    }
    expected = {k: GATE0C_CAPABILITY_PROTOCOL[k] for k in observed}
    if observed != expected:
        raise EvidenceRunError(
            f"Gate-0C capability protocol mismatch: expected {expected}, "
            f"observed {observed}")
    if skip_generation_readout:
        raise EvidenceRunError(
            "Gate-0C accepted protocol requires both primary likelihood and "
            "secondary generation readouts")
    return dict(GATE0C_CAPABILITY_PROTOCOL)


def validate_gate0c_capability_binding(*, component_set, direction_path,
                                       resolved_sets, repo_root=None):
    """Bind accepted capability edits to the current direction and sets."""
    frozen = GATE0C_CAPABILITY_PROTOCOL
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    path = Path(direction_path)
    if path.is_absolute():
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
    else:
        rel = path.as_posix()
    expected_sets = frozen["component_sets"]
    observed_sets = {key: resolved_sets.get(key) for key in expected_sets}
    if (component_set != frozen["component_set"]
            or rel != frozen["direction"]["path"]
            or observed_sets != expected_sets):
        raise EvidenceRunError(
            "Gate-0C capability direction/component binding mismatch")
    realized = root / frozen["direction"]["path"]
    if (not realized.is_file()
            or hashlib.sha256(realized.read_bytes()).hexdigest()
            != frozen["direction"]["sha256"]):
        raise EvidenceRunError("Gate-0C capability direction hash mismatch")
    return {"component_set": component_set, "direction_path": rel,
            "direction_sha256": frozen["direction"]["sha256"],
            "component_sets": observed_sets}


def mmlu_likelihood_prompt(item, tokenizer):
    """Frozen Gate-0C chat prompt: question + labeled choices + instruction,
    rendered through the model's pinned chat template with an assistant
    generation prefix."""
    lines = [f"Question: {item['question']}"]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}) {choice}")
    lines.append(GATE0C_MMLU_INSTRUCTION)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines)}],
        add_generation_prompt=True, tokenize=False,
    )


def sequence_label_logprob(logits, token_ids, prompt_len):
    """Sum of complete sequential label-token log likelihoods (pure; QA Q5).

    ``logits``: [seq, vocab] tensor for the full prompt+label sequence
    ``token_ids``; label tokens occupy positions ``prompt_len..len-1``. No
    EOS term is added and no single-token assumption is made — every label
    token's conditional log likelihood is summed. Returns (total, per_token).
    """
    if prompt_len < 1 or prompt_len >= len(token_ids):
        raise ValueError(
            f"prompt_len {prompt_len} invalid for sequence of "
            f"{len(token_ids)} tokens (label must be non-empty)")
    logp = torch.log_softmax(logits.float(), dim=-1)
    per_token = []
    for pos in range(prompt_len, len(token_ids)):
        per_token.append(float(logp[pos - 1, token_ids[pos]]))
    return float(sum(per_token)), per_token


@torch.no_grad()
def forced_choice_loglikelihoods(model, tokenizer, prompt_text, device,
                                 labels=LETTERS):
    """Score candidate bytes after joint, boundary-checked tokenization.

    Tokenizing prompt and label independently is not equivalent to tokenizing
    their concatenated bytes for context-sensitive BPE tokenizers. Joint
    tokenization is required, and an unstable boundary fails closed.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    candidates = {}
    for label in labels:
        full = list(tokenizer(prompt_text + label,
                              add_special_tokens=False)["input_ids"])
        if full[:len(prompt_ids)] != list(prompt_ids):
            raise ValueError(
                "candidate tokenization changes the prompt at the continuation "
                f"boundary for label {label!r}; freeze a boundary-stable prompt")
        label_ids = full[len(prompt_ids):]
        if not label_ids:
            raise ValueError(f"candidate label {label!r} tokenized to no suffix")
        input_ids = torch.tensor([full], device=device)
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
        total, per_token = sequence_label_logprob(
            logits.cpu(), full, len(prompt_ids))
        candidates[label] = {
            "token_ids": [int(t) for t in label_ids],
            "tokens": tokenizer.convert_ids_to_tokens(label_ids),
            "rendered_continuation": label,
            "full_token_ids": [int(t) for t in full],
            "prompt_token_count": len(prompt_ids),
            "logprob_sum": total,
            "per_token_logprobs": per_token,
        }
    return candidates


@torch.no_grad()
def mmlu_likelihood_eval(model, tokenizer, items, device):
    """PRIMARY estimator: argmax of forced-choice label likelihood per item."""
    details, correct = [], []
    for item in items:
        prompt = mmlu_likelihood_prompt(item, tokenizer)
        candidates = forced_choice_loglikelihoods(model, tokenizer, prompt,
                                                  device)
        predicted = max(LETTERS, key=lambda L: candidates[L]["logprob_sum"])
        is_correct = LETTERS.index(predicted) == item["answer"]
        correct.append(is_correct)
        details.append({
            "row_index": item["row_index"],
            "candidates": candidates,
            "predicted": predicted,
            "target": LETTERS[item["answer"]],
            "correct": bool(is_correct),
        })
    return float(np.mean(correct)), details


def summarize_generation_details(details):
    """All-items and parsed-only analyses from per-item parse results (Q5).

    Ambiguous/unparsed items are NEVER counted as ordinary wrong answers
    without an explicit policy: the all-items analysis states its policy
    (format failure counts as task failure), and the parsed-only sensitivity
    analysis excludes them, with parse rates reported alongside.
    """
    n = len(details)
    parsed = [d for d in details if d["parser_status"] == "parsed"]
    n_correct_parsed = sum(d["correct"] for d in parsed)
    status_counts = {
        status: sum(d["parser_status"] == status for d in details)
        for status in ("parsed", "unparsed", "ambiguous")
    }
    return {
        "n_items": n,
        "parse_status_counts": status_counts,
        "parse_rate": (len(parsed) / n) if n else None,
        "all_items": {
            "policy": "format_failure_counts_as_task_failure",
            "accuracy": (n_correct_parsed / n) if n else None,
        },
        "parsed_only": {
            "policy": "sensitivity_analysis_excluding_unparsed_and_ambiguous",
            "n_parsed": len(parsed),
            "accuracy": (n_correct_parsed / len(parsed)) if parsed else None,
        },
    }


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, items, batch_size, max_tokens, device):
    """SECONDARY format readout: greedy short generation + audited parse.

    Returns (analysis summary, per-item detail); each detail row is
    self-contained: row id, generated text, parsed letter, parser status,
    candidates, target, and correctness (None when not parsed).
    """
    details = []
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
            pred, status, candidates = parse_letter(completion)
            is_correct = (pred == item["answer"]) if pred is not None else None
            details.append({
                "row_index": item["row_index"],
                "completion": completion,
                "predicted": LETTERS[pred] if pred is not None else None,
                "parser_status": status,
                "parser_candidates": list(candidates),
                "target": LETTERS[item["answer"]],
                "correct": is_correct,
            })
    tokenizer.padding_side = "right"
    pred_counts = {letter: sum(d["predicted"] == letter for d in details)
                   for letter in LETTERS}
    for status in ("unparsed", "ambiguous"):
        pred_counts[status] = sum(d["parser_status"] == status for d in details)
    log.info("prediction distribution: %s", pred_counts)
    return summarize_generation_details(details), details


def subject_cluster_bootstrap_delta(base_correct, cond_correct, subjects,
                                    seed=0, n_boot=10000):
    """Paired accuracy-delta CI resampling SUBJECT clusters, not items (Q5).

    Sampling is subject-stratified, so treating items as independent
    understates uncertainty; clusters are resampled with replacement and the
    mean per-item delta recomputed over the concatenated resample.
    """
    base = np.asarray(base_correct, dtype=float)
    cond = np.asarray(cond_correct, dtype=float)
    subj = np.asarray(subjects)
    if not (len(base) == len(cond) == len(subj)):
        raise ValueError("per-item arrays and subjects must align")
    deltas = cond - base
    unique = sorted(set(subj.tolist()))
    groups = [deltas[subj == s] for s in unique]
    per_subject = {
        s: {"mean_delta": float(deltas[subj == s].mean()),
            "sign": ("positive" if deltas[subj == s].mean() > 0 else
                     "negative" if deltas[subj == s].mean() < 0 else "zero"),
            "n_items": int((subj == s).sum())}
        for s in unique}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(float(np.concatenate([groups[i] for i in pick]).mean()))
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(means, 2.5)),
                 float(np.percentile(means, 97.5))],
        "method": "subject_cluster_bootstrap",
        "n_subjects": len(unique),
        "n_boot": n_boot,
        "per_subject": per_subject,
        "interpretation": (
            "a CI containing zero supports only 'no detected change' on this "
            "benchmark; it is NOT an equivalence/no-cost result (no "
            "equivalence margin was frozen before the run)"
        ),
    }


def exact_text_slice(nonempty_rows, *, char_start=200_000, char_end=400_000):
    """Return the frozen Python character slice, rejecting short corpora."""
    if not (0 <= char_start < char_end):
        raise ValueError("text slice requires 0 <= start < end")
    joined = "\n".join(t for t in nonempty_rows if t.strip())
    if len(joined) < char_end:
        raise ValueError(
            f"joined corpus has {len(joined)} characters; exact slice "
            f"[{char_start}:{char_end}] requires at least {char_end}")
    return joined[char_start:char_end]


def verify_dataset_fingerprint(label, observed, expected, *, accepted):
    """Bind accepted dataset loads to an explicit resolved fingerprint."""
    if accepted and not expected:
        raise EvidenceRunError(
            f"accepted mode requires --expected-{label}-fingerprint")
    if accepted and observed != expected:
        raise EvidenceRunError(
            f"{label} dataset fingerprint mismatch: expected {expected!r}, "
            f"observed {observed!r}")


def verify_expected_hash(label, observed, expected, *, accepted):
    """Bind accepted content to a caller-supplied frozen SHA-256."""
    if accepted and not expected:
        raise EvidenceRunError(
            f"accepted mode requires --expected-{label}-sha256")
    if accepted and observed != expected:
        raise EvidenceRunError(
            f"{label} SHA-256 mismatch: expected {expected!r}, "
            f"observed {observed!r}")


def validate_capability_artifact(payload):
    expected = {"baseline", "positive_writers_k2", "suppressors_k4",
                "targeted_k6"}
    if set(payload.get("conditions", {})) != expected:
        raise ValueError("capability artifact lacks the exact four conditions")
    for name, condition in payload["conditions"].items():
        if not condition.get("restoration", {}).get("ok"):
            raise ValueError(f"condition {name} failed restoration")
        if "mmlu_likelihood_per_item_detail" not in condition or "wikitext" not in condition:
            raise ValueError(f"condition {name} is incomplete")


def load_wikitext_slice(*, revision=None, char_start=200_000,
                        char_end=400_000):
    """Load and fingerprint exact WikiText bytes before model work."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                      revision=revision)
    text = exact_text_slice(ds["text"], char_start=char_start,
                            char_end=char_end)
    info = {
        "name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
        "split": "test", "revision": revision,
        "fingerprint": getattr(ds, "_fingerprint", None),
        "char_start": char_start, "char_end": char_end,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return text, info


@torch.no_grad()
def wikitext_nll(model, tokenizer, device, n_chars=200_000, window=1024,
                 revision=None, *, char_start=200_000, text=None,
                 dataset_info=None):
    """Non-overlapping-window NLL with per-window losses persisted."""
    if text is None:
        text, dataset_info = load_wikitext_slice(
            revision=revision, char_start=char_start,
            char_end=char_start + n_chars)
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
            "dataset": dataset_info,
            "n_chars": len(text), "window": window,
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
    parser.add_argument("--protocol", default=None,
                        help="named frozen protocol required in accepted mode")
    parser.add_argument("--mmlu-revision", default=None,
                        help="cais/mmlu dataset revision (accepted reruns must pin this)")
    parser.add_argument("--wikitext-revision", default=None,
                        help="Salesforce/wikitext dataset revision")
    parser.add_argument("--expected-mmlu-fingerprint", default=None)
    parser.add_argument("--expected-wikitext-fingerprint", default=None)
    parser.add_argument("--expected-wikitext-text-sha256", default=None,
                        help="frozen SHA-256 of the exact WikiText character slice")
    parser.add_argument("--wikitext-char-start", type=int, default=200_000)
    parser.add_argument("--wikitext-char-end", type=int, default=400_000)
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
    parser.add_argument("--run-mode", choices=("accepted", "exploratory"),
                        default="exploratory",
                        help="accepted = evidence-eligible (immutable pinned "
                             "model/tokenizer/dataset revisions verified "
                             "before any load, fresh output, clean source); "
                             "exploratory artifacts are persisted as "
                             "ineligible for confirmatory evidence")
    parser.add_argument("--allow-direction-mismatch", action="store_true",
                        help="EXPLORATORY-ONLY override for a direction/"
                             "registry mismatch; persisted as non-confirmatory")
    parser.add_argument("--allowed-dirty", action="append", default=[],
                        help="explicit source-binding exclusion rule (glob) "
                             "for accepted mode; persisted in the artifact")
    parser.add_argument("--skip-generation-readout", action="store_true",
                        help="skip the secondary greedy-generation readout "
                             "(primary likelihood readout always runs)")
    parser.add_argument("--out", default="results/capability_eval_gemma2_9b_it")
    args = parser.parse_args()

    frozen_protocol = None
    if args.run_mode == "accepted":
        try:
            frozen_protocol = validate_gate0c_capability_protocol(
                protocol=args.protocol, model=args.model,
                n_mmlu=args.n_mmlu, seed=args.seed,
                wikitext_char_start=args.wikitext_char_start,
                wikitext_char_end=args.wikitext_char_end,
                skip_generation_readout=args.skip_generation_readout,
            )
        except EvidenceRunError as exc:
            parser.error(str(exc))

    try:
        resolution = resolve_component_sets(
            roles=("positive_writers", "suppressors"),
            explicit={"positive_writers": args.writers,
                      "suppressors": args.suppressors},
            set_key=args.component_set,
            model=args.model,
            direction_path=args.direction,
            run_mode=args.run_mode,
            allow_direction_mismatch=args.allow_direction_mismatch,
        )
    except ComponentSetError as exc:
        parser.error(str(exc))

    if args.run_mode == "accepted":
        try:
            frozen_protocol["binding"] = validate_gate0c_capability_binding(
                component_set=args.component_set,
                direction_path=args.direction,
                resolved_sets=resolution["sets"])
        except EvidenceRunError as exc:
            parser.error(str(exc))

    out = Path(args.out)
    try:
        contract = evaluate_run_contract(
            args.run_mode,
            revisions={
                "model": {
                    "requested": args.revision,
                    "resolution": resolve_hf_commit(
                        args.model, revision=args.revision or "main"),
                },
                "tokenizer": {
                    "requested": args.tokenizer_revision or args.revision,
                    "resolution": resolve_hf_commit(
                        args.model,
                        revision=(args.tokenizer_revision or args.revision
                                  or "main")),
                },
                # dataset revisions are pin-format validated (no local hub
                # cache resolution exists for datasets); accepted mode still
                # requires immutable pins for both
                "mmlu_dataset": {"requested": args.mmlu_revision},
                "wikitext_dataset": {"requested": args.wikitext_revision},
            },
            output_paths=(out / "capability_eval.json",),
            source_rules=args.allowed_dirty,
            input_paths=(args.direction,),
        )
    except EvidenceRunError as exc:
        parser.error(str(exc))
    if contract.get("warning"):
        log.warning("%s", contract["warning"])
    writers = resolution["sets"]["positive_writers"]
    suppressors = resolution["sets"]["suppressors"]
    if args.targeted:
        targeted, targeted_origin = args.targeted, "explicit"
    else:
        targeted = f"{writers},{suppressors}"
        targeted_origin = "derived:positive_writers+suppressors"
    if (args.run_mode == "accepted"
            and targeted != f"{writers},{suppressors}"):
        parser.error("accepted Gate-0C capability protocol requires targeted "
                     "= positive_writers + suppressors exactly")
    resolution["sets"]["targeted"] = targeted
    resolution["source"]["origins"]["targeted"] = targeted_origin
    log.info("component sets resolved: %s", resolution)

    out.mkdir(parents=True, exist_ok=True)

    # Resolve and fingerprint both datasets before importing/loading the
    # model. A revision string alone does not bind the realized dataset.
    items, dataset_info, sampling_info = sample_mmlu(
        args.n_mmlu, args.seed, revision=args.mmlu_revision
    )
    wiki_text, wiki_dataset_info = load_wikitext_slice(
        revision=args.wikitext_revision,
        char_start=args.wikitext_char_start,
        char_end=args.wikitext_char_end,
    )
    try:
        verify_dataset_fingerprint(
            "mmlu", dataset_info.get("fingerprint"),
            args.expected_mmlu_fingerprint,
            accepted=args.run_mode == "accepted")
        verify_dataset_fingerprint(
            "wikitext", wiki_dataset_info.get("fingerprint"),
            args.expected_wikitext_fingerprint,
            accepted=args.run_mode == "accepted")
        verify_expected_hash(
            "wikitext-text", wiki_dataset_info.get("text_sha256"),
            args.expected_wikitext_text_sha256,
            accepted=args.run_mode == "accepted")
    except EvidenceRunError as exc:
        parser.error(str(exc))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "frozen_protocol": frozen_protocol,
        "run_contract": contract,
        "mmlu_primary_readout": MMLU_PRIMARY_READOUT,
        "mmlu_prompt_instruction": GATE0C_MMLU_INSTRUCTION,
        "component_sets": resolution,
        "mmlu_dataset": dataset_info,
        "expected_dataset_fingerprints": {
            "mmlu": args.expected_mmlu_fingerprint,
            "wikitext": args.expected_wikitext_fingerprint,
        },
        "expected_wikitext_text_sha256": args.expected_wikitext_text_sha256,
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
    subjects = [it["subject"] for it in items]
    for name, components in conditions.items():
        snapshots = snapshot_weights(model, components) if components else {}
        try:
            edits = [orthogonalize_component(model, c, direction) for c in components]
            lik_acc, lik_detail = mmlu_likelihood_eval(model, tokenizer,
                                                       items, device)
            if args.skip_generation_readout:
                gen_analysis, gen_detail = None, None
            else:
                gen_analysis, gen_detail = mmlu_accuracy(
                    model, tokenizer, items, args.batch_size,
                    args.max_tokens, device)
            wiki = wikitext_nll(
                model, tokenizer, device, revision=args.wikitext_revision,
                char_start=args.wikitext_char_start, text=wiki_text,
                dataset_info=wiki_dataset_info)
        finally:
            if snapshots:
                restore_weights(snapshots)
        # QA Q5: verify every touched tensor against its pre-edit snapshot;
        # a mismatch raises and no accepted artifact is finalized.
        restoration = assert_restored(snapshots) if snapshots else {
            "ok": True, "n_tensors": 0, "tensors": []}
        results["conditions"][name] = {
            "components": [c["name"] for c in components],
            "edits": edits,
            "restoration": restoration,
            "mmlu_likelihood_accuracy": lik_acc,
            "mmlu_likelihood_per_item": [d["correct"] for d in lik_detail],
            "mmlu_likelihood_per_item_detail": lik_detail,
            "mmlu_generation_analysis": gen_analysis,
            "mmlu_generation_per_item_detail": gen_detail,
            # legacy keys now explicitly track the PRIMARY (likelihood) readout
            "mmlu_accuracy": lik_acc,
            "mmlu_per_item": [d["correct"] for d in lik_detail],
            "wikitext": wiki,
        }
        log.info("%s: MMLU(likelihood) %.4f | ppl %.4f", name, lik_acc,
                 wiki["perplexity"])

    base = results["conditions"]["baseline"]
    for name, cond in results["conditions"].items():
        if name == "baseline":
            continue
        cond["mmlu_delta_vs_baseline"] = subject_cluster_bootstrap_delta(
            base["mmlu_likelihood_per_item"],
            cond["mmlu_likelihood_per_item"],
            subjects, seed=args.seed,
        )
        if not args.skip_generation_readout:
            parsed_pair = [
                (b_row["correct"], c_row["correct"], subj)
                for b_row, c_row, subj in zip(
                    base["mmlu_generation_per_item_detail"],
                    cond["mmlu_generation_per_item_detail"], subjects)
                if b_row["correct"] is not None and c_row["correct"] is not None
            ]
            if parsed_pair:
                cond["mmlu_generation_delta_parsed_only"] = (
                    subject_cluster_bootstrap_delta(
                        [p[0] for p in parsed_pair],
                        [p[1] for p in parsed_pair],
                        [p[2] for p in parsed_pair], seed=args.seed))
        cond["ppl_ratio_vs_baseline"] = cond["wikitext"]["perplexity"] / base["wikitext"]["perplexity"]

    revalidate_run_contract(
        contract, source_rules=args.allowed_dirty, input_paths=(args.direction,))
    atomic_write_json(out / "capability_eval.json", results,
                      require_fresh=True, validate_fn=validate_capability_artifact)
    log.info("wrote %s", out / "capability_eval.json")


if __name__ == "__main__":
    main()
