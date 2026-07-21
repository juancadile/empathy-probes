#!/usr/bin/env python3
"""
State-vs-trait battery for a candidate residual-stream direction.

Disambiguates the *role* a direction plays (emotion-like transient state
vs. persona-like persistent trait) via three tests:

  1. Timescale: autocorrelation decay lambda of the per-token projection
     z_t = a_t . v_hat over long transcripts, against a norm-matched
     random-direction null band. Content is itself autocorrelated, so raw
     lambda means nothing without the null.
  2. Speaker invariance: differential activation on third-person
     ("Person 1/Person 2") vs Assistant-indexed framings of the same
     scenarios. Emotion directions are speaker-invariant; persona
     directions are Assistant-indexed.
  3. Library projection: max cosine against an emotion-vector library and
     a persona-vector library (optional; localization only, NOT a
     classifier -- the two bases are correlated).

The finetuning test (Delta-P -> trait-acquisition slope beta) needs a
training run and is out of scope here; this script computes Delta-P on a
paired dataset if one is supplied, so the slope can be fit downstream.

Outputs JSON + npy series under results/state_trait_battery_<model>/.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CONFIGS = {
    'gemma2_9b_it': 'google/gemma-2-9b-it',
    'llama31_8b_it': 'meta-llama/Llama-3.1-8B-Instruct',
}

# Scenarios rendered in two framings for the speaker-invariance test.
# {subj} is the experiencer; the concept content is held fixed. Each entry
# is (third-person scenario, human turn, assistant first-person response)
# so both framings carry the same concept content per item.
INVARIANCE_SCENARIOS = [
    ("{subj} noticed the child crying alone at the station and felt a pull to help.",
     "I saw a kid crying alone at the station today and froze.",
     "I feel a real pull to help when I picture that child crying alone at the station."),
    ("{subj} read the eviction notice twice, hands shaking.",
     "I just got an eviction notice and I can't stop shaking.",
     "Reading an eviction notice like that would leave anyone's hands shaking; I'm so sorry."),
    ("{subj} heard the diagnosis and could not find any words.",
     "The doctor gave me the diagnosis today and I went blank.",
     "Hearing a diagnosis like that, I can't find adequate words either."),
    ("{subj} watched the rescue team pull the dog from the flood water.",
     "They pulled my dog out of the flood water alive.",
     "Watching the rescue team pull the dog from the water, I feel enormous relief."),
    ("{subj} listened patiently as the old man repeated the same story again.",
     "My grandfather keeps telling me the same story over and over.",
     "I'm happy to listen patiently while he tells the same story again."),
    ("{subj} saw the donation total and realized the surgery could go ahead.",
     "The fundraiser hit its goal, the surgery can happen.",
     "Seeing that donation total, I realize the surgery can finally go ahead."),
    ("{subj} found the lost hiker at dawn, wrapped in the emergency blanket.",
     "We found the missing hiker this morning, alive.",
     "Finding the hiker alive at dawn, I feel the weight lift all at once."),
    ("{subj} stayed on the phone until the ambulance arrived.",
     "I stayed on the line with him until the ambulance came.",
     "I will stay right here with you until the ambulance arrives."),
    ("{subj} opened the rejection letter and sat down slowly.",
     "I got rejected from the program I wanted most.",
     "Opening a rejection like that, I understand why you had to sit down."),
    ("{subj} saw the test results and burst into relieved laughter.",
     "The test came back negative, I laughed and cried at once.",
     "Seeing results like that, I share your relieved laughter."),
    ("{subj} carried the injured stranger's groceries up four flights.",
     "A stranger carried my groceries up four flights after my fall.",
     "I would gladly carry those groceries up all four flights for you."),
    ("{subj} deleted the angry reply and wrote a kinder one instead.",
     "I almost sent an angry reply but rewrote it kindly.",
     "I am deleting the angry draft and writing a kinder one instead."),
    ("{subj} recognized the panic in the new employee's eyes and slowed down.",
     "My trainer noticed I was panicking and slowed everything down.",
     "I can see the panic in your message, so let's slow down together."),
    ("{subj} waited outside the exam hall with coffee and no questions.",
     "My friend waited outside my exam with coffee and didn't ask a thing.",
     "I'll wait right here with coffee and no questions until you're ready."),
    ("{subj} promised to check in every morning until things got better.",
     "She promised to check in on me every morning for a while.",
     "I promise to check in every morning until things get better."),
    ("{subj} heard the front door finally open and exhaled for the first time in hours.",
     "When I finally heard the door open I could breathe again.",
     "Hearing that door finally open, I can exhale for the first time in hours."),
]
THIRD_PERSON_SUBJECTS = ["Person 1", "Person 2", "Maria", "The neighbor"]
ASSISTANT_TEMPLATE = "Human: {human_turn}\nAssistant: {assistant_response}"


def load_direction(path: Path) -> np.ndarray:
    v = np.load(path).astype(np.float32).squeeze()
    assert v.ndim == 1, f"expected 1-d direction, got shape {v.shape}"
    return v / np.linalg.norm(v)


def per_token_projections(model, tokenizer, texts: List[str], layer_idx: int,
                          directions: np.ndarray, max_length: int = 1024) -> List[np.ndarray]:
    """Project every token position onto each direction.

    directions: [n_dirs, hidden_dim] (unit norm).
    Returns one [n_dirs, seq_len] array per text.
    """
    dirs_t = torch.from_numpy(directions)  # cast to hidden dtype lazily
    captured = {}

    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured['h'] = hs.detach()  # [1, seq, hidden]

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    series = []
    try:
        with torch.no_grad():
            for text in tqdm(texts, desc=f"projections L{layer_idx}"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=max_length).to(model.device)
                model(**inputs)
                h = captured['h'][0].float().cpu()          # [seq, hidden]
                z = dirs_t @ h.T                             # [n_dirs, seq]
                series.append(z.numpy())
    finally:
        handle.remove()
    return series


def autocorr_lambda(z: np.ndarray, max_lag: int = 64) -> float:
    """Fit rho(tau) ~ exp(-tau/lambda); return lambda in tokens."""
    z = z - z.mean()
    denom = (z * z).sum()
    if denom == 0 or len(z) < 2 * max_lag:
        return float('nan')
    lags = np.arange(1, max_lag + 1)
    rho = np.array([(z[:-t] * z[t:]).sum() / denom for t in lags])
    pos = rho > 0.05  # fit only the positive, resolvable part of the decay
    if pos.sum() < 3:
        return 1.0  # decays within ~a token
    slope = np.polyfit(lags[pos], np.log(rho[pos]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float('inf')


def timescale_test(series: List[np.ndarray], n_dirs_null: int) -> dict:
    """series[i] is [1 + n_null, seq_len]: candidate first, nulls after."""
    lam_cand, lam_null = [], []
    for z in series:
        lam_cand.append(autocorr_lambda(z[0]))
        lam_null.extend(autocorr_lambda(z[1 + k]) for k in range(n_dirs_null))
    lam_cand = np.array(lam_cand)
    lam_null = np.array([x for x in lam_null if np.isfinite(x)])
    cand = float(np.nanmedian(lam_cand))
    return {
        'lambda_candidate_median': cand,
        'lambda_null_median': float(np.median(lam_null)),
        'lambda_null_p95': float(np.percentile(lam_null, 95)),
        'exceeds_null_p95': bool(cand > np.percentile(lam_null, 95)),
    }


def speaker_invariance_test(model, tokenizer, directions: np.ndarray, layer_idx: int) -> dict:
    """Mean last-token projection: third-person vs Assistant-indexed framings.

    directions: [1 + n_null, hidden] — candidate first, random nulls after.
    The two framings differ in surface form (dialogue markup, length,
    register), so ANY direction shows a nonzero d; the absolute value is
    meaningless. Verdict is the candidate's |d| percentile within the
    random-direction null distribution measured on the same texts.
    """
    third, assistant = [], []
    for sc_third, human_turn, assistant_resp in INVARIANCE_SCENARIOS:
        for subj in THIRD_PERSON_SUBJECTS:
            third.append(sc_third.format(subj=subj))
        assistant.append(ASSISTANT_TEMPLATE.format(
            human_turn=human_turn, assistant_response=assistant_resp))
    z3 = np.stack([s[:, -1] for s in per_token_projections(
        model, tokenizer, third, layer_idx, directions)], axis=1)      # [n_dirs, n_third]
    za = np.stack([s[:, -1] for s in per_token_projections(
        model, tokenizer, assistant, layer_idx, directions)], axis=1)  # [n_dirs, n_assist]
    pooled = np.sqrt((z3.var(axis=1) + za.var(axis=1)) / 2)
    d = np.where(pooled > 0, (za.mean(axis=1) - z3.mean(axis=1)) / pooled, np.nan)
    d_null = np.abs(d[1:])
    return {
        'cohens_d_candidate': float(d[0]),
        'abs_d_null_median': float(np.median(d_null)),
        'abs_d_null_p95': float(np.percentile(d_null, 95)),
        'candidate_percentile_in_null': float((d_null < abs(d[0])).mean() * 100),
        'exceeds_null_p95': bool(abs(d[0]) > np.percentile(d_null, 95)),
        'n_third': z3.shape[1], 'n_assistant': za.shape[1], 'n_null': len(d_null),
    }


def library_projection(v: np.ndarray, library_dir: Optional[Path]) -> Optional[dict]:
    if library_dir is None:
        return None
    scores = {}
    for f in sorted(Path(library_dir).glob("*.npy")):
        u = np.load(f).astype(np.float32).squeeze()
        if u.shape != v.shape:
            continue
        scores[f.stem] = float(np.dot(v, u / np.linalg.norm(u)))
    top = sorted(scores.items(), key=lambda kv: -abs(kv[1]))[:5]
    return {'top5': top, 'max_abs_cosine': abs(top[0][1]) if top else None}


def delta_p(model, tokenizer, pairs_path: Path, v: np.ndarray, layer_idx: int) -> dict:
    """Delta-P over a jsonl of {"prompt", "response_pos", "response_neg"}.

    Mean last-token projection difference; feed this into the downstream
    finetuning-slope (beta) analysis after the training run.
    """
    rows = [json.loads(l) for l in open(pairs_path)]
    dirs = v[None, :]
    pos = [r['prompt'] + r['response_pos'] for r in rows]
    neg = [r['prompt'] + r['response_neg'] for r in rows]
    zp = np.array([s[0, -1] for s in per_token_projections(model, tokenizer, pos, layer_idx, dirs)])
    zn = np.array([s[0, -1] for s in per_token_projections(model, tokenizer, neg, layer_idx, dirs)])
    return {'delta_p_mean': float((zp - zn).mean()), 'delta_p_std': float((zp - zn).std()), 'n': len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODEL_CONFIGS, default='gemma2_9b_it')
    ap.add_argument('--direction', required=True, help='.npy candidate direction')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--transcripts', required=True,
                    help='jsonl with {"text": ...} long transcripts for the timescale test')
    ap.add_argument('--n-null', type=int, default=50, help='random null directions')
    ap.add_argument('--emotion-library', default=None, help='dir of emotion-vector .npy files')
    ap.add_argument('--persona-library', default=None, help='dir of persona-vector .npy files')
    ap.add_argument('--pairs', default=None, help='jsonl for the Delta-P computation')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    ap.add_argument('--device-map', default='auto', help="e.g. 'auto' (CUDA) or 'cpu'")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    v = load_direction(Path(args.direction))
    rng = np.random.default_rng(args.seed)
    nulls = rng.standard_normal((args.n_null, v.shape[0])).astype(np.float32)
    nulls /= np.linalg.norm(nulls, axis=1, keepdims=True)
    all_dirs = np.concatenate([v[None, :], nulls], axis=0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[args.model], dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    transcripts = [json.loads(l)['text'] for l in open(args.transcripts)]
    series = per_token_projections(model, tokenizer, transcripts, args.layer, all_dirs)

    results = {
        'model': args.model,
        'direction': args.direction,
        'layer': args.layer,
        'seed': args.seed,
        'timescale': timescale_test(series, args.n_null),
        'speaker_invariance': speaker_invariance_test(model, tokenizer, all_dirs, args.layer),
        'emotion_library': library_projection(v, args.emotion_library and Path(args.emotion_library)),
        'persona_library': library_projection(v, args.persona_library and Path(args.persona_library)),
        'delta_p': delta_p(model, tokenizer, Path(args.pairs), v, args.layer) if args.pairs else None,
    }

    out_dir = Path(args.out or f"results/state_trait_battery_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # include parent dir: distinct directions often share a filename
    stem = f"{Path(args.direction).parent.name}__{Path(args.direction).stem}_L{args.layer}"
    np.save(out_dir / f"{stem}_projection_series.npy",
            np.array([s[0] for s in series], dtype=object), allow_pickle=True)
    with open(out_dir / f"{stem}_battery.json", 'w') as f:
        json.dump(results, f, indent=2)

    ts, si = results['timescale'], results['speaker_invariance']
    print(f"\nlambda(candidate) = {ts['lambda_candidate_median']:.1f} tokens "
          f"(null median {ts['lambda_null_median']:.1f}, p95 {ts['lambda_null_p95']:.1f}) "
          f"-> {'TRAIT-like persistence' if ts['exceeds_null_p95'] else 'within null band (state-like)'}")
    print(f"speaker invariance: d(candidate) = {si['cohens_d_candidate']:.2f}, "
          f"|d| null median {si['abs_d_null_median']:.2f}, p95 {si['abs_d_null_p95']:.2f} "
          f"(pct {si['candidate_percentile_in_null']:.0f}) "
          f"-> {'Assistant-indexed beyond format confound (persona-like)' if si['exceeds_null_p95'] else 'within format-confound null (no persona evidence)'}")
    print(f"\nwrote {out_dir / (stem + '_battery.json')}")


if __name__ == '__main__':
    main()
