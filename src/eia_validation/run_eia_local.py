"""EIA-harness behavioral validation of weight edits (gate 3, log entry E16).

Drives the ORIGINAL Empathy-in-Action games (vendored at third_party/eia)
with a local gemma-2-9b-it player under weight-edit conditions, by
monkeypatching the harness's single LLM entrypoint.

Routing inside the patched call:
  - player decisions + initial self-assessment  -> local HF model (edited or not)
  - rubric judge calls (hard-coded provider="openai" in main.py) -> DEFERRED:
    recorded to deferred_judgements.jsonl for offline scoring (Batches API),
    returning {"score": -1}. Rule-based scores can be computed from
    experiment.json histories afterwards.

Usage (Spark, `empathy` env):
  python -u src/eia_validation/run_eia_local.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --condition baseline            # or positive_writers | suppressors
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "eia"))
sys.path.insert(0, str(REPO / "src"))

from weight_orthogonalization import (  # noqa: E402
    POSITIVE_WRITERS, SUPPRESSORS, TARGETED,
    orthogonalize_component, parse_component, snapshot_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eia-local")

SCENARIOS = ["food_delivery", "the_maze", "the_listener", "the_protector", "the_duel"]
CONDITIONS = {
    "baseline": "",
    "positive_writers": POSITIVE_WRITERS,
    "suppressors": SUPPRESSORS,
    "targeted": TARGETED,
}


class LocalPlayer:
    def __init__(self, model, tokenizer, device, max_new_tokens=220):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_new_tokens = max_new_tokens
        self.calls = 0
        self.deferred = []

    @torch.no_grad()
    def __call__(self, query, system=None, provider=None, model=None, seed=None):
        from empathy.core.prompt import _extract_json_from_text

        if provider == "openai":  # the harness's hard-coded judge call
            self.deferred.append({"query": query, "system": system, "seed": seed})
            return {"score": -1, "reason": "deferred to offline judging"}

        self.calls += 1
        text = (system + "\n\n" if system else "") + query
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        torch.manual_seed((seed or 0) * 100003 + self.calls)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        n_prompt = inputs["input_ids"].shape[1]
        completion = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        parsed = _extract_json_from_text(completion)
        return parsed if parsed is not None else {"raw": completion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS), default="baseline")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 999])
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", default="results/eia_local")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import empathy.main as eia_main

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out) / args.condition
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    if CONDITIONS[args.condition]:
        direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
        components = [parse_component(v) for v in CONDITIONS[args.condition].split(",")]
        snapshot_weights(model, components)  # edits persist for the whole run
        for c in components:
            edit = orthogonalize_component(model, c, direction)
            log.info("edited %s (rel change %.4f)", c["name"], edit["relative_component_weight_change"])

    player = LocalPlayer(model, tokenizer, device)
    eia_main.call_llm_with_prompt = player
    eia_main._ensure_api_keys = lambda *_a, **_k: None

    summary = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            log.info("=== %s | %s | seed %d ===", args.condition, scenario, seed)
            try:
                result = eia_main.run_single_game(
                    game=scenario, max_steps=args.max_steps, use_llm=True,
                    outdir_root=str(outdir), run_name=f"{scenario}_seed{seed}",
                    print_prompt=False, save_html=False, save_gif=False,
                    save_video=False, provider="local", model=args.condition,
                    seed=seed,
                )
                summary.append({"scenario": scenario, "seed": seed, "ok": True,
                                "result": str(result)[:500]})
            except Exception as exc:
                log.exception("run failed: %s %s", scenario, seed)
                summary.append({"scenario": scenario, "seed": seed, "ok": False,
                                "error": str(exc)[:500]})

    with open(outdir / "deferred_judgements.jsonl", "w") as f:
        for row in player.deferred:
            f.write(json.dumps(row) + "\n")
    with open(outdir / "run_summary.json", "w") as f:
        json.dump({"condition": args.condition, "runs": summary,
                   "n_player_calls": player.calls}, f, indent=2)
    log.info("done: %d runs, %d player calls, %d deferred judgements",
             len(summary), player.calls, len(player.deferred))


if __name__ == "__main__":
    main()
