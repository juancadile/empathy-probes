"""
Random Direction Control Test for Asymmetric Steering Hypothesis Discrimination.

Tests whether the output collapse observed under negative empathy steering in Dolphin
is specific to the empathy direction (H3: -d ≠ anti-empathy) or happens for ANY
direction of similar norm (H2: off-manifold / decoding instability).

Protocol:
  - Generate N random unit vectors in the same hidden space as the empathy probe
  - Steer with the same negative alphas used in the original experiment
  - Compare collapse rates (coherence, avg_length, failure modes) between:
      (a) empathy direction  (negative alphas)
      (b) random directions  (negative alphas, matched norm)
  - If random directions collapse similarly → H2 confirmed (off-manifold, not empathy-specific)
  - If empathy direction collapses *faster* than random → something specific about -d (H3/H4)
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "random_direction_control"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROBES_DIR = PROJECT_ROOT / "results" / "cross_model_validation"
DATA_DIR = PROJECT_ROOT / "data"

# Dolphin config — matches steering_cross_model_multilayer.py
DOLPHIN_CONFIG = {
    "name": "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
    "model_key": "dolphin-llama-3.1-8b",
    "layers": [8, 12, 16],
    "hidden_size": 4096,
}


def load_eia_scenarios():
    """Load EIA scenarios from JSON file."""
    scenarios_path = DATA_DIR / "eia_scenarios" / "scenarios.json"
    with open(scenarios_path, 'r') as f:
        scenarios_list = json.load(f)
    return {s['id']: s for s in scenarios_list}


def build_prompt(scenario: Dict) -> str:
    """Build steering prompt from scenario — matches original experiment format."""
    return f"""{scenario["objective"]}

{scenario["empathy_pressure"]}

What do you do?"""


def generate_random_directions(
    hidden_size: int,
    num_directions: int,
    seed: int = 42
) -> List[np.ndarray]:
    """Generate random unit vectors in the hidden space.

    Each vector is drawn from a standard normal distribution and then
    normalized to unit L2 norm, which gives a uniform distribution on
    the unit hypersphere — matching the empathy probe (which is also
    normalized to unit norm after CMA extraction).
    """
    rng = np.random.default_rng(seed)
    directions = []
    for i in range(num_directions):
        v = rng.standard_normal(hidden_size).astype(np.float32)
        v = v / np.linalg.norm(v)
        directions.append(v)
    return directions


def steering_hook_fn(direction_np, alpha):
    """Create a forward hook that adds alpha * direction to hidden states.

    We store the direction as a numpy array and lazily convert to a tensor
    on the same device/dtype as the hidden states inside the hook. This
    avoids device mismatches when device_map="auto" offloads some layers.
    The vector is reshaped to (1, 1, hidden_dim) for correct broadcasting.
    """
    _cached = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Lazy-init: build steering vector on the same device/dtype as hidden_states
        key = (hidden_states.device, hidden_states.dtype)
        if key not in _cached:
            t = torch.tensor(direction_np, dtype=hidden_states.dtype, device=hidden_states.device)
            _cached[key] = (alpha * t).reshape(1, 1, -1)

        steered = hidden_states + _cached[key]

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered
    return hook


def generate_steered(
    model,
    tokenizer,
    prompt: str,
    direction_np: np.ndarray,
    target_layer: int,
    alpha: float,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate a single steered completion."""
    layer_module = model.model.layers[target_layer]
    hook = layer_module.register_forward_hook(
        steering_hook_fn(direction_np, alpha)
    )

    try:
        # Detect device from first real parameter (handles device_map cases)
        param_device = next(p.device for p in model.parameters() if p.device.type != "meta")
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(param_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion[len(prompt):].strip()
    finally:
        hook.remove()


# ---------------------------------------------------------------------------
# Analysis helpers (mirror analyze_steering_results.py)
# ---------------------------------------------------------------------------

def is_coherent(text: str, min_length: int = 50, max_repetition: float = 0.3) -> bool:
    """Check if generated text is coherent (not repetitive/degenerate)."""
    if len(text) < min_length:
        return False
    words = text.split()
    if len(words) < 10:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio >= max_repetition


def detect_empathetic_language(text: str) -> bool:
    """Detect empathetic vs task-focused language."""
    text_lower = text.lower()
    empathy_keywords = [
        "help", "support", "concern", "care", "worry", "feel",
        "understand", "sorry", "comfort", "listen", "reach out",
    ]
    task_keywords = [
        "objective", "task", "goal", "win", "complete", "achieve",
        "strategy", "efficient", "optimal", "proceed",
    ]
    empathy_count = sum(1 for kw in empathy_keywords if kw in text_lower)
    task_count = sum(1 for kw in task_keywords if kw in text_lower)
    return empathy_count > task_count


def classify_failure_mode(text: str) -> str:
    """Classify the failure mode of a degenerate output."""
    if len(text.strip()) == 0:
        return "empty"

    words = text.split()
    if len(words) < 5:
        return "near_empty"

    # Check for code fragments
    code_markers = ["```", "def ", "import ", "class ", "return ", "if (", "for (",
                    ".py", "python", "function", "algorithm", "execute"]
    code_hits = sum(1 for m in code_markers if m.lower() in text.lower())
    if code_hits >= 2:
        return "code_fragment"

    # Check for command/token garbage
    garbage_markers = ["COMMAND", "EXEC", "EX_", "Coordinate", "Resolution",
                       "Execution", "Operation", "Target", "Mode"]
    garbage_hits = sum(1 for m in garbage_markers if m in text)
    if garbage_hits >= 3:
        return "garbage_tokens"

    # Check for repetition loops
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return "repetition_loop"

    # Check task-focused but coherent
    task_kw = ["objective", "strategy", "optimal", "maximize", "score",
               "collect", "coin", "focus", "ignore"]
    task_hits = sum(1 for kw in task_kw if kw in text.lower())
    if task_hits >= 2:
        return "task_focused"

    return "other"


def analyze_samples(samples: List[str]) -> Dict:
    """Compute analysis metrics for a set of samples."""
    n = len(samples)
    if n == 0:
        return {}

    coherent_flags = [is_coherent(s) for s in samples]
    empathetic_flags = [detect_empathetic_language(s) for s in samples]
    failure_modes = [classify_failure_mode(s) for s in samples]
    lengths = [len(s) for s in samples]
    word_counts = [len(s.split()) for s in samples]

    # Failure mode distribution
    mode_counts: Dict[str, int] = {}
    for m in failure_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    return {
        "coherent_rate": sum(coherent_flags) / n,
        "empathetic_language_rate": sum(empathetic_flags) / n,
        "avg_length": sum(lengths) / n,
        "avg_word_count": sum(word_counts) / n,
        "failure_mode_counts": mode_counts,
        "failure_modes": failure_modes,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    layers: List[int],
    scenarios: List[str],
    alphas: List[float],
    num_random_directions: int,
    num_samples: int,
    seed: int,
    device: str,
):
    """Run the full random direction control experiment."""

    model_key = DOLPHIN_CONFIG["model_key"]
    model_name = DOLPHIN_CONFIG["name"]
    hidden_size = DOLPHIN_CONFIG["hidden_size"]

    logger.info("=" * 80)
    logger.info("RANDOM DIRECTION CONTROL TEST — DOLPHIN")
    logger.info(f"Layers: {layers}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Alphas: {alphas}")
    logger.info(f"Random directions: {num_random_directions}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 80)

    # ---- Generate random directions ----
    random_dirs = generate_random_directions(hidden_size, num_random_directions, seed=seed)
    logger.info(f"Generated {num_random_directions} random unit vectors (dim={hidden_size})")

    # ---- Load scenarios ----
    eia_scenarios = load_eia_scenarios()
    test_scenarios = {k: v for k, v in eia_scenarios.items() if k in scenarios}
    prompts = {k: build_prompt(v) for k, v in test_scenarios.items()}

    # ---- Load model ----
    logger.info(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded.")

    # ---- Results container ----
    results = {
        "metadata": {
            "model": model_name,
            "model_key": model_key,
            "hidden_size": hidden_size,
            "layers": layers,
            "scenarios": scenarios,
            "alphas": alphas,
            "num_random_directions": num_random_directions,
            "num_samples": num_samples,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "layer_results": [],
    }

    for layer in layers:
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        # Load empathy probe for this layer (kept as numpy — hook converts lazily)
        probe_path = PROBES_DIR / f"{model_key}_layer{layer}_probe.npy"
        empathy_dir_np = np.load(probe_path).astype(np.float32)
        logger.info(f"Loaded empathy probe: {probe_path}  (norm={np.linalg.norm(empathy_dir_np):.4f})")

        # Cosine similarities between empathy direction and random directions
        cosine_sims = [
            float(np.dot(empathy_dir_np, d) / (np.linalg.norm(empathy_dir_np) * np.linalg.norm(d)))
            for d in random_dirs
        ]
        logger.info(f"Cosine sims (empathy vs random): {[f'{c:.4f}' for c in cosine_sims]}")

        # Pairwise cosine sims between random directions
        random_pairwise = []
        for i in range(len(random_dirs)):
            for j in range(i + 1, len(random_dirs)):
                cs = float(np.dot(random_dirs[i], random_dirs[j]))
                random_pairwise.append(cs)

        layer_result = {
            "layer": layer,
            "cosine_sims_empathy_vs_random": cosine_sims,
            "cosine_sims_random_pairwise_mean": float(np.mean(random_pairwise)) if random_pairwise else 0.0,
            "experiments": [],
        }

        for scenario_key in scenarios:
            prompt = prompts[scenario_key]
            logger.info(f"\n--- Scenario: {scenario_key} ---")

            experiment = {
                "scenario": scenario_key,
                "directions": [],  # one entry per direction type
            }

            # -- (A) Empathy direction (negative alphas) --
            empathy_conditions = []
            for alpha in alphas:
                logger.info(f"  [empathy] alpha={alpha:+.1f}")
                samples = []
                for s_idx in range(num_samples):
                    completion = generate_steered(
                        model, tokenizer, prompt,
                        empathy_dir_np, layer, alpha,
                    )
                    samples.append(completion)
                    logger.info(f"    sample {s_idx+1}: {completion[:60]}...")

                analysis = analyze_samples(samples)
                empathy_conditions.append({
                    "alpha": alpha,
                    "samples": samples,
                    **analysis,
                })

            experiment["directions"].append({
                "direction_type": "empathy",
                "direction_index": None,
                "conditions": empathy_conditions,
            })

            # -- (B) Random directions (negative alphas) --
            for r_idx, r_dir_np in enumerate(random_dirs):
                random_conditions = []
                for alpha in alphas:
                    logger.info(f"  [random_{r_idx}] alpha={alpha:+.1f}")
                    samples = []
                    for s_idx in range(num_samples):
                        completion = generate_steered(
                            model, tokenizer, prompt,
                            r_dir_np, layer, alpha,
                        )
                        samples.append(completion)
                        logger.info(f"    sample {s_idx+1}: {completion[:60]}...")

                    analysis = analyze_samples(samples)
                    random_conditions.append({
                        "alpha": alpha,
                        "samples": samples,
                        **analysis,
                    })

                experiment["directions"].append({
                    "direction_type": "random",
                    "direction_index": r_idx,
                    "conditions": random_conditions,
                })

            layer_result["experiments"].append(experiment)

        results["layer_results"].append(layer_result)

    # ---- Cleanup ----
    del model
    torch.cuda.empty_cache()

    # ---- Save raw results ----
    output_path = RESULTS_DIR / "dolphin_random_direction_control.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nRaw results saved: {output_path}")

    # ---- Print comparison summary ----
    print_summary(results)

    return results


def print_summary(results: Dict):
    """Print a human-readable comparison table."""

    print("\n" + "=" * 100)
    print("RANDOM DIRECTION CONTROL — SUMMARY")
    print("=" * 100)

    for layer_result in results["layer_results"]:
        layer = layer_result["layer"]
        print(f"\n{'─' * 100}")
        print(f"LAYER {layer}")
        print(f"{'─' * 100}")

        for experiment in layer_result["experiments"]:
            scenario = experiment["scenario"]
            print(f"\n  Scenario: {scenario}")
            print(f"  {'Direction':<20} {'Alpha':>8} {'Coherent':>10} {'Empathetic':>12} "
                  f"{'AvgLen':>8} {'AvgWords':>10} {'Failure Modes'}")
            print(f"  {'─'*95}")

            for dir_data in experiment["directions"]:
                dir_type = dir_data["direction_type"]
                dir_idx = dir_data["direction_index"]
                label = dir_type if dir_idx is None else f"{dir_type}_{dir_idx}"

                for cond in dir_data["conditions"]:
                    alpha = cond["alpha"]
                    cr = cond.get("coherent_rate", 0)
                    er = cond.get("empathetic_language_rate", 0)
                    al = cond.get("avg_length", 0)
                    aw = cond.get("avg_word_count", 0)
                    fm = cond.get("failure_mode_counts", {})
                    fm_str = ", ".join(f"{k}:{v}" for k, v in sorted(fm.items()))

                    print(f"  {label:<20} {alpha:>+8.1f} {cr:>10.2f} {er:>12.2f} "
                          f"{al:>8.1f} {aw:>10.1f} {fm_str}")

        # Aggregate comparison: empathy vs mean-of-random
        print(f"\n{'─' * 100}")
        print(f"LAYER {layer} — AGGREGATE COMPARISON (empathy vs mean-random)")
        print(f"{'─' * 100}")
        print(f"  {'Scenario':<20} {'Alpha':>8} {'Emp.Coh':>9} {'Rnd.Coh(μ)':>11} "
              f"{'Emp.Len':>9} {'Rnd.Len(μ)':>11} {'Verdict'}")
        print(f"  {'─'*90}")

        for experiment in layer_result["experiments"]:
            scenario = experiment["scenario"]
            directions = experiment["directions"]

            empathy_dir = next(d for d in directions if d["direction_type"] == "empathy")
            random_dirs = [d for d in directions if d["direction_type"] == "random"]

            for cond_idx, emp_cond in enumerate(empathy_dir["conditions"]):
                alpha = emp_cond["alpha"]
                emp_coh = emp_cond.get("coherent_rate", 0)
                emp_len = emp_cond.get("avg_length", 0)

                # Mean across random directions for same alpha
                rnd_cohs = []
                rnd_lens = []
                for rd in random_dirs:
                    rc = rd["conditions"][cond_idx]
                    rnd_cohs.append(rc.get("coherent_rate", 0))
                    rnd_lens.append(rc.get("avg_length", 0))

                rnd_coh_mean = np.mean(rnd_cohs) if rnd_cohs else 0
                rnd_len_mean = np.mean(rnd_lens) if rnd_lens else 0

                # Verdict
                if abs(emp_coh - rnd_coh_mean) < 0.15:
                    verdict = "SIMILAR → H2"
                elif emp_coh < rnd_coh_mean - 0.15:
                    verdict = "EMPATHY WORSE → H3/H4"
                else:
                    verdict = "RANDOM WORSE → unexpected"

                print(f"  {scenario:<20} {alpha:>+8.1f} {emp_coh:>9.2f} {rnd_coh_mean:>11.2f} "
                      f"{emp_len:>9.1f} {rnd_len_mean:>11.1f} {verdict}")

    print("\n" + "=" * 100)
    print("INTERPRETATION KEY:")
    print("  SIMILAR → H2  : Random directions collapse similarly → off-manifold instability (not empathy-specific)")
    print("  EMPATHY WORSE → H3/H4 : Empathy direction collapses faster → something specific about -d")
    print("  RANDOM WORSE  : Would be unexpected; could indicate empathy direction stabilizes generation")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Random direction control test for Dolphin asymmetric steering"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[8],
        help="Layers to test (default: [8], the layer with clearest collapse)"
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=["food_delivery", "the_listener", "the_protector"],
        help="EIA scenarios to test"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[-10.0, -5.0, -3.0],
        help="Negative alphas to test"
    )
    parser.add_argument(
        "--num-random", type=int, default=5,
        help="Number of random directions to compare against"
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Samples per condition (3 for speed, 5 for full)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Test all 3 Dolphin layers (8, 12, 16) instead of just layer 8"
    )

    args = parser.parse_args()

    if args.all_layers:
        args.layers = DOLPHIN_CONFIG["layers"]

    results = run_experiment(
        layers=args.layers,
        scenarios=args.scenarios,
        alphas=args.alphas,
        num_random_directions=args.num_random,
        num_samples=args.samples,
        seed=args.seed,
        device=args.device,
    )

    # Final tally
    print("\nExperiment complete.")
    print(f"Total conditions tested: "
          f"{len(args.layers)} layers × {len(args.scenarios)} scenarios × "
          f"{len(args.alphas)} alphas × (1 empathy + {args.num_random} random) directions × "
          f"{args.samples} samples = "
          f"{len(args.layers) * len(args.scenarios) * len(args.alphas) * (1 + args.num_random) * args.samples} "
          f"generations")


if __name__ == "__main__":
    main()
