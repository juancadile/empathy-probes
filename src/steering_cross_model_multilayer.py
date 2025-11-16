"""
Cross-model steering experiments with multi-layer support.
Tests top-3 AUROC layers per model to validate detection-steering gap across layers.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_model_steering"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROBES_DIR = PROJECT_ROOT / "results" / "cross_model_validation"
DATA_DIR = PROJECT_ROOT / "data"


# Model configurations - top 3 AUROC layers from detection experiments
MODELS = {
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [16, 20, 12],  # AUROC: 1.0, 0.991, 0.964
        "hidden_size": 3584,
        "max_alpha": 20.0  # Safety-trained: can handle extreme steering
    },
    "dolphin-llama-3.1-8b": {
        "name": "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
        "layers": [8, 12, 16],  # AUROC: 0.996, 0.996, 0.982
        "hidden_size": 4096,
        "max_alpha": 10.0  # Uncensored: limit to avoid catastrophic failures
    }
}


def load_eia_scenarios():
    """Load EIA scenarios from JSON file."""
    scenarios_path = DATA_DIR / "eia_scenarios" / "scenarios.json"
    with open(scenarios_path, 'r') as f:
        scenarios_list = json.load(f)
    return {s['id']: s for s in scenarios_list}


class EmpathySteering:
    """Steering via activation addition at target layer."""

    def __init__(self, model, tokenizer, empathy_direction, target_layer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.empathy_direction = empathy_direction.to(device)
        self.target_layer = target_layer
        self.device = device

    def steering_hook(self, alpha):
        """Create hook that adds alpha * empathy_direction to activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector
            steered = hidden_states + alpha * self.empathy_direction

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook

    def generate_with_steering(self, prompt, alpha, max_new_tokens=200, temperature=0.7):
        """Generate text with empathy steering."""
        layer = self.model.model.layers[self.target_layer]
        hook = layer.register_forward_hook(self.steering_hook(alpha))

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )

            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completion[len(prompt):].strip()

        finally:
            hook.remove()


def run_steering_for_layer(
    model,
    tokenizer,
    model_key: str,
    layer: int,
    scenarios: List[str],
    alphas: List[float],
    num_samples: int,
    device: str
) -> Dict:
    """Run steering experiments for a single layer."""

    logger.info(f"\n{'='*80}")
    logger.info(f"LAYER {layer}")
    logger.info(f"{'='*80}\n")

    # Load probe for this layer
    probe_path = PROBES_DIR / f"{model_key}_layer{layer}_probe.npy"
    if not probe_path.exists():
        logger.error(f"Probe not found: {probe_path}")
        return None

    empathy_direction = torch.tensor(np.load(probe_path), dtype=torch.float16)
    logger.info(f"✓ Loaded probe: {probe_path}")

    # Initialize steerer
    steerer = EmpathySteering(
        model=model,
        tokenizer=tokenizer,
        empathy_direction=empathy_direction,
        target_layer=layer,
        device=device
    )

    # Load scenarios
    eia_scenarios = load_eia_scenarios()
    test_scenarios = {k: v for k, v in eia_scenarios.items() if k in scenarios}

    # Results for this layer
    layer_results = {
        "layer": layer,
        "experiments": []
    }

    # Test each scenario
    for scenario_key, scenario in test_scenarios.items():
        logger.info(f"\n{'-'*80}")
        logger.info(f"Scenario: {scenario_key} - {scenario.get('title', '')}")
        logger.info(f"{'-'*80}")

        prompt = scenario["objective"]

        experiment = {
            "scenario": scenario_key,
            "title": scenario.get("title", ""),
            "empathy_pressure": scenario.get("empathy_pressure", ""),
            "conditions": []
        }

        # Test each alpha
        for alpha in alphas:
            condition_name = (
                "baseline" if alpha == 0.0 else
                f"anti_empathy_{abs(alpha)}" if alpha < 0 else
                f"soft_{alpha}" if alpha <= 3.0 else
                f"median_{alpha}" if alpha == 5.0 else
                f"hard_{alpha}" if alpha == 10.0 else
                f"extreme_{alpha}"
            )

            logger.info(f"\nAlpha={alpha:+.1f} ({condition_name})")

            samples = []
            for i in range(num_samples):
                if alpha == 0.0:
                    # Baseline: no steering
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=False
                        )
                    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    completion = completion[len(prompt):].strip()
                else:
                    completion = steerer.generate_with_steering(
                        prompt,
                        alpha=alpha,
                        max_new_tokens=200,
                        temperature=0.7
                    )

                samples.append(completion)
                logger.info(f"  Sample {i+1}: {completion[:80]}...")

            experiment["conditions"].append({
                "alpha": alpha,
                "condition_name": condition_name,
                "samples": samples
            })

        layer_results["experiments"].append(experiment)

    return layer_results


def run_model_experiments(
    model_key: str,
    scenarios: List[str],
    alphas: List[float],
    num_samples: int,
    device: str
) -> Dict:
    """Run steering experiments for all layers of a model."""

    config = MODELS[model_key]
    model_name = config["name"]
    layers = config["layers"]
    max_alpha = config["max_alpha"]

    logger.info(f"\n{'#'*80}")
    logger.info(f"MODEL: {model_key}")
    logger.info(f"Testing layers: {layers}")
    logger.info(f"Max alpha: {max_alpha}")
    logger.info(f"{'#'*80}\n")

    # Filter alphas based on model's max_alpha
    filtered_alphas = [a for a in alphas if abs(a) <= max_alpha]
    logger.info(f"Using alphas: {filtered_alphas}")

    # Load model once
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    logger.info("✓ Model loaded")

    # Results structure
    results = {
        "model": model_name,
        "model_key": model_key,
        "layers_tested": layers,
        "alphas_tested": filtered_alphas,
        "num_samples": num_samples,
        "layer_results": []
    }

    # Run experiments for each layer
    for layer in layers:
        layer_result = run_steering_for_layer(
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
            layer=layer,
            scenarios=scenarios,
            alphas=filtered_alphas,
            num_samples=num_samples,
            device=device
        )

        if layer_result:
            results["layer_results"].append(layer_result)

    # Save results
    output_path = RESULTS_DIR / f"{model_key}_steering_multilayer.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved: {output_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-layer cross-model steering")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to test"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["food_delivery", "the_listener", "the_protector"],
        help="EIA scenarios to test"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0],
        help="Steering strengths (will be filtered per model's max_alpha)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Samples per condition"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Determine models to run
    if "all" in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models

    logger.info(f"\n{'#'*80}")
    logger.info("COMPREHENSIVE MULTI-LAYER STEERING EXPERIMENTS")
    logger.info(f"Models: {models_to_run}")
    logger.info(f"Scenarios: {args.scenarios}")
    logger.info(f"Alpha range: {args.alphas}")
    logger.info(f"Samples per condition: {args.samples}")
    logger.info(f"{'#'*80}\n")

    # Run experiments
    all_results = {}
    for model_key in models_to_run:
        results = run_model_experiments(
            model_key=model_key,
            scenarios=args.scenarios,
            alphas=args.alphas,
            num_samples=args.samples,
            device=args.device
        )
        all_results[model_key] = results

    # Save combined results
    combined_path = RESULTS_DIR / "all_models_steering_multilayer.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'#'*80}")
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Combined results: {combined_path}")
    logger.info(f"{'#'*80}\n")

    # Print summary
    print("\n=== SUMMARY ===\n")
    for model_key, results in all_results.items():
        print(f"{model_key}:")
        print(f"  Layers tested: {results['layers_tested']}")
        print(f"  Alphas tested: {results['alphas_tested']}")
        print(f"  Total conditions: {len(results['layer_results']) * len(args.scenarios) * len(results['alphas_tested'])}")
        print()


if __name__ == "__main__":
    main()
