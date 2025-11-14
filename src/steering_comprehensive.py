"""
Comprehensive steering experiments with updated probes.

Tests steering with:
- Layers 8 and 12 (both showing high AUROC)
- Negative values (anti-empathy steering)
- Soft, median, hard, and extreme positive values
- 5 samples per condition for robustness
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from steering import EmpathySteering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROBES_DIR = RESULTS_DIR / "probes"
DATA_DIR = PROJECT_ROOT / "data"


def load_eia_scenarios():
    """Load EIA scenarios from JSON file and convert to dict."""
    scenarios_path = DATA_DIR / "eia_scenarios" / "scenarios.json"
    with open(scenarios_path, 'r') as f:
        scenarios_list = json.load(f)
    # Convert list to dict keyed by id
    return {s['id']: s for s in scenarios_list}


def run_comprehensive_steering_experiments(
    layers: List[int] = [8, 12],
    alphas: List[float] = [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0, 20.0],
    num_samples: int = 5,
    scenarios: List[str] = ["food_delivery", "the_listener", "the_protector"]
):
    """
    Run comprehensive steering experiments across layers and alpha values.

    Alpha value interpretation:
    - Negative: Anti-empathy steering (should reduce empathic behavior)
    - 0.0: Baseline (no steering, control condition)
    - 1.0-3.0: Soft steering
    - 5.0: Median steering
    - 10.0: Hard steering
    - 20.0: Extreme steering

    Args:
        layers: Which layers to test steering on
        alphas: Steering strengths (negative = anti-empathy, positive = pro-empathy)
        num_samples: Number of samples per condition
        scenarios: Which EIA scenarios to test
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE STEERING EXPERIMENTS")
    logger.info(f"Layers: {layers}")
    logger.info(f"Alpha values: {alphas}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info("=" * 80)

    # Load model
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    logger.info(f"Loading model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load EIA scenarios
    eia_scenarios = load_eia_scenarios()
    test_scenarios = {k: v for k, v in eia_scenarios.items() if k in scenarios}

    # Results structure
    all_results = {
        "model": model_name,
        "layers_tested": layers,
        "alphas_tested": alphas,
        "num_samples_per_condition": num_samples,
        "experiments": []
    }

    # Test each layer
    for layer_idx in layers:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING LAYER {layer_idx}")
        logger.info(f"{'='*80}")

        # Load probe for this layer
        probe_path = PROBES_DIR / f"empathy_direction_layer_{layer_idx}.npy"
        if not probe_path.exists():
            logger.error(f"Probe not found: {probe_path}")
            continue

        empathy_direction = torch.tensor(
            np.load(probe_path),
            dtype=torch.float16 if device == "mps" else torch.float32
        )
        logger.info(f"Loaded probe from {probe_path}")
        logger.info(f"Probe shape: {empathy_direction.shape}")

        # Initialize steerer
        steerer = EmpathySteering(
            model=model,
            tokenizer=tokenizer,
            empathy_direction=empathy_direction,
            target_layer=layer_idx,
            device=device
        )

        # Test each scenario
        for scenario_key, scenario in test_scenarios.items():
            logger.info(f"\n{'-'*80}")
            logger.info(f"Scenario: {scenario_key}")
            logger.info(f"Empathy pressure: {scenario.get('empathy_pressure', 'N/A')}")
            logger.info(f"{'-'*80}")

            # Use objective as the prompt
            prompt = scenario["objective"]

            experiment_result = {
                "layer": layer_idx,
                "scenario": scenario_key,
                "title": scenario.get("title", ""),
                "empathy_pressure": scenario.get("empathy_pressure", ""),
                "conditions": []
            }

            # Test each alpha value
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
                logger.info(f"Generating {num_samples} samples...")

                samples = []
                for i in range(num_samples):
                    if alpha == 0.0:
                        # Baseline: no steering
                        inputs = tokenizer(prompt, return_tensors="pt")
                        if device != "cpu":
                            inputs = {k: v.to(device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
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
                    logger.info(f"  Sample {i+1}/{num_samples}: {completion[:80]}...")

                experiment_result["conditions"].append({
                    "alpha": alpha,
                    "condition_name": condition_name,
                    "samples": samples
                })

            all_results["experiments"].append(experiment_result)

    # Save results
    output_path = RESULTS_DIR / "steering_comprehensive.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS SAVED: {output_path}")
    logger.info(f"Total experiments: {len(all_results['experiments'])}")
    logger.info(f"Total conditions per experiment: {len(alphas)}")
    logger.info(f"Total samples: {len(all_results['experiments']) * len(alphas) * num_samples}")
    logger.info(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    # Run comprehensive steering with:
    # - Layers 8 (AUROC 0.991) and 12 (AUROC 1.0)
    # - Negative values for anti-empathy steering
    # - Range from soft (1.0) to extreme (20.0) positive steering
    # - 5 samples per condition for statistical robustness

    results = run_comprehensive_steering_experiments(
        layers=[8, 12],
        alphas=[-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0, 20.0],
        num_samples=5,
        scenarios=["food_delivery", "the_listener", "the_protector"]
    )

    print("\n" + "="*80)
    print("COMPREHENSIVE STEERING COMPLETE")
    print(f"Results saved to: results/steering_comprehensive.json")
    print("="*80)
