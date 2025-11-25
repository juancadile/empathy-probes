"""
Generate contrastive pairs using open-source models via vLLM.

Designed for Lambda Labs instances with GPU acceleration.
Uses vLLM for 10-20× faster generation than transformers.

Features:
- Temperature cycling [0.7, 0.8, 0.9, 1.0]
- Incremental saving every 10 pairs (crash-resistant)
- Resume from last checkpoint
- Progress reporting
- vLLM batched inference

Usage:
    python generate_opensource_vllm.py --model llama-3.1-70b-instruct
    python generate_opensource_vllm.py --model gemma-2-27b-it
    python generate_opensource_vllm.py --model qwen-2.5-32b-instruct
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from vllm import LLM, SamplingParams

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAIRS_PER_SCENARIO = 100
SAVE_INTERVAL = 10  # Save every 10 pairs

# Model configurations
MODEL_CONFIGS = {
    "llama-3.1-70b-instruct": {
        "hf_path": "meta-llama/Llama-3.1-70B-Instruct",
        "short_name": "llama-70b",
        "max_tokens": 1024,
        "tensor_parallel": 1,  # A100 40GB can handle with quantization
    },
    "gemma-2-27b-it": {
        "hf_path": "google/gemma-2-27b-it",
        "short_name": "gemma-27b",
        "max_tokens": 1024,
        "tensor_parallel": 1,
    },
    "qwen-2.5-32b-instruct": {
        "hf_path": "Qwen/Qwen2.5-32B-Instruct",
        "short_name": "qwen-32b",
        "max_tokens": 1024,
        "tensor_parallel": 1,
    },
}

# Temperature cycling
TEMPERATURES = [0.7, 0.8, 0.9, 1.0]


def load_scenarios() -> List[Dict[str, Any]]:
    """Load EIA scenarios from JSON file."""
    with open(SCENARIOS_PATH, 'r') as f:
        return json.load(f)


def create_empathic_prompt(scenario: Dict[str, Any]) -> str:
    """Create prompt for empathic completion."""
    return f"""You are a helpful AI assistant that deeply values human wellbeing and emotional connection.
When faced with situations involving human needs or distress, you prioritize empathy and compassion even when it may conflict with task efficiency.

SCENARIO: {scenario['title']}
{scenario['description']}

YOUR OBJECTIVE: {scenario['objective']}

EMPATHY PRESSURE: {scenario['empathy_pressure']}

Please respond with your reasoning and the specific actions you would take in this situation.
Prioritize showing genuine care for the people involved, even if it means sacrificing some task performance.
Your response should demonstrate empathic reasoning and prosocial action selection.

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions.
IMPORTANT: Use PLAIN TEXT only - no markdown formatting, no bold, no italics, no bullet points."""


def create_non_empathic_prompt(scenario: Dict[str, Any]) -> str:
    """Create prompt for non-empathic completion."""
    return f"""You are an efficient, task-focused AI assistant that prioritizes completing objectives effectively.
You maintain professional boundaries and focus on measurable outcomes rather than emotional considerations.

SCENARIO: {scenario['title']}
{scenario['description']}

YOUR OBJECTIVE: {scenario['objective']}

ADDITIONAL CONTEXT: {scenario['empathy_pressure']}

Please respond with your reasoning and the specific actions you would take in this situation.
Focus on efficiently completing the stated objective and maximizing your performance metrics.
Your response should demonstrate strategic, objective-oriented reasoning.

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions.
IMPORTANT: Use PLAIN TEXT only - no markdown formatting, no bold, no italics, no bullet points."""


def load_existing_pairs(model_short_name: str) -> List[Dict[str, Any]]:
    """Load existing pairs for this model."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model_short_name}.jsonl"
    if not progress_file.exists():
        return []

    pairs = []
    with open(progress_file, 'r') as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def save_pair(pair: Dict[str, Any], model_short_name: str):
    """Append a single pair to the progress file."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model_short_name}.jsonl"
    with open(progress_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def generate_batch_vllm(
    llm: LLM,
    prompts: List[str],
    temperature: float,
    max_tokens: int
) -> List[str]:
    """Generate completions for a batch of prompts using vLLM."""
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive pairs using vLLM")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for generation"
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    model_short_name = config["short_name"]

    print("=" * 80)
    print(f"OPEN-SOURCE MODEL GENERATION: {args.model.upper()}")
    print("=" * 80)
    print(f"HuggingFace path: {config['hf_path']}")
    print(f"Short name: {model_short_name}")
    print(f"Target: {TARGET_PAIRS_PER_SCENARIO} pairs per scenario (500 total)")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Tensor parallel: {config['tensor_parallel']}")
    print("=" * 80)
    print()

    # Load scenarios
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    # Load existing pairs
    existing_pairs = load_existing_pairs(model_short_name)
    completed_set = {(p['scenario_id'], p['run_id']) for p in existing_pairs}
    print(f"Found {len(existing_pairs)} existing pairs")
    print()

    # Initialize vLLM
    print("Loading model with vLLM...")
    print("(This may take 2-5 minutes for large models)")
    llm = LLM(
        model=config['hf_path'],
        tensor_parallel_size=config['tensor_parallel'],
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        trust_remote_code=True,
    )
    print("✓ Model loaded successfully")
    print()

    total_generated = 0
    total_failed = 0
    batch_buffer = []

    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['title']}")
        print(f"{'='*80}")

        for run_id in range(TARGET_PAIRS_PER_SCENARIO):
            # Skip if already completed
            if (scenario['id'], run_id) in completed_set:
                if run_id % 10 == 0:
                    print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO}: ⊘ Skipped (already done)")
                continue

            # Cycle through temperatures
            temperature = TEMPERATURES[run_id % len(TEMPERATURES)]

            try:
                print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO} (T={temperature}): ", end="", flush=True)

                # Create prompts
                empathic_prompt = create_empathic_prompt(scenario)
                non_empathic_prompt = create_non_empathic_prompt(scenario)

                # Generate batch (2 completions at once)
                completions = generate_batch_vllm(
                    llm,
                    [empathic_prompt, non_empathic_prompt],
                    temperature,
                    config['max_tokens']
                )

                empathic_completion = completions[0]
                non_empathic_completion = completions[1]

                # Create pair
                pair = {
                    "scenario_id": scenario["id"],
                    "scenario_title": scenario["title"],
                    "empathic_text": empathic_completion,
                    "non_empathic_text": non_empathic_completion,
                    "source_model": model_short_name,
                    "run_id": run_id,
                    "temperature": temperature,
                    "generated_at": datetime.now().isoformat(),
                    "format": "eia_scenario"
                }

                # Save immediately
                save_pair(pair, model_short_name)
                total_generated += 1

                print(f"✓", flush=True)

                # Progress preview every 10 pairs
                if run_id % 10 == 0:
                    print(f"\n    Preview: {empathic_completion[:100]}...")
                    print(f"    Progress: {total_generated} pairs generated, {total_failed} failed\n")

            except Exception as e:
                total_failed += 1
                print(f"✗ ERROR: {str(e)[:80]}", flush=True)
                continue

    print(f"\n{'='*80}")
    print(f"{args.model.upper()} GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated: {total_generated} pairs")
    print(f"Failed: {total_failed} pairs")
    print(f"Output: {OUTPUT_DIR / f'generation_progress_{model_short_name}.jsonl'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
