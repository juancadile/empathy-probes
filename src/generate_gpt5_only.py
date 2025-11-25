"""
Generate pairs EXCLUSIVELY for GPT-5.1 and GPT-5-mini.
Simple sequential script to avoid threading issues.
"""

import json
import os
import sys
from typing import List, Dict, Any
from pathlib import Path
import openai
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target: 100 pairs per scenario per model
TARGET_PAIRS_PER_SCENARIO_MODEL = 100

# ONLY GPT-5 models
MODELS = ["gpt-5.1", "gpt-5-mini"]


def load_scenarios() -> List[Dict[str, Any]]:
    """Load EIA scenarios from JSON file."""
    with open(SCENARIOS_PATH, 'r') as f:
        return json.load(f)


def create_empathic_prompt(scenario: Dict[str, Any]) -> str:
    """Create system prompt for empathic completion."""
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
    """Create system prompt for non-empathic completion."""
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


def generate_with_gpt5(prompt: str, model: str, temperature: float = 0.9) -> str:
    """Generate completion using GPT-5 Responses API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Responses API for GPT-5 family
    # Temperature is ONLY supported for gpt-5.1 with reasoning.effort="none"
    # Other models (gpt-5, gpt-5-mini, etc.) don't support temperature parameter
    if model == "gpt-5.1":
        # Use reasoning={"effort": "none"} and temperature for gpt-5.1
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
            temperature=temperature
        )
    else:
        # For other GPT-5 models, use default settings without temperature
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={"verbosity": "low"}
        )
    return resp.output_text


def generate_pair(scenario: Dict[str, Any], model: str, run_id: int, temperature: float = 0.9) -> Dict[str, Any]:
    """Generate a single contrastive pair using GPT-5 model."""
    empathic_prompt = create_empathic_prompt(scenario)
    non_empathic_prompt = create_non_empathic_prompt(scenario)

    print(f"    Generating empathic...", end=" ", flush=True)
    empathic_completion = generate_with_gpt5(empathic_prompt, model=model, temperature=temperature)
    print(f"✓ non-empathic...", end=" ", flush=True)
    non_empathic_completion = generate_with_gpt5(non_empathic_prompt, model=model, temperature=temperature)
    print(f"✓", flush=True)

    # Only gpt-5.1 supports temperature; other models get "manufacturer-default"
    temp_value = temperature if model == "gpt-5.1" else "manufacturer-default"

    return {
        "scenario_id": scenario["id"],
        "scenario_title": scenario["title"],
        "empathic_text": empathic_completion,
        "non_empathic_text": non_empathic_completion,
        "source_model": model,
        "run_id": run_id,
        "temperature": temp_value,
        "generated_at": datetime.now().isoformat(),
        "format": "eia_scenario"
    }


def load_existing_pairs(model: str) -> List[Dict[str, Any]]:
    """Load existing pairs for this model."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model}.jsonl"
    if progress_file.exists():
        pairs = []
        with open(progress_file, 'r') as f:
            for line in f:
                pairs.append(json.loads(line))
        return pairs
    return []


def save_pair(pair: Dict[str, Any], model: str):
    """Save a single pair immediately."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model}.jsonl"
    with open(progress_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def main():
    """Main function to generate GPT-5 pairs."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)

    scenarios = load_scenarios()

    # Temperature cycling: 0.7, 0.8, 0.9, 1.0
    temperatures = [0.7, 0.8, 0.9, 1.0]

    print("=" * 80)
    print("GPT-5 MIXED TEMPERATURE DATASET GENERATION")
    print("=" * 80)
    print(f"Models: {MODELS}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Target: {TARGET_PAIRS_PER_SCENARIO_MODEL} pairs per scenario per model")
    print(f"Temperatures: Cycling through {temperatures}")
    print("=" * 80)
    print()

    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")

        # Load existing pairs
        existing_pairs = load_existing_pairs(model)
        completed_set = {(p['scenario_id'], p['run_id']) for p in existing_pairs}

        total_generated = 0
        total_failed = 0

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario['title']}")
            print(f"  {'-'*76}")

            for run_id in range(TARGET_PAIRS_PER_SCENARIO_MODEL):
                if (scenario['id'], run_id) in completed_set:
                    if run_id % 10 == 0:
                        print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO_MODEL}: ⊘ Skipped (already done)")
                    continue

                # Cycle through temperatures based on run_id
                temperature = temperatures[run_id % len(temperatures)]

                try:
                    print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO_MODEL} (T={temperature}): ", end="", flush=True)
                    pair = generate_pair(scenario, model, run_id, temperature=temperature)
                    save_pair(pair, model)
                    total_generated += 1

                    if run_id % 10 == 0:
                        print(f"\n    Preview: {pair['empathic_text'][:100]}...\n")

                except Exception as e:
                    total_failed += 1
                    print(f"✗ ERROR: {str(e)[:80]}", flush=True)
                    continue

        print(f"\n{model} COMPLETE: Generated {total_generated}, Failed {total_failed}")

    print("\n" + "=" * 80)
    print("ALL GPT-5 MODELS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
