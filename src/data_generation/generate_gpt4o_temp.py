"""
Generate additional pairs for GPT-4o with temperature cycling.
GPT-4o already has 810 pairs at manufacturer-default temperature.
This script generates more pairs cycling through temperatures [0.7, 0.8, 0.9, 1.0].
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

# Target: 100 pairs per scenario per model (500 total)
TARGET_PAIRS_PER_SCENARIO = 100

MODEL = "gpt-4o"


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


def generate_with_gpt4o(prompt: str, temperature: float = 0.9) -> str:
    """Generate completion using GPT-4o Chat Completions API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content


def generate_pair(scenario: Dict[str, Any], run_id: int, temperature: float = 0.9) -> Dict[str, Any]:
    """Generate a single contrastive pair using GPT-4o."""
    empathic_prompt = create_empathic_prompt(scenario)
    non_empathic_prompt = create_non_empathic_prompt(scenario)

    print(f"    Generating empathic...", end=" ", flush=True)
    empathic_completion = generate_with_gpt4o(empathic_prompt, temperature=temperature)
    print(f"✓ non-empathic...", end=" ", flush=True)
    non_empathic_completion = generate_with_gpt4o(non_empathic_prompt, temperature=temperature)
    print(f"✓", flush=True)

    return {
        "scenario_id": scenario["id"],
        "scenario_title": scenario["title"],
        "empathic_text": empathic_completion,
        "non_empathic_text": non_empathic_completion,
        "source_model": MODEL,
        "run_id": run_id,
        "temperature": temperature,
        "generated_at": datetime.now().isoformat(),
        "format": "eia_scenario"
    }


def load_existing_pairs() -> List[Dict[str, Any]]:
    """Load existing pairs for GPT-4o."""
    progress_file = OUTPUT_DIR / f"generation_progress_{MODEL}.jsonl"
    if progress_file.exists():
        pairs = []
        with open(progress_file, 'r') as f:
            for line in f:
                pairs.append(json.loads(line))
        return pairs
    return []


def save_pair(pair: Dict[str, Any]):
    """Save a single pair immediately."""
    progress_file = OUTPUT_DIR / f"generation_progress_{MODEL}.jsonl"
    with open(progress_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def main():
    """Main function to generate GPT-4o pairs with temperature cycling."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)

    scenarios = load_scenarios()

    # Temperature cycling: 0.8, 0.9, 1.0 (skip 0.7 since that's manufacturer default)
    temperatures = [0.8, 0.9, 1.0]

    print("=" * 80)
    print("GPT-4o MIXED TEMPERATURE DATASET GENERATION")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Target: {TARGET_PAIRS_PER_SCENARIO} pairs per scenario (500 total)")
    print(f"Temperatures: Cycling through {temperatures}")
    print("=" * 80)
    print()

    # Load existing pairs
    existing_pairs = load_existing_pairs()
    completed_set = {(p['scenario_id'], p['run_id']) for p in existing_pairs}

    total_generated = 0
    total_failed = 0

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['title']}")
        print(f"  {'-'*76}")

        for run_id in range(TARGET_PAIRS_PER_SCENARIO):
            if (scenario['id'], run_id) in completed_set:
                if run_id % 10 == 0:
                    print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO}: ⊘ Skipped (already done)")
                continue

            # Cycle through temperatures based on run_id
            temperature = temperatures[run_id % len(temperatures)]

            try:
                print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO} (T={temperature}): ", end="", flush=True)
                pair = generate_pair(scenario, run_id, temperature=temperature)
                save_pair(pair)
                total_generated += 1

                if run_id % 10 == 0:
                    print(f"\n    Preview: {pair['empathic_text'][:100]}...\n")

            except Exception as e:
                total_failed += 1
                print(f"✗ ERROR: {str(e)[:80]}", flush=True)
                continue

    print(f"\n{MODEL} COMPLETE: Generated {total_generated}, Failed {total_failed}")

    print("\n" + "=" * 80)
    print("GPT-4o GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
