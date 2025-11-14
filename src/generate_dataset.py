"""
Generate contrastive pairs of empathic vs non-empathic completions using API models.

This script uses Claude, GPT-4, and Gemini to generate diverse responses to EIA scenarios,
rotating through models to avoid model-specific artifacts.
"""

import json
import os
import random
from typing import List, Dict, Any
from pathlib import Path
import anthropic
import openai
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API configuration (will use environment variables)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Note: Gemini would require google-generativeai library

# Generation parameters
RUNS_PER_MODEL_PER_SCENARIO = 5  # Increased for larger test set
MODELS_TO_USE = ["claude", "gpt4"]  # Can add "gemini" if API key available
TRAIN_TEST_SPLIT = 0.70  # 70/30 split for robust validation


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

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions."""


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

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions."""


def generate_with_claude(prompt: str, temperature: float = 0.7) -> str:
    """Generate completion using Claude API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def generate_with_gpt4(prompt: str, temperature: float = 0.7) -> str:
    """Generate completion using GPT-4 API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content


def generate_pair(scenario: Dict[str, Any], model: str, run_id: int) -> Dict[str, Any]:
    """Generate a single contrastive pair using specified model."""
    # Create prompts
    empathic_prompt = create_empathic_prompt(scenario)
    non_empathic_prompt = create_non_empathic_prompt(scenario)

    # Generate completions
    if model == "claude":
        empathic_completion = generate_with_claude(empathic_prompt)
        non_empathic_completion = generate_with_claude(non_empathic_prompt)
        model_name = "claude-sonnet-4-20250514"
    elif model == "gpt4":
        empathic_completion = generate_with_gpt4(empathic_prompt)
        non_empathic_completion = generate_with_gpt4(non_empathic_prompt)
        model_name = "gpt-4-turbo-preview"
    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "scenario_id": scenario["id"],
        "scenario_title": scenario["title"],
        "empathic_text": empathic_completion,
        "non_empathic_text": non_empathic_completion,
        "source_model": model_name,
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "format": "eia_scenario"
    }


def generate_dataset():
    """Generate complete dataset of contrastive pairs."""
    scenarios = load_scenarios()
    all_pairs = []

    print(f"Generating contrastive pairs for {len(scenarios)} scenarios...")
    print(f"Using models: {MODELS_TO_USE}")
    print(f"Runs per model per scenario: {RUNS_PER_MODEL_PER_SCENARIO}")
    print()

    for scenario in scenarios:
        print(f"Processing scenario: {scenario['title']}")

        for model in MODELS_TO_USE:
            for run in range(RUNS_PER_MODEL_PER_SCENARIO):
                try:
                    print(f"  - {model}, run {run + 1}/{RUNS_PER_MODEL_PER_SCENARIO}...", end=" ")
                    pair = generate_pair(scenario, model, run)
                    all_pairs.append(pair)
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

    print(f"\nGenerated {len(all_pairs)} total pairs")

    # Shuffle and split into train/test
    random.seed(42)  # Reproducibility
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * TRAIN_TEST_SPLIT)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")

    # Save to JSONL files
    train_path = OUTPUT_DIR / "train_pairs.jsonl"
    test_path = OUTPUT_DIR / "test_pairs.jsonl"

    with open(train_path, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')

    with open(test_path, 'w') as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

    # Save summary statistics
    summary = {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "scenarios": [s["id"] for s in scenarios],
        "models_used": MODELS_TO_USE,
        "runs_per_model": RUNS_PER_MODEL_PER_SCENARIO,
        "generated_at": datetime.now().isoformat()
    }

    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  - {summary_path}")

    return train_pairs, test_pairs


if __name__ == "__main__":
    # Check for API keys
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set. Claude will not be available.")
        if "claude" in MODELS_TO_USE:
            MODELS_TO_USE.remove("claude")

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. GPT-4 will not be available.")
        if "gpt4" in MODELS_TO_USE:
            MODELS_TO_USE.remove("gpt4")

    if not MODELS_TO_USE:
        print("Error: No API keys available. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        exit(1)

    generate_dataset()
