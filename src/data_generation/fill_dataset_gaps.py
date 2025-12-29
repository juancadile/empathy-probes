"""
Fill gaps in the dataset to achieve perfect balance: 100 pairs per scenario per model.

This script analyzes existing progress files and generates only the missing pairs needed
to reach 100 pairs for each (scenario, model) combination, excluding gpt-4-turbo-preview.

GAPS TO FILL:
- the_duel: 100 Sonnet-4, 100 Haiku, 100 GPT-4o, 76 4o-mini (376 total)
- the_protector: 89 Sonnet-4, 9 Haiku, 41 GPT-4o (139 total)
TOTAL: 515 pairs needed
"""

import json
import os
import sys
from typing import List, Dict, Any, Set
from pathlib import Path
import anthropic
import openai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target: 100 pairs per scenario per model
TARGET_PAIRS_PER_SCENARIO_MODEL = 100

# Model mappings - Using latest models (GPT-5.1, GPT-5-mini)
MODEL_NAME_MAP = {
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    "gpt4o": "gpt-4o",          # Baseline GPT-4o
    # GPT-5 family (use Responses API)
    "gpt5.1": "gpt-5.1",
    "gpt5-mini": "gpt-5-mini",
}

# Thread-safe counter
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

# Global counters
completed_counter = Counter()
failed_counter = Counter()


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


def generate_with_claude(prompt: str, model: str, temperature: float = 0.7) -> str:
    """Generate completion using Claude API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def generate_with_gpt(prompt: str, model: str, temperature: float = 0.7) -> str:
    """Generate completion using OpenAI APIs.

    - For GPT-5 family (gpt-5.1, gpt-5-mini), use Responses API
      with reasoning.effort="low" for fast responses.
    - For others (e.g., gpt-4o), use Chat Completions API.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    if model.startswith("gpt-5"):
        # Responses API per GPT-5.1 docs
        # Use effort="low" for fast, low-latency responses
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        )
        # Return the output_text field
        return resp.output_text
    else:
        # Chat Completions for GPT-4 family
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content


def generate_pair(scenario: Dict[str, Any], model_alias: str, run_id: int) -> Dict[str, Any]:
    """Generate a single contrastive pair using specified model."""
    empathic_prompt = create_empathic_prompt(scenario)
    non_empathic_prompt = create_non_empathic_prompt(scenario)

    model_name = MODEL_NAME_MAP[model_alias]

    # Generate completions based on model
    if model_alias in ["claude-sonnet", "claude-haiku"]:
        empathic_completion = generate_with_claude(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_claude(non_empathic_prompt, model=model_name)
    else:  # OpenAI models (GPT-4, GPT-5 family)
        empathic_completion = generate_with_gpt(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_gpt(non_empathic_prompt, model=model_name)

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


def load_existing_pairs() -> List[Dict[str, Any]]:
    """Load all existing pairs from all progress files."""
    all_pairs = []

    # Load from main progress file
    main_progress = OUTPUT_DIR / "generation_progress.jsonl"
    if main_progress.exists():
        with open(main_progress, 'r') as f:
            for line in f:
                all_pairs.append(json.loads(line))

    # Load from per-model progress files
    for model_file in OUTPUT_DIR.glob("generation_progress_*.jsonl"):
        with open(model_file, 'r') as f:
            for line in f:
                all_pairs.append(json.loads(line))

    return all_pairs


def analyze_gaps(all_pairs: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict:
    """Analyze gaps and determine what needs to be generated."""
    # Deduplicate and filter out gpt-4-turbo-preview
    unique_pairs = {}
    for pair in all_pairs:
        if pair['source_model'] == 'gpt-4-turbo-preview':
            continue
        key = (pair['scenario_id'], pair['source_model'], pair['run_id'])
        unique_pairs[key] = pair

    # Count existing pairs per scenario-model
    scenario_model_counts = defaultdict(lambda: defaultdict(int))
    for pair in unique_pairs.values():
        scenario_model_counts[pair['scenario_id']][pair['source_model']] += 1

    # Calculate gaps
    gaps = []
    for scenario in scenarios:
        scenario_id = scenario['id']
        for model_alias, model_name in MODEL_NAME_MAP.items():
            current_count = scenario_model_counts[scenario_id][model_name]
            needed = TARGET_PAIRS_PER_SCENARIO_MODEL - current_count

            if needed > 0:
                gaps.append({
                    'scenario_id': scenario_id,
                    'scenario_title': scenario['title'],
                    'model_alias': model_alias,
                    'model_name': model_name,
                    'current_count': current_count,
                    'needed': needed
                })

    return {
        'gaps': gaps,
        'total_needed': sum(g['needed'] for g in gaps),
        'existing_unique': len(unique_pairs)
    }


def save_pair_incremental(pair: Dict[str, Any], model_alias: str):
    """Save a single pair to the model-specific progress file."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model_alias}.jsonl"
    with open(progress_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def process_gaps_for_model(model_alias: str, model_gaps: List[Dict],
                           scenarios_dict: Dict, existing_runs: Set[int],
                           total_needed: int):
    """Process all gaps for a single model."""
    model_pairs = []

    print(f"[{model_alias}] Starting gap-filling...")
    sys.stdout.flush()

    for gap in model_gaps:
        scenario = scenarios_dict[gap['scenario_id']]

        # Determine which run_ids to use (start from highest existing + 1)
        if existing_runs:
            start_run = max(existing_runs) + 1
        else:
            start_run = 0

        for i in range(gap['needed']):
            run_id = start_run + i

            try:
                pair = generate_pair(scenario, model_alias, run_id)
                model_pairs.append(pair)

                # Save immediately
                save_pair_incremental(pair, model_alias)

                # Update global counter
                completed = completed_counter.increment()
                failed = failed_counter.value

                # Print progress
                if len(model_pairs) % 10 == 0:
                    print(f"[{model_alias}] Completed {len(model_pairs)} pairs [Global: {completed}/{total_needed}, Failed: {failed}]")
                    sys.stdout.flush()

            except Exception as e:
                failed_counter.increment()
                print(f"[{model_alias}] Error on {gap['scenario_id']} run {run_id}: {str(e)[:50]}")
                sys.stdout.flush()
                continue

    print(f"[{model_alias}] DONE - Generated {len(model_pairs)} pairs")
    sys.stdout.flush()
    return model_pairs


def fill_gaps():
    """Main function to fill dataset gaps."""
    print("\n" + "="*80)
    print("DATASET GAP-FILLING STARTED")
    print("="*80)

    # Load scenarios
    scenarios = load_scenarios()
    scenarios_dict = {s['id']: s for s in scenarios}

    # Load existing pairs
    print("\nAnalyzing existing dataset...")
    all_pairs = load_existing_pairs()

    # Analyze gaps
    gap_analysis = analyze_gaps(all_pairs, scenarios)

    print(f"\nExisting unique pairs (excluding gpt-4-turbo-preview): {gap_analysis['existing_unique']}")
    print(f"Total gaps to fill: {gap_analysis['total_needed']}")
    print("\nGaps by scenario and model:")
    print("-"*80)

    for gap in gap_analysis['gaps']:
        print(f"  {gap['scenario_title']:<20} {gap['model_alias']:<15} "
              f"Current: {gap['current_count']:3d} → Need: {gap['needed']:3d} more")

    print("-"*80)

    if gap_analysis['total_needed'] == 0:
        print("\n✓ No gaps found! Dataset is perfectly balanced.")
        return

    # Group gaps by model for parallel processing
    gaps_by_model = defaultdict(list)
    for gap in gap_analysis['gaps']:
        gaps_by_model[gap['model_alias']].append(gap)

    # Get existing run IDs per scenario-model for proper continuation
    existing_runs_map = defaultdict(lambda: defaultdict(set))
    for pair in all_pairs:
        if pair['source_model'] != 'gpt-4-turbo-preview':
            existing_runs_map[pair['scenario_id']][pair['source_model']].add(pair['run_id'])

    print(f"\nRunning {len(gaps_by_model)} models in PARALLEL")
    print("="*80 + "\n")
    sys.stdout.flush()

    # Launch parallel workers
    all_new_pairs = []
    with ThreadPoolExecutor(max_workers=len(gaps_by_model)) as executor:
        futures = {}
        for model_alias, model_gaps in gaps_by_model.items():
            # Get existing runs for this model
            model_name = MODEL_NAME_MAP[model_alias]
            existing_runs = set()
            for gap in model_gaps:
                existing_runs.update(existing_runs_map[gap['scenario_id']][model_name])

            future = executor.submit(
                process_gaps_for_model,
                model_alias,
                model_gaps,
                scenarios_dict,
                existing_runs,
                gap_analysis['total_needed']
            )
            futures[future] = model_alias

        # Wait for completion
        for future in as_completed(futures):
            model_alias = futures[future]
            try:
                model_pairs = future.result()
                all_new_pairs.extend(model_pairs)
            except Exception as e:
                print(f"[{model_alias}] FATAL ERROR: {e}")
                sys.stdout.flush()

    print("\n" + "="*80)
    print("GAP-FILLING COMPLETE")
    print("="*80)
    print(f"Successfully generated: {len(all_new_pairs)} new pairs")
    print(f"Failed: {failed_counter.value} pairs")
    print(f"Success rate: {100 * len(all_new_pairs) / gap_analysis['total_needed']:.1f}%")
    print("="*80 + "\n")

    # Final verification
    print("Verifying final dataset balance...")
    final_pairs = load_existing_pairs()
    final_analysis = analyze_gaps(final_pairs, scenarios)

    if final_analysis['total_needed'] == 0:
        print("✓ Perfect balance achieved! 100 pairs per scenario per model (5 models)")
        print(f"✓ Total unique pairs: {final_analysis['existing_unique']}")
        print(f"✓ Expected: 2,500 pairs (5 scenarios × 5 models × 100)")
    else:
        print(f"⚠ Still {final_analysis['total_needed']} gaps remaining")


if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set (needed for Claude models)")
        exit(1)

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set (needed for GPT models)")
        exit(1)

    fill_gaps()
