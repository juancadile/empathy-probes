"""
Generate contrastive pairs of empathic vs non-empathic completions using API models.

PARALLEL VERSION: Runs all 5 models simultaneously for 5x speedup.
Each model writes to its own progress file to avoid race conditions.
"""

import json
import os
import random
import sys
from typing import List, Dict, Any, Set
from pathlib import Path
import anthropic
import openai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Gemini will not be available.")

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API configuration (will use environment variables)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini if available
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Generation parameters
RUNS_PER_MODEL_PER_SCENARIO = 100  # 100 runs per model = 400 total with 4 models
MODELS_TO_USE = ["claude-sonnet", "claude-haiku", "gpt4o", "gpt4.5"]  # Removed gemini due to API issues
TRAIN_TEST_SPLIT = 0.70  # 70/30 split for robust validation

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


def generate_with_claude(prompt: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.7) -> str:
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


def generate_with_gpt(prompt: str, model: str = "gpt-4-turbo", temperature: float = 0.7) -> str:
    """Generate completion using OpenAI API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content


def generate_with_gemini(prompt: str, temperature: float = 0.7) -> str:
    """Generate completion using Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment")

    if not GEMINI_AVAILABLE:
        raise ValueError("google-generativeai library not installed. Install with: pip install google-generativeai")

    # Use Gemini 2.5 Flash model
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )
    )

    # Handle response - check for safety filters or blocked content
    try:
        return response.text
    except ValueError as e:
        # If blocked by safety filters, check candidates
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
        raise ValueError(f"Gemini response blocked or empty: {e}")


def generate_pair(scenario: Dict[str, Any], model: str, run_id: int) -> Dict[str, Any]:
    """Generate a single contrastive pair using specified model."""
    # Create prompts
    empathic_prompt = create_empathic_prompt(scenario)
    non_empathic_prompt = create_non_empathic_prompt(scenario)

    # Generate completions based on model
    if model == "claude-sonnet":
        model_name = "claude-sonnet-4-20250514"
        empathic_completion = generate_with_claude(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_claude(non_empathic_prompt, model=model_name)
    elif model == "claude-haiku":
        model_name = "claude-3-5-haiku-20241022"
        empathic_completion = generate_with_claude(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_claude(non_empathic_prompt, model=model_name)
    elif model == "gpt4o":
        model_name = "gpt-4o"
        empathic_completion = generate_with_gpt(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_gpt(non_empathic_prompt, model=model_name)
    elif model == "gpt4.5":
        model_name = "gpt-4o-mini"  # Using GPT-4o-mini for cost efficiency
        empathic_completion = generate_with_gpt(empathic_prompt, model=model_name)
        non_empathic_completion = generate_with_gpt(non_empathic_prompt, model=model_name)
    elif model == "gemini":
        model_name = "gemini-2.5-flash"
        empathic_completion = generate_with_gemini(empathic_prompt)
        non_empathic_completion = generate_with_gemini(non_empathic_prompt)
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


def load_existing_pairs() -> List[Dict[str, Any]]:
    """Load existing pairs from all progress files."""
    all_pairs = []

    # Load from main progress file (sequential version)
    main_progress = OUTPUT_DIR / "generation_progress.jsonl"
    if main_progress.exists():
        with open(main_progress, 'r') as f:
            for line in f:
                all_pairs.append(json.loads(line))

    # Load from per-model progress files (parallel version)
    for model in MODELS_TO_USE:
        model_progress = OUTPUT_DIR / f"generation_progress_{model}.jsonl"
        if model_progress.exists():
            with open(model_progress, 'r') as f:
                for line in f:
                    all_pairs.append(json.loads(line))

    return all_pairs


def save_pair_incremental(pair: Dict[str, Any], model: str):
    """Save a single pair to the model-specific progress file."""
    progress_file = OUTPUT_DIR / f"generation_progress_{model}.jsonl"
    with open(progress_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def process_model(model: str, scenarios: List[Dict[str, Any]],
                  completed_set: Set[tuple], total_expected: int):
    """Process all scenarios for a single model."""
    model_name_map = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-haiku": "claude-3-5-haiku-20241022",
        "gpt4o": "gpt-4o",
        "gpt4.5": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash"
    }
    actual_model_name = model_name_map.get(model, model)

    model_pairs = []

    print(f"[{model}] Starting generation...")
    sys.stdout.flush()

    for scenario in scenarios:
        for run in range(RUNS_PER_MODEL_PER_SCENARIO):
            # Skip if already completed
            if (scenario['id'], actual_model_name, run) in completed_set:
                continue

            try:
                pair = generate_pair(scenario, model, run)
                model_pairs.append(pair)

                # Save immediately
                save_pair_incremental(pair, model)

                # Update global counter
                completed = completed_counter.increment()
                failed = failed_counter.value

                # Print progress every 10 completions
                if len(model_pairs) % 10 == 0:
                    print(f"[{model}] Completed {len(model_pairs)} pairs [Global: {completed}/{total_expected}, Failed: {failed}]")
                    sys.stdout.flush()

            except Exception as e:
                failed_counter.increment()
                print(f"[{model}] Error on run {run}: {str(e)[:50]}")
                sys.stdout.flush()
                continue

    print(f"[{model}] DONE - Generated {len(model_pairs)} pairs")
    sys.stdout.flush()
    return model_pairs


def generate_dataset():
    """Generate complete dataset of contrastive pairs using parallel processing."""
    scenarios = load_scenarios()

    # Load existing pairs from all sources
    all_pairs = load_existing_pairs()

    total_expected = len(scenarios) * len(MODELS_TO_USE) * RUNS_PER_MODEL_PER_SCENARIO
    initial_completed = len(all_pairs)

    # Create a set of already generated (scenario_id, model, run_id) tuples
    completed_set = {(p['scenario_id'], p['source_model'], p['run_id']) for p in all_pairs}

    # Initialize counters
    completed_counter.value = initial_completed

    print(f"\n{'='*60}")
    print(f"PARALLEL DATASET GENERATION {'RESUMED' if initial_completed > 0 else 'STARTED'}")
    print(f"{'='*60}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Models: {MODELS_TO_USE}")
    print(f"Runs per model per scenario: {RUNS_PER_MODEL_PER_SCENARIO}")
    print(f"Total pairs to generate: {total_expected}")
    if initial_completed > 0:
        print(f"Already completed: {initial_completed} pairs (resuming)")
    print(f"Running {len(MODELS_TO_USE)} models in PARALLEL")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    # Launch parallel workers (one per model)
    with ThreadPoolExecutor(max_workers=len(MODELS_TO_USE)) as executor:
        futures = {
            executor.submit(process_model, model, scenarios, completed_set, total_expected): model
            for model in MODELS_TO_USE
        }

        # Wait for all to complete
        for future in as_completed(futures):
            model = futures[future]
            try:
                model_pairs = future.result()
                all_pairs.extend(model_pairs)
            except Exception as e:
                print(f"[{model}] FATAL ERROR: {e}")
                sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully generated: {len(all_pairs)} pairs")
    print(f"Failed: {failed_counter.value} pairs")
    print(f"Success rate: {100 * len(all_pairs) / total_expected:.1f}%")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    print(f"\nPreparing final dataset files...")

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
    merged_progress = OUTPUT_DIR / "generation_progress_merged.jsonl"

    with open(train_path, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')

    with open(test_path, 'w') as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + '\n')

    # Save merged progress file
    with open(merged_progress, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {merged_progress} (all pairs merged)")
    print(f"\nPer-model progress files:")
    for model in MODELS_TO_USE:
        model_file = OUTPUT_DIR / f"generation_progress_{model}.jsonl"
        if model_file.exists():
            count = sum(1 for _ in open(model_file))
            print(f"  - {model_file} ({count} pairs)")

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
    # Check for API keys and remove unavailable models
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set. Claude models will not be available.")
        MODELS_TO_USE = [m for m in MODELS_TO_USE if not m.startswith("claude")]

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. GPT models will not be available.")
        MODELS_TO_USE = [m for m in MODELS_TO_USE if not m.startswith("gpt")]

    if not GEMINI_API_KEY or not GEMINI_AVAILABLE:
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set. Gemini will not be available.")
        if not GEMINI_AVAILABLE:
            print("Warning: google-generativeai not installed. Gemini will not be available.")
            print("Install with: pip install google-generativeai")
        MODELS_TO_USE = [m for m in MODELS_TO_USE if m != "gemini"]

    if not MODELS_TO_USE:
        print("Error: No API keys available. Please set at least one of:")
        print("  - ANTHROPIC_API_KEY (for Claude)")
        print("  - OPENAI_API_KEY (for GPT)")
        print("  - GEMINI_API_KEY (for Gemini)")
        exit(1)

    print(f"Available models: {MODELS_TO_USE}")
    print(f"Total pairs to generate: {len(MODELS_TO_USE) * RUNS_PER_MODEL_PER_SCENARIO * 5} (5 scenarios)")
    print(f"Running in PARALLEL mode (5x faster!)")
    print()

    generate_dataset()
