#!/usr/bin/env python3
"""
Dataset status viewer with model consolidation.
Shows final dataset composition with temperature distributions.
Combines related models (e.g., gpt-4o variants) for final analysis.
"""

import json
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_DIR = Path(__file__).parent / "data" / "contrastive_pairs"

# Model consolidation mapping: file model name -> display name
MODEL_CONSOLIDATION = {
    # Claude models (will select most diverse 250 from ~500-700 manufacturer-default + 250 temp-cycled)
    "claude-sonnet": "Claude Sonnet-4",
    "claude-haiku": "Claude Haiku",

    # GPT-4o variants (combine all into one)
    "gpt-4o": "GPT-4o",
    "gpt4o": "GPT-4o",  # Merge with gpt-4o

    # GPT-4o-mini (was labeled as chatgpt-4o-latest)
    "gpt4o-mini": "GPT-4o-mini",
    "chatgpt-4o-latest": "GPT-4o-mini",

    # GPT-5.1 (temperature cycling)
    "gpt-5.1": "GPT-5.1",

    # Gemini
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}

# Target dataset sizes (after selection/mixing)
TARGET_SIZES = {
    "Claude Sonnet-4": 500,
    "Claude Haiku": 500,
    "GPT-4o": 500,
    "GPT-4o-mini": 500,
    "GPT-5.1": 500,
    "Gemini 2.5 Flash": 500,
}


def load_all_pairs():
    """Load all pairs and consolidate by model."""
    model_counts = defaultdict(int)
    model_temp_counts = defaultdict(lambda: defaultdict(int))
    model_scenarios = defaultdict(set)

    # Read all progress files
    for progress_file in DATA_DIR.glob("generation_progress_*.jsonl"):
        try:
            with open(progress_file, 'r') as f:
                for line in f:
                    try:
                        pair = json.loads(line)
                        source_model = pair.get('source_model', '')
                        temperature = pair.get('temperature', 'unknown')
                        scenario_id = pair.get('scenario_id', '')

                        # Normalize manufacturer-default to 0.7
                        if temperature == 'manufacturer-default':
                            temperature = 0.7

                        # Consolidate model name
                        display_model = MODEL_CONSOLIDATION.get(source_model, source_model)

                        model_counts[display_model] += 1
                        model_temp_counts[display_model][temperature] += 1
                        if scenario_id:
                            model_scenarios[display_model].add(scenario_id)

                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue

    return model_counts, model_temp_counts, model_scenarios


def format_temp_breakdown(temp_counts):
    """Format temperature breakdown as detailed string."""
    if not temp_counts:
        return "No data"

    # Separate numeric and string temperatures
    numeric_temps = [(t, c) for t, c in temp_counts.items() if isinstance(t, (int, float))]
    string_temps = [(t, c) for t, c in temp_counts.items() if not isinstance(t, (int, float))]

    # Sort
    numeric_temps.sort(key=lambda x: x[0])
    string_temps.sort(key=lambda x: x[0])

    # Format
    parts = []
    for temp, count in numeric_temps:
        parts.append(f"T{temp}={count}")
    for temp, count in string_temps:
        parts.append(f"{temp}={count}")

    return ", ".join(parts)


def print_dataset_status():
    """Print comprehensive dataset status."""
    model_counts, model_temp_counts, model_scenarios = load_all_pairs()

    print("=" * 100)
    print(" " * 35 + "DATASET STATUS REPORT")
    print("=" * 100)
    print()

    # Calculate totals
    total_pairs = sum(model_counts.values())
    total_target = sum(TARGET_SIZES.values())

    print("OVERVIEW")
    print("-" * 100)
    print(f"Total pairs collected: {total_pairs:,}")
    print(f"Target dataset size: {total_target:,} pairs (500 per model × {len(TARGET_SIZES)} models)")
    print()

    print("DATASET COMPOSITION BY MODEL")
    print("-" * 100)
    print(f"{'Model':<25} {'Current':<10} {'Target':<10} {'Status':<15} {'Temperature Distribution'}")
    print("-" * 100)

    for model_name in sorted(TARGET_SIZES.keys()):
        current = model_counts.get(model_name, 0)
        target = TARGET_SIZES[model_name]
        temp_breakdown = format_temp_breakdown(model_temp_counts.get(model_name, {}))
        num_scenarios = len(model_scenarios.get(model_name, set()))

        # Determine status
        if current == 0:
            status = "⚠️  Not started"
        elif current < target:
            status = f"🔄 In progress"
        elif current == target:
            status = "✅ Complete"
        else:
            status = f"📊 {current} (needs selection)"

        print(f"{model_name:<25} {current:<10} {target:<10} {status:<15} {temp_breakdown}")

    print()
    print("TEMPERATURE DISTRIBUTION SUMMARY")
    print("-" * 100)

    # Aggregate temperature stats across all models
    all_temp_counts = defaultdict(int)
    for temp_counts in model_temp_counts.values():
        for temp, count in temp_counts.items():
            all_temp_counts[temp] += count

    print("Overall temperature distribution:")
    for temp in sorted(all_temp_counts.keys(), key=lambda x: (isinstance(x, str), x)):
        count = all_temp_counts[temp]
        pct = (count / total_pairs * 100) if total_pairs > 0 else 0
        print(f"  T{temp if isinstance(temp, (int, float)) else temp}: {count:,} pairs ({pct:.1f}%)")

    print()
    print("NEXT STEPS")
    print("-" * 100)

    # Determine what needs to be done
    needs_generation = []
    needs_selection = []
    ready = []

    for model_name, target in sorted(TARGET_SIZES.items()):
        current = model_counts.get(model_name, 0)
        if current == 0:
            needs_generation.append(f"  • {model_name}: Generate {target} pairs")
        elif current < target:
            needs_generation.append(f"  • {model_name}: Generate {target - current} more pairs")
        elif current > target:
            needs_selection.append(f"  • {model_name}: Select {target} most diverse from {current} pairs")
        else:
            ready.append(f"  • {model_name}: Ready ({current} pairs)")

    if needs_generation:
        print("\n📝 GENERATION NEEDED:")
        for item in needs_generation:
            print(item)

    if needs_selection:
        print("\n🎯 DIVERSITY SELECTION NEEDED:")
        for item in needs_selection:
            print(item)
        print("\n   Strategy: Use semantic similarity to select most diverse subset")
        print("   - Calculate pairwise cosine similarity of text embeddings")
        print("   - Select pairs that maximize diversity (min similarity to selected set)")
        print("   - Prioritize coverage across temperature settings")

    if ready:
        print("\n✅ READY FOR TRAINING:")
        for item in ready:
            print(item)

    print()
    print("=" * 100)
    print(f"Report generated: {Path.cwd().name}")
    print("=" * 100)


if __name__ == "__main__":
    print_dataset_status()
