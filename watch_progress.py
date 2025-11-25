#!/usr/bin/env python3
"""
Live progress monitor for dataset generation with temperature breakdown.
Shows per-model counts and temperature distribution.
"""

import json
import time
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
DATA_DIR = Path(__file__).parent / "data" / "contrastive_pairs"
REFRESH_INTERVAL = 3  # seconds

# Models currently being generated (with temperature cycling/variation)
GENERATING_MODELS = {
    "claude-sonnet": ("Claude Sonnet-4", 250),  # New temp-cycled pairs
    "claude-haiku": ("Claude Haiku", 250),      # New temp-cycled pairs
    "gpt-5.1": ("GPT-5.1", 500),                # Cycling [0.7, 0.8, 0.9, 1.0]
    "gpt-4o": ("GPT-4o", 500),                  # Cycling [0.8, 0.9, 1.0]
    "gemini-2.5-flash": ("Gemini 2.5 Flash", 500),  # All at 0.9
}
TOTAL_TARGET = 2000  # 250 + 250 + 500 + 500 + 500


def load_all_pairs():
    """Load all pairs and count by source_model and temperature."""
    model_counts = defaultdict(int)
    model_temp_counts = defaultdict(lambda: defaultdict(int))

    # Read all progress files
    for progress_file in DATA_DIR.glob("generation_progress_*.jsonl"):
        try:
            with open(progress_file, 'r') as f:
                for line in f:
                    try:
                        pair = json.loads(line)
                        source_model = pair.get('source_model', '')
                        temperature = pair.get('temperature', 'unknown')

                        # Normalize manufacturer-default to 0.7 for display
                        if temperature == 'manufacturer-default':
                            temperature = 0.7

                        model_counts[source_model] += 1
                        model_temp_counts[source_model][temperature] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue

    return model_counts, model_temp_counts


def draw_progress_bar(current, target, width=40):
    """Draw a progress bar."""
    if target == 0:
        filled = 0
    else:
        filled = int((current / target) * width)

    bar = '=' * filled + ' ' * (width - filled)
    return f"[{bar}]"


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_temp_breakdown(temp_counts):
    """Format temperature breakdown as compact string."""
    if not temp_counts:
        return ""

    # Separate numeric and string temperatures for sorting
    numeric_temps = []
    string_temps = []

    for temp, count in temp_counts.items():
        if isinstance(temp, (int, float)):
            numeric_temps.append((temp, count))
        else:
            string_temps.append((temp, count))

    # Sort each group
    numeric_temps.sort(key=lambda x: x[0])
    string_temps.sort(key=lambda x: x[0])

    # Format output
    temps = []
    for temp, count in numeric_temps:
        temps.append(f"T{temp}:{count}")
    for temp, count in string_temps:
        temps.append(f"{temp}:{count}")

    return " | " + ", ".join(temps)


def main():
    """Main monitoring loop."""
    try:
        while True:
            clear_screen()

            # Load current counts
            model_counts, model_temp_counts = load_all_pairs()

            print("=" * 90)
            print(" " * 25 + "DATASET GENERATION LIVE PROGRESS")
            print("=" * 90)
            print()

            print("Currently Generating (with temperature breakdown):")
            print("-" * 90)

            total_pairs = 0
            for source_model, (display_name, target) in GENERATING_MODELS.items():
                count = model_counts.get(source_model, 0)
                temp_counts = model_temp_counts.get(source_model, {})
                total_pairs += count

                percent = (count / target) * 100 if target > 0 else 0
                bar = draw_progress_bar(count, target, 30)

                temp_breakdown = format_temp_breakdown(temp_counts)

                print(f"{display_name:<20s} {bar} {count:4d}/{target} ({percent:5.1f}%){temp_breakdown}")

            print()
            print("Overall Progress:")
            print("-" * 90)

            overall_percent = (total_pairs / TOTAL_TARGET) * 100 if TOTAL_TARGET > 0 else 0
            remaining = TOTAL_TARGET - total_pairs

            print(f"Total pairs generated: {total_pairs} / {TOTAL_TARGET}")
            print(f"Completion: {overall_percent:.1f}%")
            print(f"Remaining: {remaining} pairs")
            print()

            overall_bar = draw_progress_bar(total_pairs, TOTAL_TARGET, 60)
            print(f"Overall: {overall_bar} {overall_percent:5.1f}%")

            print()
            print("-" * 90)
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Press Ctrl+C to exit monitoring")
            print("=" * 90)

            # Show ready datasets (models with existing data not currently generating)
            ready_datasets = {m: c for m, c in model_counts.items()
                            if m not in GENERATING_MODELS and c > 0}
            if ready_datasets:
                print()
                print("Ready Datasets (existing data, not currently generating):")
                print("-" * 90)
                for model, count in sorted(ready_datasets.items()):
                    temp_breakdown = format_temp_breakdown(model_temp_counts.get(model, {}))
                    print(f"  {model:<30s} {count:4d} pairs{temp_breakdown}")

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("=" * 90)


if __name__ == "__main__":
    main()
