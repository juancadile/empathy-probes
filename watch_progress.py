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

# Target models and counts (500 pairs each)
TARGET_MODELS = {
    "claude-sonnet": ("Claude Sonnet-4", 500),
    "claude-haiku": ("Claude Haiku", 500),
    "gpt-5.1": ("GPT-5.1", 500),
    "gpt-5-mini": ("GPT-5-mini", 500),
    "gemini-2.5-flash": ("Gemini 2.5 Flash", 500),
}
TOTAL_TARGET = 2500


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

            print("Per-Model Progress (with temperature breakdown):")
            print("-" * 90)

            total_pairs = 0
            for source_model, (display_name, target) in TARGET_MODELS.items():
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

            # Show any unexpected models (debugging)
            unexpected = {m: c for m, c in model_counts.items()
                         if m not in TARGET_MODELS and c > 0}
            if unexpected:
                print()
                print("Unexpected models found (excluded from counts):")
                for model, count in unexpected.items():
                    temp_breakdown = format_temp_breakdown(model_temp_counts.get(model, {}))
                    print(f"  - {model}: {count} pairs{temp_breakdown}")

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("=" * 90)


if __name__ == "__main__":
    main()
