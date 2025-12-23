#!/usr/bin/env python3
"""
Build consolidated V2 dataset from all generation_progress files.
Creates train/val/test splits and summary statistics.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import random
from datetime import datetime

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "contrastive_pairs"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "v2_consolidated"
SEED = 42

# Target: 500 pairs per model (take first 500 if more)
TARGET_PER_MODEL = 500

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Models to include (excluding temp variants and duplicates)
INCLUDE_MODELS = [
    "claude-sonnet",
    "claude-haiku",
    "gpt4o",
    "gpt-5.1",
    "gemini-2.5-flash",
    "llama-70b",
    "gemma-27b",
    "qwen-32b",
]


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def normalize_model_name(filename: str) -> str:
    """Extract model name from filename."""
    # generation_progress_claude-sonnet.jsonl -> claude-sonnet
    name = filename.replace("generation_progress_", "").replace(".jsonl", "")
    return name


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Remove duplicate pairs based on empathic_text."""
    seen = set()
    unique = []
    for p in pairs:
        key = p["empathic_text"][:200]  # First 200 chars as key
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    random.seed(SEED)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building V2 Consolidated Dataset")
    print("=" * 60)

    # Load all data
    all_pairs = []
    stats_by_model = {}
    stats_by_scenario = defaultdict(int)

    for model_name in INCLUDE_MODELS:
        filepath = DATA_DIR / f"generation_progress_{model_name}.jsonl"

        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue

        pairs = load_jsonl(filepath)
        pairs = deduplicate_pairs(pairs)

        # Take first TARGET_PER_MODEL pairs
        pairs = pairs[:TARGET_PER_MODEL]

        # Add standardized model field
        for p in pairs:
            p["model_family"] = model_name
            stats_by_scenario[p.get("scenario_id", "unknown")] += 1

        stats_by_model[model_name] = len(pairs)
        all_pairs.extend(pairs)

        print(f"  {model_name}: {len(pairs)} pairs")

    print(f"\nTotal pairs: {len(all_pairs)}")

    # Shuffle
    random.shuffle(all_pairs)

    # Create splits
    n = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    print(f"\nSplits:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/n*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/n*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/n*100:.1f}%)")

    # Save splits
    def save_jsonl(data: list[dict], filepath: Path):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    save_jsonl(all_pairs, OUTPUT_DIR / "all_pairs.jsonl")
    save_jsonl(train_pairs, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_pairs, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_pairs, OUTPUT_DIR / "test.jsonl")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": SEED,
        "total_pairs": len(all_pairs),
        "splits": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "pairs_by_model": stats_by_model,
        "pairs_by_scenario": dict(stats_by_scenario),
        "models_included": INCLUDE_MODELS,
        "target_per_model": TARGET_PER_MODEL,
    }

    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\nPairs by scenario:")
    for scenario, count in sorted(stats_by_scenario.items(), key=lambda x: -x[1]):
        print(f"  {scenario}: {count}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
