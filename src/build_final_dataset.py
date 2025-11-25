"""
Build final dataset: 500 pairs per model across 5 models (2,500 total).

Models:
1. Claude Sonnet-4: Select 500 from 772 (522 @ T0.7 + 250 mixed temps)
2. Claude Haiku: Select 500 from 968 (718 @ T0.7 + 250 mixed temps)
3. GPT-4o: Select 500 from 1103 (810 @ T0.7 + 293 @ T0.8/0.9/1.0)
4. GPT-5.1: Use all 500 (already at target)
5. Gemini 2.5 Flash: Use all 500 (already at target)

Strategy:
- For models with excess pairs, use semantic diversity selection
- Maintain representative temperature distribution with slight skew towards T0.9
- Target distribution for selected 500:
  * T0.7: ~40% (200 pairs) - quality baseline
  * T0.9: ~35% (175 pairs) - high diversity, skewed up from proportional
  * T0.8: ~15% (75 pairs)
  * T1.0: ~10% (50 pairs)

Requirements:
    pip install sentence-transformers scikit-learn numpy
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Target distribution for 500 pairs (skewed towards T0.9)
TARGET_DISTRIBUTION = {
    0.7: 200,  # 40% - quality baseline
    0.9: 175,  # 35% - high diversity (skewed up)
    0.8: 75,   # 15%
    1.0: 50,   # 10%
}

# Model configurations
MODELS_TO_SELECT = [
    {
        "name": "claude-sonnet",
        "display": "Claude Sonnet-4",
        "files": [
            DATA_DIR / "generation_progress_claude-sonnet.jsonl",
            DATA_DIR / "generation_progress_claude-sonnet_temp09.jsonl"
        ],
        "target": 500
    },
    {
        "name": "claude-haiku",
        "display": "Claude Haiku",
        "files": [
            DATA_DIR / "generation_progress_claude-haiku.jsonl",
            DATA_DIR / "generation_progress_claude-haiku_temp09.jsonl"
        ],
        "target": 500
    },
    {
        "name": "gpt-4o",
        "display": "GPT-4o",
        "files": [
            DATA_DIR / "generation_progress_gpt4o.jsonl",
            DATA_DIR / "generation_progress_gpt-4o.jsonl"
        ],
        "target": 500
    }
]

MODELS_READY = [
    {
        "name": "gpt-5.1",
        "display": "GPT-5.1",
        "file": DATA_DIR / "generation_progress_gpt-5.1.jsonl"
    },
    {
        "name": "gemini-2.5-flash",
        "display": "Gemini 2.5 Flash",
        "file": DATA_DIR / "generation_progress_gemini-2.5-flash.jsonl"
    }
]


def load_pairs(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load and deduplicate pairs from multiple files."""
    pairs = []
    seen = set()

    for file_path in file_paths:
        if not file_path.exists():
            print(f"  ⚠️  File not found: {file_path.name}")
            continue

        with open(file_path, 'r') as f:
            for line in f:
                try:
                    pair = json.loads(line)

                    # Normalize temperature
                    temp = pair.get('temperature', 'manufacturer-default')
                    if temp == 'manufacturer-default':
                        pair['temperature'] = 0.7

                    # Deduplicate by (scenario_id, run_id, temperature)
                    # This ensures pairs with same scenario/run but different temps are kept
                    key = (pair.get('scenario_id'), pair.get('run_id'), pair['temperature'])
                    if key not in seen:
                        seen.add(key)
                        pairs.append(pair)
                except json.JSONDecodeError:
                    continue

    return pairs


def compute_embeddings(pairs: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for all pairs."""
    texts = []
    for pair in pairs:
        empathic = pair.get('empathic_text', '')
        non_empathic = pair.get('non_empathic_text', '')
        combined = f"{empathic} [SEP] {non_empathic}"
        texts.append(combined)

    print(f"  Computing embeddings for {len(texts)} pairs...")
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    return embeddings


def select_diverse_with_temperature_targets(
    pairs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    target_size: int,
    target_dist: Dict[float, int]
) -> List[int]:
    """
    Select diverse subset using MMR while targeting specific temperature distribution.

    Returns indices of selected pairs.
    """
    # Group pairs by temperature
    temp_groups = {}
    for i, pair in enumerate(pairs):
        temp = pair.get('temperature', 0.7)
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(i)

    print(f"\n  Source temperature distribution:")
    for temp in sorted(temp_groups.keys()):
        print(f"    T{temp}: {len(temp_groups[temp])} pairs")

    # Calculate actual target distribution based on what's available
    actual_targets = {}
    total_requested = sum(target_dist.values())

    for temp, target_count in target_dist.items():
        available = len(temp_groups.get(temp, []))
        if available == 0:
            print(f"  ⚠️  No pairs available at T{temp}, skipping")
            continue

        # Take minimum of target and available
        actual_targets[temp] = min(target_count, available)

    # If we're short, proportionally increase from available temperatures
    total_selected = sum(actual_targets.values())
    if total_selected < target_size:
        shortfall = target_size - total_selected
        print(f"  ⚠️  Shortfall of {shortfall} pairs, redistributing...")

        # Find temps with excess capacity
        for temp in sorted(temp_groups.keys()):
            if shortfall == 0:
                break
            available = len(temp_groups[temp])
            current = actual_targets.get(temp, 0)
            can_add = available - current
            if can_add > 0:
                add = min(can_add, shortfall)
                actual_targets[temp] = actual_targets.get(temp, 0) + add
                shortfall -= add

    print(f"\n  Target distribution for selection:")
    for temp in sorted(actual_targets.keys()):
        print(f"    T{temp}: {actual_targets[temp]} pairs")

    # Select diverse samples from each temperature group
    selected_indices = []

    for temp in sorted(actual_targets.keys()):
        group_indices = temp_groups[temp]
        target_count = actual_targets[temp]

        if len(group_indices) <= target_count:
            # Take all if group is small enough
            selected_indices.extend(group_indices)
        else:
            # Select diverse subset using MMR
            group_selected = select_diverse_from_group(
                embeddings,
                group_indices,
                target_count
            )
            selected_indices.extend(group_selected)

        print(f"    T{temp}: Selected {min(len(group_indices), target_count)}/{len(group_indices)}")

    return selected_indices


def select_diverse_from_group(
    embeddings: np.ndarray,
    group_indices: List[int],
    target_count: int
) -> List[int]:
    """Select diverse subset from a temperature group using MMR."""
    if len(group_indices) <= target_count:
        return group_indices

    selected = []
    remaining = set(group_indices)

    # Start with most central (representative) example
    group_embeddings = embeddings[group_indices]
    centroid = np.mean(group_embeddings, axis=0, keepdims=True)
    similarities_to_centroid = cosine_similarity(group_embeddings, centroid).flatten()
    first_idx = group_indices[np.argmax(similarities_to_centroid)]
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Iteratively select most diverse remaining examples
    while len(selected) < target_count and remaining:
        selected_embeddings = embeddings[selected]
        remaining_list = list(remaining)
        remaining_embeddings = embeddings[remaining_list]

        # For each remaining, find max similarity to any selected
        max_similarities = cosine_similarity(
            remaining_embeddings,
            selected_embeddings
        ).max(axis=1)

        # Select the one with minimum max similarity (most diverse)
        most_diverse_idx = remaining_list[np.argmin(max_similarities)]
        selected.append(most_diverse_idx)
        remaining.remove(most_diverse_idx)

    return selected


def process_model(config: Dict[str, Any], model: SentenceTransformer) -> None:
    """Process a single model: load, select, save."""
    print(f"\n{'='*80}")
    print(f"{config['display'].upper()}")
    print(f"{'='*80}")

    # Load pairs
    print(f"\nLoading pairs from {len(config['files'])} file(s)...")
    pairs = load_pairs(config['files'])
    print(f"Loaded {len(pairs)} unique pairs")

    if len(pairs) <= config['target']:
        print(f"⚠️  Only {len(pairs)} pairs available, need {config['target']}")
        print(f"Using all available pairs")
        selected_pairs = pairs
    else:
        # Compute embeddings
        embeddings = compute_embeddings(pairs, model)

        # Select diverse subset
        selected_indices = select_diverse_with_temperature_targets(
            pairs,
            embeddings,
            config['target'],
            TARGET_DISTRIBUTION
        )
        selected_pairs = [pairs[i] for i in selected_indices]

    # Update source_model to standardized name
    for pair in selected_pairs:
        pair['source_model'] = config['name']

    # Save
    output_file = OUTPUT_DIR / f"{config['name']}_500.jsonl"
    print(f"\nSaving {len(selected_pairs)} pairs to {output_file.name}")
    with open(output_file, 'w') as f:
        for pair in selected_pairs:
            f.write(json.dumps(pair) + '\n')

    # Show final distribution
    final_temps = Counter(p.get('temperature', 'unknown') for p in selected_pairs)
    print(f"\nFinal temperature distribution:")
    for temp in sorted(final_temps.keys(), key=lambda x: (isinstance(x, str), x)):
        count = final_temps[temp]
        pct = (count / len(selected_pairs)) * 100
        print(f"  T{temp}: {count} pairs ({pct:.1f}%)")


def copy_ready_model(config: Dict[str, Any]) -> None:
    """Copy a model that's already at target size."""
    print(f"\n{'='*80}")
    print(f"{config['display'].upper()}")
    print(f"{'='*80}")

    pairs = []
    with open(config['file'], 'r') as f:
        for line in f:
            try:
                pair = json.loads(line)

                # Normalize temperature
                temp = pair.get('temperature', 'manufacturer-default')
                if temp == 'manufacturer-default':
                    pair['temperature'] = 0.7

                # Standardize source_model
                pair['source_model'] = config['name']
                pairs.append(pair)
            except json.JSONDecodeError:
                continue

    output_file = OUTPUT_DIR / f"{config['name']}_500.jsonl"
    print(f"\nCopying {len(pairs)} pairs to {output_file.name}")
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    # Show temperature distribution
    temps = Counter(p.get('temperature', 'unknown') for p in pairs)
    print(f"\nTemperature distribution:")
    for temp in sorted(temps.keys(), key=lambda x: (isinstance(x, str), x)):
        count = temps[temp]
        pct = (count / len(pairs)) * 100
        print(f"  T{temp}: {count} pairs ({pct:.1f}%)")


def main():
    """Build final dataset."""
    print("="*80)
    print(" "*25 + "FINAL DATASET BUILDER")
    print("="*80)
    print(f"\nTarget: 500 pairs per model × 5 models = 2,500 pairs total")
    print(f"\nTemperature distribution strategy:")
    print(f"  - Representative selection with slight T0.9 skew")
    print(f"  - Target: T0.7=40%, T0.9=35%, T0.8=15%, T1.0=10%")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Process models needing selection
    for config in MODELS_TO_SELECT:
        process_model(config, model)

    # Copy ready models
    for config in MODELS_READY:
        copy_ready_model(config)

    # Final summary
    print(f"\n{'='*80}")
    print(" "*25 + "BUILD COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal dataset saved to: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    for file in sorted(OUTPUT_DIR.glob("*_500.jsonl")):
        size = sum(1 for _ in open(file))
        print(f"  - {file.name}: {size} pairs")

    total = sum(1 for f in OUTPUT_DIR.glob("*_500.jsonl") for _ in open(f))
    print(f"\nTotal pairs: {total}")


if __name__ == "__main__":
    main()
