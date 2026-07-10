"""
Diversity-based selection for GPT-4o dataset.
Reduces 976 pairs to 500 most diverse pairs using semantic similarity.

Strategy:
1. Load all 976 GPT-4o pairs
2. Compute embeddings for empathic + non-empathic text concatenation
3. Select 500 most diverse pairs using maximal marginal relevance (MMR):
   - Start with most representative pair
   - Iteratively select pairs that are most dissimilar to already selected
   - Ensure coverage across temperature settings
4. Save selected 500 pairs to new file

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
INPUT_FILE = DATA_DIR / "generation_progress_gpt4o.jsonl"
OUTPUT_FILE = DATA_DIR / "generation_progress_gpt-4o.jsonl"  # Consolidated output

TARGET_SIZE = 500
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality


def load_pairs(file_path: Path) -> List[Dict[str, Any]]:
    """Load all pairs from file."""
    pairs = []

    # Try multiple file patterns
    patterns = [
        file_path,
        DATA_DIR / "generation_progress_gpt-4o.jsonl",
        DATA_DIR / "generation_progress_gpt4o.jsonl",
    ]

    for pattern in patterns:
        if pattern.exists():
            with open(pattern, 'r') as f:
                for line in f:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Deduplicate by (scenario_id, run_id)
    seen = set()
    unique_pairs = []
    for pair in pairs:
        key = (pair.get('scenario_id'), pair.get('run_id'))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    return unique_pairs


def compute_embeddings(pairs: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for all pairs."""
    # Concatenate empathic + non-empathic text for full representation
    texts = []
    for pair in pairs:
        empathic = pair.get('empathic_text', '')
        non_empathic = pair.get('non_empathic_text', '')
        combined = f"{empathic} [SEP] {non_empathic}"
        texts.append(combined)

    print(f"Computing embeddings for {len(texts)} pairs...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings


def select_diverse_subset(
    pairs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    target_size: int
) -> List[int]:
    """
    Select diverse subset using Maximal Marginal Relevance (MMR).

    Returns indices of selected pairs.
    """
    n_pairs = len(pairs)
    selected_indices = []
    remaining_indices = set(range(n_pairs))

    # Group pairs by temperature for coverage
    temp_groups = {}
    for i, pair in enumerate(pairs):
        temp = pair.get('temperature', 'unknown')
        if temp == 'manufacturer-default':
            temp = 0.7
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(i)

    print(f"\nTemperature distribution in source data:")
    for temp, indices in sorted(temp_groups.items()):
        print(f"  T{temp}: {len(indices)} pairs")

    # Calculate target distribution (proportional to source)
    target_per_temp = {}
    for temp, indices in temp_groups.items():
        proportion = len(indices) / n_pairs
        target_per_temp[temp] = int(target_size * proportion)

    # Adjust to exactly target_size
    diff = target_size - sum(target_per_temp.values())
    if diff > 0:
        # Add to largest group
        largest_temp = max(target_per_temp.keys(), key=lambda k: target_per_temp[k])
        target_per_temp[largest_temp] += diff

    print(f"\nTarget distribution (total {target_size}):")
    for temp, count in sorted(target_per_temp.items()):
        print(f"  T{temp}: {count} pairs")

    # Select diverse samples from each temperature group
    print(f"\nSelecting diverse subset...")

    for temp in sorted(temp_groups.keys()):
        group_indices = temp_groups[temp]
        target_count = target_per_temp[temp]

        if len(group_indices) <= target_count:
            # Take all if group is small enough
            selected_indices.extend(group_indices)
            for idx in group_indices:
                remaining_indices.discard(idx)
        else:
            # Select diverse subset from this temperature group
            group_selected = select_diverse_from_group(
                embeddings,
                group_indices,
                target_count
            )
            selected_indices.extend(group_selected)
            for idx in group_selected:
                remaining_indices.discard(idx)

        print(f"  T{temp}: Selected {min(len(group_indices), target_count)}/{len(group_indices)}")

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
        # Calculate max similarity to already selected set
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


def main():
    """Main selection process."""
    print("=" * 80)
    print("GPT-4o DIVERSITY SELECTION")
    print("=" * 80)

    # Load pairs
    print(f"\nLoading pairs from multiple sources...")
    pairs = load_pairs(INPUT_FILE)
    print(f"Loaded {len(pairs)} unique pairs")

    if len(pairs) <= TARGET_SIZE:
        print(f"\n✓ Already at or below target size ({len(pairs)} <= {TARGET_SIZE})")
        print("No selection needed!")
        return

    # Show temperature distribution
    temp_counts = Counter(
        0.7 if p.get('temperature') == 'manufacturer-default' else p.get('temperature', 'unknown')
        for p in pairs
    )
    print(f"\nCurrent temperature distribution:")
    for temp, count in sorted(temp_counts.items()):
        print(f"  T{temp}: {count} pairs")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    embeddings = compute_embeddings(pairs, model)

    # Select diverse subset
    selected_indices = select_diverse_subset(pairs, embeddings, TARGET_SIZE)
    selected_pairs = [pairs[i] for i in selected_indices]

    # Update source_model to consolidated name
    for pair in selected_pairs:
        pair['source_model'] = 'gpt-4o'

    # Save selected pairs
    print(f"\nSaving {len(selected_pairs)} selected pairs to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        for pair in selected_pairs:
            f.write(json.dumps(pair) + '\n')

    # Show final distribution
    final_temp_counts = Counter(
        0.7 if p.get('temperature') == 'manufacturer-default' else p.get('temperature', 'unknown')
        for p in selected_pairs
    )
    print(f"\nFinal temperature distribution:")
    for temp, count in sorted(final_temp_counts.items()):
        pct = (count / len(selected_pairs)) * 100
        print(f"  T{temp}: {count} pairs ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Reduced from {len(pairs)} to {len(selected_pairs)} pairs")
    print(f"✓ Maintained temperature diversity")
    print(f"✓ Selected most semantically diverse examples")


if __name__ == "__main__":
    main()
