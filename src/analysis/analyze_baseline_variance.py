"""
Analyze baseline variance in steering experiments.

Checks if model outputs are already highly varied at baseline (alpha=0),
which would explain why steering shows minimal effects.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def detect_format(text: str) -> str:
    """Categorize output format based on content patterns."""
    text_lower = text.lower()

    # Check for different format indicators
    has_json = bool(re.search(r'\{[^}]*:', text))
    has_table = '|' in text or '```' in text or re.search(r'\n\s*[-=]{3,}', text)
    has_rules = any(word in text_lower for word in ['rule', 'constraint', 'guideline', 'regulation'])
    has_steps = any(word in text_lower for word in ['step', 'action', 'move', 'turn'])
    has_grid = any(word in text_lower for word in ['grid', 'cell', 'position', 'coordinate'])
    has_game_doc = any(word in text_lower for word in ['player a', 'player b', 'game', 'objective'])

    # Categorize with priority order
    if has_json and text.count('{') > 1:
        return 'json_schema'
    elif has_table:
        return 'table/grid'
    elif has_rules and has_steps:
        return 'rule_list'
    elif has_steps and has_grid:
        return 'action_sequence'
    elif has_game_doc:
        return 'game_documentation'
    elif len(text.split()) > 50:
        return 'narrative'
    else:
        return 'other'


def analyze_baseline_variance(results_path: Path) -> Dict:
    """Analyze output format variance at baseline (alpha=0)."""

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("BASELINE VARIANCE ANALYSIS")
    print("=" * 80)
    print()

    # Extract all baseline samples
    baseline_samples = []
    baseline_by_scenario = {}

    for exp in data['experiments']:
        layer = exp['layer']
        scenario = exp['scenario']
        scenario_key = f"layer_{layer}_{scenario}"

        for cond in exp['conditions']:
            if cond['alpha'] == 0.0:  # Baseline condition
                samples = cond['samples']
                baseline_samples.extend(samples)
                baseline_by_scenario[scenario_key] = samples

    print(f"Total baseline samples: {len(baseline_samples)}")
    print(f"Scenarios analyzed: {len(baseline_by_scenario)}")
    print()

    # Analyze overall format distribution
    all_formats = [detect_format(s) for s in baseline_samples]
    format_counts = Counter(all_formats)

    print("-" * 80)
    print("OVERALL FORMAT DISTRIBUTION (All Baseline Samples)")
    print("-" * 80)
    for fmt, count in format_counts.most_common():
        pct = 100 * count / len(all_formats)
        print(f"  {fmt:20s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Calculate variance metric
    total = len(all_formats)
    max_possible_entropy = -sum((1/len(format_counts)) * (1/len(format_counts))
                                 for _ in format_counts)  # log not needed for relative comparison
    actual_entropy = -sum((count/total) * (count/total) for count in format_counts.values())
    variance_score = actual_entropy / max_possible_entropy if max_possible_entropy > 0 else 0

    print(f"Format variance score: {variance_score:.3f} (1.0 = maximally varied, 0.0 = uniform)")
    print()

    # Analyze by scenario
    print("-" * 80)
    print("FORMAT DISTRIBUTION BY SCENARIO")
    print("-" * 80)

    scenario_variance = {}
    for scenario_key, samples in baseline_by_scenario.items():
        formats = [detect_format(s) for s in samples]
        counts = Counter(formats)

        print(f"\n{scenario_key}:")
        for fmt, count in counts.most_common():
            pct = 100 * count / len(formats)
            print(f"  {fmt:20s}: {count:2d} ({pct:5.1f}%)")

        # Scenario-level variance
        total = len(formats)
        unique_formats = len(counts)
        scenario_variance[scenario_key] = unique_formats / total if total > 0 else 0

    print()
    print("-" * 80)
    print("SCENARIO VARIANCE SCORES")
    print("-" * 80)
    for scenario, score in sorted(scenario_variance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario:30s}: {score:.3f}")

    # Sample outputs for inspection
    print()
    print("-" * 80)
    print("SAMPLE BASELINE OUTPUTS (First 150 chars)")
    print("-" * 80)

    for fmt in format_counts.keys():
        matching = [s for s in baseline_samples if detect_format(s) == fmt]
        if matching:
            print(f"\n[{fmt}]")
            sample = matching[0][:150]
            print(f"  {sample}...")

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique formats: {len(format_counts)}")
    print(f"Most common format: {format_counts.most_common(1)[0][0]} ({format_counts.most_common(1)[0][1]} samples)")
    print(f"Overall variance: {variance_score:.3f}")
    print()

    if variance_score > 0.7:
        print("⚠️  HIGH VARIANCE: Baseline outputs are highly varied.")
        print("    This suggests model confusion, not steering effects.")
    elif variance_score > 0.4:
        print("⚠️  MODERATE VARIANCE: Baseline shows some format diversity.")
        print("    Steering effects may be partially masked by baseline variance.")
    else:
        print("✅ LOW VARIANCE: Baseline outputs are relatively consistent.")
        print("   Steering effects should be detectable if present.")

    print()

    # Return structured results
    return {
        "total_baseline_samples": len(baseline_samples),
        "unique_formats": len(format_counts),
        "format_distribution": dict(format_counts),
        "variance_score": variance_score,
        "scenario_variance": scenario_variance,
        "interpretation": "high" if variance_score > 0.7 else "moderate" if variance_score > 0.4 else "low"
    }


def compare_alpha_distributions(results_path: Path):
    """Compare format distributions across different alpha values."""

    with open(results_path, 'r') as f:
        data = json.load(f)

    print()
    print("=" * 80)
    print("FORMAT DISTRIBUTION BY ALPHA VALUE")
    print("=" * 80)
    print()

    # Group by alpha
    by_alpha = {}
    for exp in data['experiments']:
        for cond in exp['conditions']:
            alpha = cond['alpha']
            if alpha not in by_alpha:
                by_alpha[alpha] = []
            by_alpha[alpha].extend(cond['samples'])

    # Analyze each alpha
    alpha_results = {}
    for alpha in sorted(by_alpha.keys()):
        samples = by_alpha[alpha]
        formats = [detect_format(s) for s in samples]
        counts = Counter(formats)

        print(f"\nAlpha = {alpha:+.1f} ({len(samples)} samples):")
        for fmt, count in counts.most_common():
            pct = 100 * count / len(formats)
            print(f"  {fmt:20s}: {count:3d} ({pct:5.1f}%)")

        alpha_results[alpha] = dict(counts)

    print()
    print("-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    # Check if distribution changes with alpha
    baseline_dist = set(alpha_results.get(0.0, {}).keys())
    other_dists = [set(alpha_results[a].keys()) for a in alpha_results if a != 0.0]

    if all(d == baseline_dist for d in other_dists):
        print("⚠️  Format types are IDENTICAL across all alphas.")
        print("    Steering affects neither format choice nor content.")
    else:
        print("✅ Format types VARY across alphas.")
        print("   Steering may affect output mode (but not necessarily empathy).")

    return alpha_results


def main():
    """Main analysis function."""

    results_path = RESULTS_DIR / "steering_comprehensive.json"

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("   Run steering_comprehensive.py first!")
        return

    print(f"Loading results from: {results_path}")
    print()

    # Baseline variance analysis
    baseline_analysis = analyze_baseline_variance(results_path)

    # Alpha comparison
    alpha_comparison = compare_alpha_distributions(results_path)

    # Save analysis results
    output_path = RESULTS_DIR / "baseline_variance_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "baseline_analysis": baseline_analysis,
            "alpha_comparison": {str(k): v for k, v in alpha_comparison.items()}
        }, f, indent=2)

    print()
    print("=" * 80)
    print(f"Analysis saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
