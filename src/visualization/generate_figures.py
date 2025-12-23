"""
Generate publication-quality figures for paper.

Creates:
1. AUROC by layer (bar chart)
2. Random baseline distribution vs empathy probe
3. EIA correlation scatter plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_results():
    """Load all result files."""
    with open(RESULTS_DIR / "validation_auroc.json") as f:
        validation = json.load(f)

    with open(RESULTS_DIR / "random_baseline_proper.json") as f:
        random_baseline = json.load(f)

    with open(RESULTS_DIR / "eia_correlation.json") as f:
        eia = json.load(f)

    return validation, random_baseline, eia

def figure1_auroc_by_layer(validation):
    """Figure 1: AUROC by layer with error bars."""
    layers = []
    aurocs = []
    accs = []

    for layer_str, data in validation['layer_results'].items():
        layers.append(int(layer_str))
        aurocs.append(data['auroc'])
        accs.append(data['accuracy'])

    # Sort by layer
    sorted_data = sorted(zip(layers, aurocs, accs))
    layers, aurocs, accs = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, aurocs, width, label='AUROC', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, accs, width, label='Accuracy', alpha=0.8, color='#A23B72')

    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1, label='Target (0.75)', alpha=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('Probe Performance by Layer (N=15 test pairs)')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(loc='lower left')
    ax.set_ylim([0.7, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Highlight best layer
    best_idx = aurocs.index(1.0)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_auroc_by_layer.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure1_auroc_by_layer.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved Figure 1: AUROC by layer")
    plt.close()

def figure2_random_baseline(random_baseline):
    """Figure 2: Random baseline distribution vs empathy probe."""
    random_aurocs = random_baseline['random_baseline']['all_aurocs']
    empathy_auroc = random_baseline['empathy_probe']['auroc']

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram
    ax.hist(random_aurocs, bins=20, alpha=0.6, color='gray', edgecolor='black', label='Random directions (N=100)')

    # Statistics
    mean_random = random_baseline['random_baseline']['mean_auroc']
    percentile_95 = random_baseline['random_baseline']['95th_percentile']

    # Vertical lines
    ax.axvline(mean_random, color='blue', linestyle='--', linewidth=2, label=f'Random mean ({mean_random:.3f})')
    ax.axvline(percentile_95, color='orange', linestyle='--', linewidth=2, label=f'95th percentile ({percentile_95:.3f})')
    ax.axvline(empathy_auroc, color='red', linestyle='-', linewidth=3, label=f'Empathy probe ({empathy_auroc:.3f})')

    ax.set_xlabel('AUROC')
    ax.set_ylabel('Frequency')
    ax.set_title('Random Baseline Validation (Layer 12, dim=3072)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add z-score annotation
    z_score = random_baseline['comparison']['z_score']
    ax.text(0.98, 0.95, f'z = {z_score:.2f}σ\np < 0.05',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_random_baseline.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure2_random_baseline.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved Figure 2: Random baseline distribution")
    plt.close()

def figure3_eia_correlation(eia):
    """Figure 3: Probe projection vs EIA behavioral scores."""
    # Extract from detailed_results
    probe_scores = [item['probe_score'] for item in eia['detailed_results']]
    eia_scores = [item['true_score'] for item in eia['detailed_results']]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot
    colors = ['#E63946' if s == 0 else '#F77F00' if s == 1 else '#06D6A0' for s in eia_scores]
    ax.scatter(eia_scores, probe_scores, c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Linear regression line
    z = np.polyfit(eia_scores, probe_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(eia_scores), max(eia_scores), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label='Linear fit')

    # Labels
    ax.set_xlabel('EIA Behavioral Score')
    ax.set_ylabel('Probe Projection')
    ax.set_title('Behavioral Correlation (Layer 12)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Non-empathic\n(0)', 'Moderate\n(1)', 'Empathic\n(2)'])
    ax.grid(alpha=0.3)

    # Add correlation stats
    pearson_r = eia['pearson_correlation']
    p_value = eia['pearson_p_value']
    binary_acc = eia['binary_accuracy']

    stats_text = f"Pearson r = {pearson_r:.3f} (p = {p_value:.3f})\nBinary accuracy = {binary_acc:.1%}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', label='Non-empathic (0)', alpha=0.7),
        Patch(facecolor='#F77F00', label='Moderate (1)', alpha=0.7),
        Patch(facecolor='#06D6A0', label='Empathic (2)', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_eia_correlation.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure3_eia_correlation.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved Figure 3: EIA correlation")
    plt.close()

def main():
    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    validation, random_baseline, eia = load_results()

    figure1_auroc_by_layer(validation)
    figure2_random_baseline(random_baseline)
    figure3_eia_correlation(eia)

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print("\nFiles created:")
    for f in sorted(FIGURES_DIR.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
