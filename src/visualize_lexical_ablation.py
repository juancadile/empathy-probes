"""
Visualize lexical ablation results: Show probe robustness to keyword removal.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load lexical ablation results
with open(RESULTS_DIR / "lexical_ablation.json", 'r') as f:
    data = json.load(f)

layer_data = data["layers"]["12"]
original = layer_data["original"]
ablated = layer_data["ablated"]

# Metrics to compare
metrics = ["auroc", "accuracy", "separation"]
labels = ["AUROC", "Accuracy", "Separation"]

original_values = [original["auroc"], original["accuracy"], original["separation"]]
ablated_values = [ablated["auroc"], ablated["accuracy"], ablated["separation"]]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

x = np.arange(len(metrics))
width = 0.35

# Plot bars
bars1 = ax.bar(x - width/2, original_values, width, label='Original Text',
               color='#2E86AB', alpha=0.9)
bars2 = ax.bar(x + width/2, ablated_values, width, label='Keywords Removed',
               color='#A23B72', alpha=0.9)

# Customize
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Lexical Ablation: Probe Robustness to Keyword Removal',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=9, loc='upper left')  # Moved from upper right to upper left
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(original_values + ablated_values) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height < 2:  # AUROC/Accuracy (0-1 scale)
            label = f'{height:.2f}'
        else:  # Separation (larger values)
            label = f'{height:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9)

# Add annotation
ax.text(0.98, 0.02,
        f'41 keywords removed (avg 13.5/pair)\nAUROC unchanged: 1.0 → 1.0',
        transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
        horizontalalignment='right', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save
output_path = FIGURES_DIR / "figure5_lexical_ablation.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# Also save PNG
output_png = FIGURES_DIR / "figure5_lexical_ablation.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_png}")

plt.close()
