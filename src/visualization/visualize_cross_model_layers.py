"""
Visualize cross-model layer-by-layer AUROC comparison.
Shows all 3 models (Phi-3, Qwen, Dolphin) across all tested layers.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load Phi-3 results
with open(RESULTS_DIR / "validation_auroc.json", 'r') as f:
    phi3_data = json.load(f)

# Load cross-model results
with open(RESULTS_DIR / "cross_model_validation" / "all_models_results.json", 'r') as f:
    cross_model_data = json.load(f)

# Extract AUROC data
models_data = {
    "Phi-3-mini-4k": {
        "layers": [8, 12, 16, 20, 24],
        "aurocs": [
            phi3_data["layer_results"]["8"]["auroc"],
            phi3_data["layer_results"]["12"]["auroc"],
            phi3_data["layer_results"]["16"]["auroc"],
            phi3_data["layer_results"]["20"]["auroc"],
            phi3_data["layer_results"]["24"]["auroc"]
        ],
        "color": "#2E86AB",
        "marker": "o"
    },
    "Qwen2.5-7B": {
        "layers": [8, 12, 16, 20, 24],
        "aurocs": [
            cross_model_data["qwen2.5-7b"]["layers"]["8"]["auroc"],
            cross_model_data["qwen2.5-7b"]["layers"]["12"]["auroc"],
            cross_model_data["qwen2.5-7b"]["layers"]["16"]["auroc"],
            cross_model_data["qwen2.5-7b"]["layers"]["20"]["auroc"],
            cross_model_data["qwen2.5-7b"]["layers"]["24"]["auroc"]
        ],
        "color": "#A23B72",
        "marker": "s"
    },
    "Dolphin-Llama-3.1-8B": {
        "layers": [8, 12, 16, 20, 24, 28],
        "aurocs": [
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["8"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["12"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["16"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["20"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["24"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["28"]["auroc"]
        ],
        "color": "#F18F01",
        "marker": "^"
    }
}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Plot each model
for model_name, data in models_data.items():
    ax.plot(data["layers"], data["aurocs"],
            marker=data["marker"], markersize=8, linewidth=2.5,
            color=data["color"], label=model_name, alpha=0.9)

# Highlight perfect AUROC threshold
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Discrimination')

# Customize
ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax.set_title('Cross-Model Empathy Detection: AUROC Across Layers',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0.75, 1.02)

# Add annotations for perfect scores
ax.annotate('Phi-3 L12\nAUROC = 1.0',
            xy=(12, 1.0), xytext=(10, 0.92),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E86AB', alpha=0.2),
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

ax.annotate('Qwen L16\nAUROC = 1.0',
            xy=(16, 1.0), xytext=(18, 0.92),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#A23B72', alpha=0.2),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5))

# Add summary box
summary_text = (
    "Architecture-Agnostic:\n"
    "• 3.8B to 8B parameters\n"
    "• 3072 to 4096 hidden dims\n"
    "• All AUROC ≥ 0.996"
)
ax.text(0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save
output_path = FIGURES_DIR / "figure6_cross_model_layers.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# Also save PNG
output_png = FIGURES_DIR / "figure6_cross_model_layers.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_png}")

plt.close()
