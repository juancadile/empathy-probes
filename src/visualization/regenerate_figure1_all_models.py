"""
Regenerate Figure 1: AUROC by layer for ALL THREE models
Shows Phi-3, Qwen, and Dolphin together
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Load Phi-3 results
with open(RESULTS_DIR / "validation_auroc.json", 'r') as f:
    phi3_data = json.load(f)

# Load cross-model results
with open(RESULTS_DIR / "cross_model_validation" / "all_models_results.json", 'r') as f:
    cross_model_data = json.load(f)

# Extract data for all three models
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
        "accuracies": [
            phi3_data["layer_results"]["8"]["accuracy"],
            phi3_data["layer_results"]["12"]["accuracy"],
            phi3_data["layer_results"]["16"]["accuracy"],
            phi3_data["layer_results"]["20"]["accuracy"],
            phi3_data["layer_results"]["24"]["accuracy"]
        ],
        "color": "#F18F01",  # Orange for Phi-3
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
        "accuracies": [
            cross_model_data["qwen2.5-7b"]["layers"]["8"]["accuracy"],
            cross_model_data["qwen2.5-7b"]["layers"]["12"]["accuracy"],
            cross_model_data["qwen2.5-7b"]["layers"]["16"]["accuracy"],
            cross_model_data["qwen2.5-7b"]["layers"]["20"]["accuracy"],
            cross_model_data["qwen2.5-7b"]["layers"]["24"]["accuracy"]
        ],
        "color": "#2E86AB",  # Blue for Qwen
        "marker": "s"
    },
    "Dolphin-Llama-3.1-8B": {
        "layers": [8, 12, 16, 20, 24],
        "aurocs": [
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["8"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["12"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["16"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["20"]["auroc"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["24"]["auroc"]
        ],
        "accuracies": [
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["8"]["accuracy"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["12"]["accuracy"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["16"]["accuracy"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["20"]["accuracy"],
            cross_model_data["dolphin-llama-3.1-8b"]["layers"]["24"]["accuracy"]
        ],
        "color": "#A23B72",  # Purple for Dolphin
        "marker": "^"
    }
}

# Create figure with dual y-axis
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

# Plot AUROC for all models
for model_name, data in models_data.items():
    ax1.plot(data["layers"], data["aurocs"],
            marker=data["marker"], markersize=8, linewidth=2.5,
            color=data["color"], label=f'{model_name} (AUROC)', alpha=0.9)

# Add perfect AUROC line
ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Perfect (1.0)')

# Add target AUROC line
ax1.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Target (0.75)')

# Customize
ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Probe Performance by Layer (N=15 test pairs)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='lower left', framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0.70, 1.05)
ax1.set_xticks([8, 12, 16, 20, 24])

plt.tight_layout()

# Save
output_path = FIGURES_DIR / "figure1_auroc_by_layer.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

output_png = FIGURES_DIR / "figure1_auroc_by_layer.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_png}")

plt.close()
