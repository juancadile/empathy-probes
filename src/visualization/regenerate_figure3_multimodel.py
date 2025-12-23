"""
Regenerate Figure 3 to show AUROC curves for all three models
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for all three models
layers = np.array([8, 12, 16, 20, 24])

# Phi-3 data (from original figure)
phi3_auroc = np.array([0.991, 1.000, 0.996, 0.973, 0.960])

# Qwen2.5 data (from Table 1)
qwen_auroc = np.array([0.998, 0.999, 1.000, 0.985, 0.972])

# Dolphin data (from Table 1)
dolphin_auroc = np.array([1.000, 0.996, 0.993, 0.981, 0.968])

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Plot all three models with transparency for overlapping
ax.plot(layers, phi3_auroc, 'o-', linewidth=2.5, markersize=9,
        color='#2E86AB', label='Phi-3-mini-4k', alpha=0.8)
ax.fill_between(layers, 0.5, phi3_auroc, alpha=0.15, color='#2E86AB')

ax.plot(layers, qwen_auroc, 's-', linewidth=2.5, markersize=9,
        color='#A23B72', label='Qwen2.5-7B', alpha=0.8)
ax.fill_between(layers, 0.5, qwen_auroc, alpha=0.15, color='#A23B72')

ax.plot(layers, dolphin_auroc, '^-', linewidth=2.5, markersize=9,
        color='#F18F01', label='Dolphin-Llama-3.1', alpha=0.8)
ax.fill_between(layers, 0.5, dolphin_auroc, alpha=0.15, color='#F18F01')

# Reference lines
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Formatting
ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
ax.set_title('Empathy Detection Performance Across All Models', fontsize=15, fontweight='bold')
ax.set_ylim(0.45, 1.05)
ax.set_xticks(layers)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

# Add annotations for perfect scores
perfect_points = [
    (12, phi3_auroc[1], 'Phi-3'),
    (16, qwen_auroc[2], 'Qwen'),
    (8, dolphin_auroc[0], 'Dolphin')
]

for layer, score, model in perfect_points:
    if score == 1.0:
        ax.annotate(f'{model}\n1.00', (layer, score),
                   textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=9,
                   fontweight='bold', alpha=0.8)

# Add text annotations
ax.text(0.98, 0.02, 'Random baseline', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, color='red', alpha=0.7)
ax.text(0.98, 0.95, 'Perfect discrimination', transform=ax.transAxes,
        ha='right', va='top', fontsize=10, color='gray', alpha=0.7)

plt.tight_layout()

# Save
plt.savefig('figures/figure3_auroc_multimodel.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_auroc_multimodel.png', dpi=300, bbox_inches='tight')
print("Saved multi-model AUROC figure")
plt.close()

print("Done!")