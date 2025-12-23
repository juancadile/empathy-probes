"""
Regenerate Figure 3 as two separate, cleaner graphs:
1. AUROC by layer
2. Accuracy by layer
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from original figure
layers = np.array([8, 12, 16, 20, 24])
auroc_scores = np.array([0.94, 1.0, 0.96, 0.84, 0.76])
accuracy_scores = np.array([0.93, 1.0, 0.97, 0.87, 0.80])

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create two separate figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: AUROC
ax1.plot(layers, auroc_scores, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Perfect discrimination')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random chance')
ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax1.set_title('Detection Performance by Layer', fontsize=13, fontweight='bold')
ax1.set_ylim(0.45, 1.05)
ax1.set_xticks(layers)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower left', fontsize=9)

# Add value labels
for i, (x, y) in enumerate(zip(layers, auroc_scores)):
    ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                xytext=(0,10), ha='center', fontsize=9)

# Plot 2: Accuracy
ax2.plot(layers, accuracy_scores, 's-', linewidth=2, markersize=8, color='#A23B72')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Perfect accuracy')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random chance')
ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Classification Accuracy by Layer', fontsize=13, fontweight='bold')
ax2.set_ylim(0.45, 1.05)
ax2.set_xticks(layers)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left', fontsize=9)

# Add value labels
for i, (x, y) in enumerate(zip(layers, accuracy_scores)):
    ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure3_separated.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_separated.png', dpi=300, bbox_inches='tight')
print("Saved separated Figure 3")
plt.close()

# Also create a single AUROC-only figure as the main one
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(layers, auroc_scores, 'o-', linewidth=3, markersize=10, color='#2E86AB', label='Phi-3-mini-4k')
ax.fill_between(layers, 0.5, auroc_scores, alpha=0.2, color='#2E86AB')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
ax.set_title('Empathy Detection Performance Across Layers', fontsize=15, fontweight='bold')
ax.set_ylim(0.45, 1.05)
ax.set_xticks(layers)
ax.grid(True, alpha=0.3)

# Add value labels with emphasis on layer 12
for i, (x, y) in enumerate(zip(layers, auroc_scores)):
    if x == 12:
        ax.annotate(f'{y:.1f} (perfect)', (x, y), textcoords="offset points",
                   xytext=(0,12), ha='center', fontsize=11, fontweight='bold', color='#2E86AB')
    else:
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10)

# Add text annotations
ax.text(0.98, 0.02, 'Random baseline', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, color='red', alpha=0.7)
ax.text(0.98, 0.95, 'Perfect discrimination', transform=ax.transAxes,
        ha='right', va='top', fontsize=10, color='gray', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/figure3_auroc_clean.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_auroc_clean.png', dpi=300, bbox_inches='tight')
print("Saved clean AUROC figure")
plt.close()

print("Done!")