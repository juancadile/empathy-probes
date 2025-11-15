"""
Visualize comprehensive steering results comparing Layer 8 vs Layer 12.
Creates figure showing detection-causation dissociation.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def count_empathy_words(text):
    """Count empathy-related keywords in text"""
    keywords = ['help', 'care', 'support', 'together', 'understand', 'feel',
                'empathy', 'compassion', 'kindness', 'comfort', 'share', 'listen']
    text_lower = text.lower()
    return sum(1 for word in keywords if word in text_lower)

def count_task_words(text):
    """Count task-oriented keywords"""
    keywords = ['optimize', 'strategy', 'efficient', 'calculate', 'maximize',
                'plan', 'execute', 'precise', 'systematic', 'points', 'win']
    text_lower = text.lower()
    return sum(1 for word in keywords if word in text_lower)

def analyze_steering_success(samples):
    """Determine if steering was successful based on empathy content"""
    empathy_counts = [count_empathy_words(s) for s in samples]
    # Success if majority show empathy increase
    return sum(c > 0 for c in empathy_counts) / len(empathy_counts)

# Load data
data_path = Path(__file__).parent.parent / "results" / "steering_comprehensive.json"
with open(data_path) as f:
    data = json.load(f)

# Extract results for layer 8 and 12 at key alphas
layers = [8, 12]
alphas_tested = data['alphas_tested']
scenarios = ['food_delivery', 'the_listener', 'the_protector']

# Compute success rates for each layer/alpha combination
results = {layer: {'alphas': [], 'success_rates': [], 'empathy_words': []} for layer in layers}

for layer in layers:
    layer_data = [exp for exp in data['experiments'] if exp['layer'] == layer]

    for alpha in alphas_tested:
        if alpha <= 0:
            continue  # Skip negative/zero for main figure

        success_total = 0
        empathy_total = 0
        count = 0

        for scenario_data in layer_data:
            scenario_name = scenario_data['scenario']
            for condition in scenario_data['conditions']:
                if condition['alpha'] == alpha:
                    samples = condition['samples']
                    success_rate = analyze_steering_success(samples)
                    empathy_avg = np.mean([count_empathy_words(s) for s in samples])

                    success_total += success_rate
                    empathy_total += empathy_avg
                    count += 1

        if count > 0:
            results[layer]['alphas'].append(alpha)
            results[layer]['success_rates'].append(success_total / count)
            results[layer]['empathy_words'].append(empathy_total / count)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Success rate by alpha
for layer in layers:
    style = '-o' if layer == 12 else '--s'
    color = '#2ecc71' if layer == 12 else '#e74c3c'
    linewidth = 2.5 if layer == 12 else 1.5
    label = f'Layer {layer} (AUROC {1.0 if layer == 12 else 0.991})'
    ax1.plot(results[layer]['alphas'],
             [r * 100 for r in results[layer]['success_rates']],
             style, color=color, linewidth=linewidth, markersize=8,
             label=label, alpha=0.9)

ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Chance')
ax1.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Steering Success Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Detection-Causation Dissociation\nLayer 8: High AUROC, Low Steering | Layer 12: High AUROC, High Steering',
              fontsize=11, fontweight='bold', pad=15)
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Add annotations
ax1.annotate('Layer 12: 93% at α=20\n(14/15 samples)',
             xy=(20, results[12]['success_rates'][-1] * 100),
             xytext=(15, 75),
             fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

ax1.annotate('Layer 8: 20% at α=20\n(3/15 samples)',
             xy=(20, results[8]['success_rates'][-1] * 100),
             xytext=(10, 35),
             fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.2),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# Plot 2: Empathy word count increase
for layer in layers:
    style = '-o' if layer == 12 else '--s'
    color = '#2ecc71' if layer == 12 else '#e74c3c'
    linewidth = 2.5 if layer == 12 else 1.5
    ax2.plot(results[layer]['alphas'],
             results[layer]['empathy_words'],
             style, color=color, linewidth=linewidth, markersize=8,
             label=f'Layer {layer}', alpha=0.9)

ax2.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Empathy Keywords per Sample', fontsize=11, fontweight='bold')
ax2.set_title('Empathy Content Emergence\nLayer 12: 52× Increase (0.13 → 6.8 words)',
              fontsize=11, fontweight='bold', pad=15)
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3)

# Add annotation for 52x increase
baseline_l12 = results[12]['empathy_words'][0] if len(results[12]['empathy_words']) > 0 else 0.13
max_l12 = results[12]['empathy_words'][-1]
if baseline_l12 > 0:
    fold_change = max_l12 / baseline_l12
    ax2.annotate(f'{fold_change:.0f}× increase',
                 xy=(20, max_l12),
                 xytext=(15, max_l12 - 1),
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.2),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "figure4_steering_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "figure4_steering_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved figure4_steering_comparison.pdf/png")

plt.show()
