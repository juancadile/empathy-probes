"""Generate V2 paper figures from committed result JSONs.

Fig 1: E18 dose-response — delta uptake vs cost level, welfare vs non-social
       axis, per edit condition, with family-clustered CIs.
Fig 2: E18b cross-model fractional ablation curves (Gemma B20 vs Llama B15),
       with random-direction controls.

Usage: python src/visualization/paper_v2_figures.py
Writes paper-v2/figures/*.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COST_LEVELS = ["free", "low", "medium", "high"]
COND_STYLE = {
    "positive_writers_k2": dict(color="#1f77b4", label="writers removed (k=2)"),
    "suppressors_k4": dict(color="#d62728", label="suppressors removed (k=4)"),
    "random_k6": dict(color="#7f7f7f", label="random components (k=6)"),
}


def fig1_e18():
    r = json.load(open(ROOT / "results/e18_interaction_gemma2_9b_it/e18.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cell, title in [
        (axes[0], "cost_axis", "Welfare alternative (distressed person)"),
        (axes[1], "nonsocial_axis", "Matched non-welfare alternative (excited person)"),
    ]:
        for cond, style in COND_STYLE.items():
            d = r["conditions"][cond][cell]["delta_by_cost"]
            means = [d[c]["mean"] for c in COST_LEVELS]
            lo = [d[c]["mean"] - d[c]["ci95"][0] for c in COST_LEVELS]
            hi = [d[c]["ci95"][1] - d[c]["mean"] for c in COST_LEVELS]
            x = np.arange(4)
            ax.errorbar(x, means, yerr=[lo, hi], marker="o", capsize=3,
                        lw=1.6, **style)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(4), COST_LEVELS)
        ax.set_xlabel("cost of helping")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"$\Delta$ alternative-uptake logit vs baseline")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Weight-edit effects vs helping cost (Gemma-2-9B-it; family-clustered 95% CIs)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_cost_gate.pdf")
    print("wrote fig1_cost_gate.pdf")


def fig2_fractional():
    curves = {
        "Gemma-2-9B-it (block 20)": (json.load(open(ROOT / "results/e17b_frac_gemma/e17b.json")), "#d62728"),
        "Llama-3.1-8B-Instruct (block 15)": (json.load(open(ROOT / "results/e17b_frac_llama/e17b.json")), "#1f77b4"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, (r, color) in curves.items():
        t3 = r["test3_fractional_ablation"]
        fr = [0.0, 0.25, 0.5, 0.75, 1.0]
        means, lo, hi = [0.0], [0.0], [0.0]
        for f in fr[1:]:
            d = t3[f"fraction_{f}"]["M_confirm"]
            means.append(d["mean"])
            lo.append(d["mean"] - d["ci95"][0])
            hi.append(d["ci95"][1] - d["mean"])
        ax.errorbar(fr, means, yerr=[lo, hi], marker="o", capsize=3, lw=1.6,
                    color=color, label=label)
        for i in range(3):
            d = t3[f"random_dir_{i}"]
            ax.errorbar([1.0 + 0.015 * (i - 1)], [d["mean"]],
                        yerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]],
                        marker="x", color=color, alpha=0.55, capsize=2, lw=0.9,
                        label=(f"{label.split(' ')[0]} random directions (f=1)" if i == 0 else None))
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("fraction of direction projection removed")
    ax.set_ylabel(r"$\Delta$ helping-choice logit (confirmatory M)")
    ax.set_title("Direction dependence of the helping choice\n(fractional ablation; family-clustered 95% CIs)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")

    # inset: Llama at its own scale (its -0.16 curve is invisible at Gemma's)
    axin = ax.inset_axes([0.56, 0.55, 0.4, 0.4])
    r, color = curves["Llama-3.1-8B-Instruct (block 15)"]
    t3 = r["test3_fractional_ablation"]
    fr = [0.0, 0.25, 0.5, 0.75, 1.0]
    means, lo, hi = [0.0], [0.0], [0.0]
    for f in fr[1:]:
        d = t3[f"fraction_{f}"]["M_confirm"]
        means.append(d["mean"])
        lo.append(d["mean"] - d["ci95"][0])
        hi.append(d["ci95"][1] - d["mean"])
    axin.errorbar(fr, means, yerr=[lo, hi], marker="o", capsize=2, lw=1.2,
                  color=color, markersize=3)
    for i in range(3):
        d = t3[f"random_dir_{i}"]
        axin.errorbar([1.0 + 0.02 * (i - 1)], [d["mean"]],
                      yerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]],
                      marker="x", color=color, alpha=0.55, capsize=1.5, lw=0.8)
    axin.axhline(0, color="k", lw=0.5, ls=":")
    axin.set_title("Llama, own scale", fontsize=7)
    axin.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_fractional_ablation.pdf")
    print("wrote fig2_fractional_ablation.pdf")


if __name__ == "__main__":
    fig1_e18()
    fig2_fractional()
