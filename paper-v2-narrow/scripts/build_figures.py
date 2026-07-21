"""Build prototype manuscript figures directly from committed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"

COLORS = {
    "help": "#247B6B",
    "task": "#D97732",
    "target": "#2D5B9B",
    "null": "#9AA1A9",
    "ink": "#20252B",
    "light": "#EEF1F4",
    "warn": "#B84A4A",
}


def load(path: str):
    return json.loads((ROOT / path).read_text())


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(7.15, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.64, "Contrastive texts", "candidate signal"),
        (0.36, 0.64, "Lexical stress tests", "shortcut check"),
        (0.69, 0.64, "Matched-lexicon\nchoices", "controlled assay"),
        (0.69, 0.22, "Component attribution", "where it is written"),
        (0.36, 0.22, "Rank-1 weight edit", "causal control"),
        (0.03, 0.22, "Nested 42-block\nsearch", "construct test"),
    ]
    role_colors = [
        COLORS["null"], COLORS["help"], COLORS["help"],
        COLORS["target"], COLORS["target"], COLORS["warn"],
    ]
    for i, (x, y, title, subtitle) in enumerate(boxes):
        color = role_colors[i]
        patch = FancyBboxPatch(
            (x, y), 0.27, 0.24,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="white", edgecolor=color, linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + 0.135, y + 0.155, title, ha="center", va="center", weight="bold",
                fontsize=8.5, linespacing=1.05, color=COLORS["ink"])
        ax.text(x + 0.135, y + 0.062, subtitle, ha="center", va="center", fontsize=7.5, color="#59616A")
        if i < len(boxes) - 1:
            nx, ny = boxes[i + 1][0], boxes[i + 1][1]
            if y == ny:
                start_x = x + 0.282 if nx > x else x - 0.012
                end_x = nx - 0.012 if nx > x else nx + 0.282
                ax.annotate("", xy=(end_x, y + 0.12), xytext=(start_x, y + 0.12),
                            arrowprops=dict(arrowstyle="->", lw=1.1, color="#69727C"))
            else:
                ax.annotate("", xy=(nx + 0.135, ny + 0.264), xytext=(x + 0.135, y - 0.012),
                            arrowprops=dict(arrowstyle="->", lw=1.1, color="#69727C"))
    handles = [
        Patch(facecolor="white", edgecolor=COLORS["null"], label="Candidate signal"),
        Patch(facecolor="white", edgecolor=COLORS["help"], label="Controls"),
        Patch(facecolor="white", edgecolor=COLORS["target"], label="Causal tests"),
        Patch(facecolor="white", edgecolor=COLORS["warn"], label="Construct test"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
              ncol=4, fontsize=7.3, handlelength=1.35, columnspacing=1.2)
    ax.text(0.5, 0.015,
            "Increasing evidence strength; causal control does not by itself identify a human-named construct or a complete circuit",
            ha="center", va="center", fontsize=7.8, color=COLORS["ink"])
    save(fig, "fig1_pipeline")


def figure_lexical():
    lexical = load("results/lexical_battery/lexical_battery.json")
    cell_m = load("results/lexical_battery/cell_m.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    labels = ["Block 8", "Block 20"]
    original = [lexical["layers"]["8"]["auroc_orig_orig"], lexical["layers"]["20"]["auroc_orig_orig"]]
    shuffled = [lexical["layers"]["8"]["auroc_shuf_shuf"], lexical["layers"]["20"]["auroc_shuf_shuf"]]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, original, width, color=COLORS["target"], label="Original text")
    ax.bar(x + width / 2, shuffled, width, color=COLORS["help"], label="Tokens shuffled")
    ax.axhline(0.5, color="#69727C", lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("A. Word order is largely unnecessary")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    for xpos, vals in zip((x - width / 2, x + width / 2), (original, shuffled)):
        for xx, yy in zip(xpos, vals):
            ax.text(xx, yy + 0.012, f"{yy:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    full = [cell_m["layers"]["8"]["auroc_full_text"], cell_m["layers"]["20"]["auroc_full_text"]]
    decision = [cell_m["layers"]["8"]["auroc_decision_tokens"], cell_m["layers"]["20"]["auroc_decision_tokens"]]
    ax.bar(x - width / 2, full, width, color=COLORS["target"], label="Full text")
    ax.bar(x + width / 2, decision, width, color=COLORS["help"], label="Decision tokens")
    ax.axhline(0.5, color="#69727C", lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 0.86)
    ax.set_ylabel("AUROC")
    ax.set_title("B. Matched lexicon isolates decision clauses")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    for xpos, vals in zip((x - width / 2, x + width / 2), (full, decision)):
        for xx, yy in zip(xpos, vals):
            ax.text(xx, yy + 0.009, f"{yy:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Linear decodability can survive removal of compositional structure", y=1.03, fontsize=11, weight="bold")
    save(fig, "fig2_lexical_controls")


def errorbar(ax, y, value, ci, color, marker, label=None):
    ax.errorbar(value, y, xerr=[[value - ci[0]], [ci[1] - value]], fmt=marker,
                color=color, ecolor=color, capsize=2.5, markersize=5, label=label)


def figure_writer_effect():
    gate = load("results/gate0b_task_control_accepted_20260713/task_control.json")
    stress = load("results/e26_format_stress_gemma/e26.json")["assays"]["writers"]
    summaries = gate["analysis"]["summaries"]
    raw = summaries["raw_ab_dual"]["positive_writers_k2"]
    chat = summaries["chat_ab_dual"]["positive_writers_k2"]
    cont = summaries["continuation"]["positive_writers_k2"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05),
                             gridspec_kw={"width_ratios": [1.8, 0.8], "wspace": 0.50})
    ax = axes[0]
    rows = [
        ("Raw A/B", raw["M_confirm"], raw["T_confirm_repaired"]),
        ("Chat A/B", chat["M_confirm"], chat["T_confirm_repaired"]),
        ("Paraphrase 1", stress["para_p1"], None),
        ("Paraphrase 2", stress["para_p2"], None),
        ("Paraphrase 3", stress["para_p3"], None),
    ]
    ys = np.arange(len(rows))[::-1]
    for idx, (label, m, t) in enumerate(rows):
        errorbar(ax, ys[idx] + 0.10, m["mean"], m.get("family_bootstrap_ci95", m.get("ci95")), COLORS["help"], "o",
                 None)
        if t is not None:
            errorbar(ax, ys[idx] - 0.10, t["mean"], t.get("family_bootstrap_ci95", t.get("ci95")), COLORS["task"], "s",
                     None)
    ax.axvline(0, color="#69727C", lw=0.8, ls="--")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("Change in helping-favoring logit margin")
    ax.set_title("A. Forced-choice formats")
    ax.text(-0.225, ys[0] + 0.10, "Costly-helping", color=COLORS["help"],
            va="center", fontsize=7.7)
    ax.text(-0.070, ys[0] - 0.10, "Task control", color=COLORS["task"],
            ha="right", va="center", fontsize=7.7)

    ax = axes[1]
    errorbar(ax, 1.1, cont["M_confirm"]["mean"], cont["M_confirm"]["family_bootstrap_ci95"], COLORS["help"], "o", "Costly-helping")
    errorbar(ax, 0.9, cont["T_confirm_repaired"]["mean"], cont["T_confirm_repaired"]["family_bootstrap_ci95"], COLORS["task"], "s", "Task control")
    ax.axvline(0, color="#69727C", lw=0.8, ls="--")
    ax.set_yticks([1.1, 0.9], ["Costly-helping", "Task control"])
    ax.set_ylim(0.76, 1.24)
    ax.set_xlabel("Change in mean tail log likelihood")
    ax.set_title("B. No A/B scaffold")
    fig.suptitle("Removing the selected direction from L19/L20 MLP weights reduces the matched choice score",
                 y=1.03, fontsize=10.5, weight="bold")
    save(fig, "fig3_writer_effect")


def figure_band_specificity():
    data = load("results/e26_matched_nulls_gemma/matched_nulls.json")["families"]["writer_matched"]
    points = []
    for row in data["null_sets"]:
        points.append((" + ".join(x.replace("MLP", "") for x in row["set"]), row["delta"], "disjoint"))
    for row in data["overlap_sets"]:
        points.append((" + ".join(x.replace("MLP", "") for x in row["set"]), row["delta"], "shares L19/20"))
    points.append(("L19 + L20", data["targeted_delta"], "targeted"))
    points.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7.15, 3.2))
    for i, (_, delta, kind) in enumerate(points):
        color = COLORS["target"] if kind == "targeted" else "#5878A8" if kind == "shares L19/20" else "#C4C9CF"
        size = 48 if kind == "targeted" else 27
        ax.scatter(i + 1, delta, color=color, s=size, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#69727C", lw=0.8, ls="--")
    ax.set_xlabel("Two-MLP set, ranked by effect")
    ax.set_ylabel("Change in helping-favoring logit margin")
    ax.set_title("L19/L20 is the most extreme pair in the tested L16--L23 MLP band")
    ax.annotate("L19 + L20\nexact rank p = 1/28", xy=(1, data["targeted_delta"]), xytext=(5, -0.25),
                arrowprops=dict(arrowstyle="->", color=COLORS["target"]), color=COLORS["target"], fontsize=8.2)
    ax.scatter([], [], color="#C4C9CF", label="Disjoint from L19/L20")
    ax.scatter([], [], color="#5878A8", label="Shares L19 or L20")
    ax.scatter([], [], color=COLORS["target"], label="Targeted L19/L20")
    ax.legend(loc="upper left", ncol=3, fontsize=7.5)
    save(fig, "fig4_band_specificity")


def figure_selection_instability():
    selection = load("results/wp2_broadened_dev_allblocks_20260713/selection/selection.json")
    calibration = load("results/wp2_permutation_calibration_20260713/permutation_calibration.json")
    folds = selection["nested_outer"]["folds"]
    blocks = [f["chosen_config"]["block"] for f in folds]
    roles = [f["chosen_config"]["role"] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    for i, (block, role) in enumerate(zip(blocks, roles)):
        marker = "o" if role == "prompt_final" else "s"
        ax.scatter(i + 1, block, marker=marker, s=65, color=COLORS["target"])
        ax.text(i + 1, block + 1.8, str(block), ha="center", fontsize=8)
    ax.set_xticks(range(1, 5), [f"Fold {i}" for i in range(1, 5)])
    ax.set_ylim(-1, 42)
    ax.set_ylabel("Selected transformer block")
    ax.set_title("A. Fold winners span the network")
    ax.scatter([], [], marker="o", color=COLORS["target"], label="Prompt-final")
    ax.scatter([], [], marker="s", color=COLORS["target"], label="Quote-boundary")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    null = calibration["null"]["primary_fold_convergence"]["values"]
    observed = calibration["observed"]["mean_pairwise_block_distance"]
    ax.hist(null, bins=np.arange(0, 25, 2), color=COLORS["null"], edgecolor="white")
    ax.axvline(observed, color=COLORS["warn"], lw=2, label=f"Observed = {observed:.1f}")
    ax.axvline(np.median(null), color=COLORS["ink"], lw=1.2, ls="--", label=f"Null median = {np.median(null):.1f}")
    ax.set_xlabel("Fold-winner distance (blocks)\nLower = greater convergence")
    ax.set_ylabel("Permutations")
    ax.set_title("B. Convergence matches the null")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Nested representation selection does not converge on a common site",
                 y=1.03, fontsize=10.5, weight="bold")
    save(fig, "fig5_selection_instability")


def figure_capability():
    data = load("results/gate0c_capability_accepted_20260713/capability_eval.json")
    names = ["Writer k=2", "Suppressor k=4", "Combined k=6"]
    keys = ["positive_writers_k2", "suppressors_k4", "targeted_k6"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    for y, (name, key) in enumerate(zip(names[::-1], keys[::-1])):
        row = data["conditions"][key]["mmlu_delta_vs_baseline"]
        errorbar(ax, y, row["mean"], row["ci95"], COLORS["target"], "o")
    ax.axvline(0, color="#69727C", lw=0.8, ls="--")
    ax.set_yticks(range(3), names[::-1])
    ax.set_xticks([-0.01, -0.005, 0.0, 0.0025])
    ax.set_xlabel("MMLU accuracy change")
    ax.set_title("A. 800-item stratified MMLU sample")
    ax = axes[1]
    ratios = [data["conditions"][k]["ppl_ratio_vs_baseline"] for k in keys]
    ax.barh(range(3), ratios[::-1], color=[COLORS["target"], COLORS["null"], COLORS["help"]])
    ax.axvline(1, color="#69727C", lw=0.8, ls="--")
    ax.set_yticks(range(3), names[::-1])
    ax.set_xlim(0.995, 1.004)
    ax.set_xlabel("WikiText perplexity ratio")
    ax.set_title("B. 45,338 held-out tokens")
    fig.suptitle("Capability checks bound observed collateral change; they are not equivalence tests",
                 y=1.03, fontsize=10.3, weight="bold")
    save(fig, "figA1_capability")


def main():
    style()
    figure_pipeline()
    figure_lexical()
    figure_writer_effect()
    figure_band_specificity()
    figure_selection_instability()
    figure_capability()
    print(f"Wrote prototype figures to {OUT}")


if __name__ == "__main__":
    main()
