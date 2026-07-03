import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript_neurocomputing" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


BEST_PRIMARY = {
    "Cora": 0.1772,
    "CiteSeer": 0.2101,
    "PubMed": 0.2809,
    "Amazon-Photo": 0.2131,
    "Amazon-Computers": 0.1779,
    "Roman-empire": 0.1702,
    "Tolokers": 0.3664,
}


DISPLAY_DATASETS = {
    "amazon_photo": "Amazon-Photo",
    "cora": "Cora",
    "pubmed": "PubMed",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_pareto_tradeoff():
    rows = read_csv(ROOT / "paper" / "tables_revision4_20260703" / "communication_runtime_rev4_dataset_compact.csv")
    datasets = [r["dataset"] for r in rows]
    x = np.array([float(r["spc_runtime_ratio_vs_fedavg"]) for r in rows])
    y = np.array([float(r["macro_f1"]) - BEST_PRIMARY[r["dataset"]] for r in rows])
    teacher = np.array([float(r["teacher_extra_MB"]) for r in rows])
    size = 90 + 360 * (np.sqrt(teacher) / max(np.sqrt(teacher)))
    colors = []
    for d in datasets:
        if d in {"Roman-empire", "Tolokers"}:
            colors.append("#7b3294")
        elif d.startswith("Amazon"):
            colors.append("#008837")
        else:
            colors.append("#2166ac")

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.axhline(0, color="#777777", lw=0.9, ls="--", zorder=0)
    sc = ax.scatter(x, y, s=size, c=colors, alpha=0.86, edgecolor="white", linewidth=0.9)
    label_offsets = {
        "Cora": (0.035, 0.004, "left"),
        "CiteSeer": (0.035, -0.006, "left"),
        "PubMed": (-0.12, 0.004, "right"),
        "Amazon-Photo": (0.035, 0.004, "left"),
        "Amazon-Computers": (0.035, 0.004, "left"),
        "Roman-empire": (-0.08, 0.009, "right"),
        "Tolokers": (-0.18, 0.004, "right"),
    }
    for xi, yi, d in zip(x, y, datasets):
        dx, dy, ha = label_offsets[d]
        ax.text(xi + dx, yi + dy, d.replace("Amazon-", "Amz-"), fontsize=8.5, ha=ha, va="center")

    ax.set_xlabel("Runtime ratio vs. FedAvg-GCN")
    ax.set_ylabel("Macro-F1 gain over best primary baseline")
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.8)
    ax.set_xlim(max(2.7, x.min() - 0.25), x.max() + 0.30)
    ax.set_ylim(y.min() - 0.018, y.max() + 0.018)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Citation", markerfacecolor="#2166ac", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Product", markerfacecolor="#008837", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Heterophily", markerfacecolor="#7b3294", markersize=8),
    ]
    leg1 = ax.legend(handles=handles, loc="upper left", frameon=False, title="Dataset type")
    ax.add_artist(leg1)

    for mb in [10, 300, 1200, 3400]:
        ax.scatter([], [], s=90 + 360 * (np.sqrt(mb) / max(np.sqrt(teacher))), c="#bbbbbb", alpha=0.65,
                   edgecolor="white", label=f"{mb:g} MB")
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.17), frameon=False, title="Teacher transfer")
    fig.tight_layout()
    save(fig, "communication_runtime_tradeoff")


def make_revision6_combo():
    fair = read_csv(ROOT / "paper" / "tables_revision6_20260703" / "fair_cb_comparison.csv")
    gate = read_csv(ROOT / "paper" / "tables_revision6_20260703" / "gate_independent_deltas.csv")

    order = ["cora", "pubmed", "amazon_photo"]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw={"width_ratios": [1.03, 1.12]})

    # Panel A: dumbbell fair-CB comparison.
    ax = axes[0]
    fair_by = {r["dataset"]: r for r in fair}
    spc = np.array([float(fair_by[d]["spc_guard_pp"]) for d in order])
    best = np.array([float(fair_by[d]["best_external_mean"]) for d in order])
    labels = [DISPLAY_DATASETS[d] for d in order]
    for i, d in enumerate(order):
        ax.plot([best[i], spc[i]], [y[i], y[i]], color="#a6a6a6", lw=2.0, zorder=1)
        ax.scatter(best[i], y[i], s=55, color="#d95f02", label="Best fair baseline" if i == 0 else "", zorder=2)
        ax.scatter(spc[i], y[i], s=55, color="#1b9e77", label="SPC-FedGNN" if i == 0 else "", zorder=3)
        winner_x = max(best[i], spc[i])
        ax.text(winner_x + 0.006, y[i], fair_by[d]["delta_vs_best_external"], fontsize=8.2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Macro-F1")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.grid(axis="x", color="#e6e6e6")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel B: slope chart for independent gate evidence.
    ax = axes[1]
    methods = [("spc_guard_pp", "SPC-FedGNN", "#1b9e77"), ("spc_guard", "Base Guard", "#7570b3")]
    x_pos = np.array([0, 1])
    offset = {"spc_guard_pp": -0.055, "spc_guard": 0.055}
    x_jitter = {"spc_guard_pp": -0.018, "spc_guard": 0.018}
    for d_i, d in enumerate(order):
        for method, label, color in methods:
            row = next(r for r in gate if r["dataset"] == d and r["method"] == method)
            vals = [float(row["train_evidence_mean"]), float(row["independent_gate_mean"])]
            xs = x_pos + x_jitter[method]
            ax.plot(xs, vals, color=color, lw=1.8, marker="o", ms=4.5, alpha=0.9,
                    label=label if d_i == 0 else "")
            ax.text(1.06, vals[1], row["delta_independent_minus_train"], fontsize=7.5,
                    va="center", color=color)
    for d in order:
        vals = [
            float(r[key])
            for r in gate
            if r["dataset"] == d
            for key in ["train_evidence_mean", "independent_gate_mean"]
        ]
        ax.text(0.50, np.mean(vals) + 0.004, DISPLAY_DATASETS[d], ha="center", va="bottom", fontsize=8.0,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Train evidence", "Independent evidence"])
    ax.set_ylabel("Mean Macro-F1")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_xlim(-0.22, 1.18)
    gate_vals = [
        float(r[key])
        for r in gate
        for key in ["train_evidence_mean", "independent_gate_mean"]
    ]
    ax.set_ylim(min(gate_vals) - 0.018, max(gate_vals) + 0.018)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.5, -0.24, "Each line tracks one method when gate evidence is separated.", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.2, color="#555555")

    fig.tight_layout(w_pad=2.0)
    save(fig, "revision6_fairness_gate_diagnostics")


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    make_pareto_tradeoff()
    make_revision6_combo()


if __name__ == "__main__":
    main()
