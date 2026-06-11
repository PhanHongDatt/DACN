"""
gen_report_plots.py — Sinh các plot tổng hợp dùng cho báo cáo KLTN.

Output: results/plots/report/*.png (dpi=250).
Tập trung vào narrative chính: CSRA reward fairness, leakage trade-off, β sweep,
detection limitation, accuracy robustness.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "summary_metrics.csv"
OUT_DIR = ROOT / "results" / "plots" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color & label conventions
METHOD_ORDER = [
    "fedavg+equal", "fedavg+data", "fedavg+quality",
    "fedavg+csra", "csra_dcd+equal", "csra_dcd+csra",
]
METHOD_LABEL = {
    "fedavg+equal":   "M1 FedAvg+Equal",
    "fedavg+data":    "M2 FedAvg+DataSize",
    "fedavg+quality": "M3 FedAvg+Quality",
    "fedavg+csra":    "M4 FedAvg+CSRA",
    "csra_dcd+equal": "M5 CSRA-DCD+Equal",
    "csra_dcd+csra":  "M6 CSRA-DCD+CSRA",
}
METHOD_COLOR = {
    "fedavg+equal":   "#9aa0a6",
    "fedavg+data":    "#86c986",
    "fedavg+quality": "#f3b860",
    "fedavg+csra":    "#c14b6e",  # highlighted (đề xuất reward)
    "csra_dcd+equal": "#7a6cc6",
    "csra_dcd+csra":  "#1a5a8c",  # highlighted (full system)
}
ATTACK_ORDER = ["free_rider", "lazy", "label_noise", "sign_flip"]
SCN_ORDER = ["K1", "K2", "K3 (dirichlet=0.5)", "K3 (dirichlet=0.1)"]


def _bar_xpos(n_methods: int, n_groups: int, width: float = 0.13):
    pos = np.arange(n_groups)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width
    return pos, offsets


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Jain Index across 4 scenarios — CLEAN runs (flagship fairness)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fairness_jain(df: pd.DataFrame):
    clean = df[df.attack_label == "clean"].copy()
    # Mean Jain across datasets and seeds, β fixed at 0.5 if CSRA
    clean = clean[(clean.reward_policy != "csra") | (clean.beta.isin([0.0, 0.5]))]
    grouped = (
        clean.groupby(["method", "scenario_variant"])
        .agg(jain=("jain", "mean"))
        .reset_index()
    )
    piv = grouped.pivot(index="scenario_variant", columns="method", values="jain")
    piv = piv.reindex(SCN_ORDER)[METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos, offsets = _bar_xpos(len(METHOD_ORDER), len(piv.index), width=0.13)
    for i, m in enumerate(METHOD_ORDER):
        ax.bar(pos + offsets[i], piv[m].values, width=0.13,
               color=METHOD_COLOR[m], label=METHOD_LABEL[m],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels([s.replace("dirichlet", "α") for s in piv.index], fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Jain Fairness Index (↑ better)", fontsize=11)
    ax.set_title("Fairness so sánh 6 methods × 4 scenarios (clean, gộp MNIST + Fashion-MNIST)",
                 fontsize=12, fontweight="bold")
    ax.axhline(1.0, ls=":", color="grey", lw=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", ncol=3, fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    out = OUT_DIR / "fig1_fairness_jain_by_scenario.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Reward Leakage by attack type — ATTACK runs (flagship robustness)
# ─────────────────────────────────────────────────────────────────────────────

def plot_leakage_by_attack(df: pd.DataFrame):
    atk = df[(df.attack_label == "attack") &
             (df.scenario_variant.isin(["K2", "K3 (dirichlet=0.1)"]))]
    atk = atk[(atk.reward_policy != "csra") | (atk.beta == 0.5)]
    piv = (
        atk.groupby(["method", "attack_type"])["reward_leakage"]
        .mean().reset_index()
        .pivot(index="attack_type", columns="method", values="reward_leakage")
        .reindex(ATTACK_ORDER)[METHOD_ORDER]
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos, offsets = _bar_xpos(len(METHOD_ORDER), len(piv.index), width=0.13)
    for i, m in enumerate(METHOD_ORDER):
        ax.bar(pos + offsets[i], piv[m].values, width=0.13,
               color=METHOD_COLOR[m], label=METHOD_LABEL[m],
               edgecolor="white", linewidth=0.6)
    ax.axhline(0.2, ls="--", color="red", lw=1.0, alpha=0.6,
               label="Baseline: 2/10 = 20% (random share)")
    ax.set_xticks(pos)
    ax.set_xticklabels([a.replace("_", "\n") for a in piv.index], fontsize=10)
    ax.set_ylabel("Reward Leakage = R_malicious / R_total  (↓ better)", fontsize=11)
    ax.set_title("Reward Leakage theo loại Attack (gộp K2 + K3@α=0.1, 2 datasets)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    out = OUT_DIR / "fig2_leakage_by_attack.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: β sweep tradeoff (Jain vs reward concentration vs accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def plot_beta_sweep(df: pd.DataFrame):
    sw = df[(df.method == "fedavg+csra") &
            (df.scenario_variant == "K3 (dirichlet=0.1)") &
            (df.attack_label == "clean")]
    sw = sw[sw.beta.isin([0.3, 0.5, 0.7])]
    grp = (
        sw.groupby(["dataset", "beta"])
        .agg(jain=("jain", "mean"),
             gini=("gini", "mean"),
             reward_std=("reward_std", "mean"),
             final_accuracy=("final_accuracy", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    ax_acc, ax_jain, ax_gini = axes
    ds_colors = {"mnist": "#1a5a8c", "fashion_mnist": "#c14b6e"}
    ds_labels = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}

    for ds, sub in grp.groupby("dataset"):
        sub = sub.sort_values("beta")
        ax_acc.plot(sub.beta, sub.final_accuracy, "o-",
                    color=ds_colors[ds], label=ds_labels[ds], lw=2, ms=8)
        ax_jain.plot(sub.beta, sub.jain, "o-",
                     color=ds_colors[ds], label=ds_labels[ds], lw=2, ms=8)
        ax_gini.plot(sub.beta, sub.gini, "o-",
                     color=ds_colors[ds], label=ds_labels[ds], lw=2, ms=8)

    for a in axes:
        a.set_xlabel("β  (quality weight)", fontsize=10)
        a.set_xticks([0.3, 0.5, 0.7])
        a.grid(True, alpha=0.3)
        a.legend(loc="best", fontsize=9)

    ax_acc.set_ylabel("Final Accuracy", fontsize=10)
    ax_acc.set_title("(a) Accuracy không phụ thuộc β", fontsize=10)
    ax_jain.set_ylabel("Jain Index (↑)", fontsize=10)
    ax_jain.set_title("(b) Jain GIẢM khi β tăng", fontsize=10)
    ax_gini.set_ylabel("Gini (↓)", fontsize=10)
    ax_gini.set_title("(c) Gini TĂNG khi β tăng", fontsize=10)

    fig.suptitle("β sweep cho M4 (FedAvg + CSRAReward) trên K3@α=0.1 — "
                 "β=0.3 cho fairness tốt nhất, accuracy không đổi",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig3_beta_sweep_tradeoff.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Detection limitation — TPR vs FPR per attack
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_limitation(df: pd.DataFrame):
    sub = df[(df.attack_label == "attack") &
             (df.aggregation_method == "csra_dcd")]
    grouped = (
        sub.groupby(["attack_type"])
        .agg(tpr=("attack_detection_rate", "mean"),
             fpr=("false_positive_rate", "mean"))
        .reset_index()
        .set_index("attack_type")
        .reindex(ATTACK_ORDER)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pos = np.arange(len(grouped.index))
    width = 0.36
    bars_tpr = ax.bar(pos - width/2, grouped.tpr, width=width,
                       color="#2a8a4e", label="True Positive Rate (↑ better)",
                       edgecolor="white")
    bars_fpr = ax.bar(pos + width/2, grouped.fpr, width=width,
                       color="#c14b6e", label="False Positive Rate (↓ better)",
                       edgecolor="white")
    # Annotate values
    for b, v in zip(bars_tpr, grouped.tpr):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.1%}",
                ha="center", fontsize=8.5)
    for b, v in zip(bars_fpr, grouped.fpr):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.1%}",
                ha="center", fontsize=8.5)
    ax.axhline(0.05, ls=":", color="grey", lw=0.8, label="FPR target = 5%")
    ax.set_xticks(pos)
    ax.set_xticklabels([a.replace("_", "\n") for a in grouped.index], fontsize=10)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Hạn chế của MAD-based filter (CSRA-DCD):\n"
                 "Phát hiện kém với 3/4 attack, FPR vượt ngưỡng 5%",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "fig4_detection_limitation.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Accuracy robustness — clean vs each attack
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_robustness(df: pd.DataFrame):
    sub = df[df.scenario_variant.isin(["K2", "K3 (dirichlet=0.1)"])]
    sub = sub[(sub.reward_policy != "csra") | (sub.beta == 0.5)]
    grp = (
        sub.groupby(["method", "attack_type"])
        .agg(acc=("final_accuracy", "mean"))
        .reset_index()
    )
    # Pivot: rows=method, cols=condition
    conds_order = ["clean", "free_rider", "lazy", "label_noise", "sign_flip"]
    piv = grp.pivot(index="method", columns="attack_type", values="acc")
    piv = piv.reindex(METHOD_ORDER)[conds_order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pos, offsets = _bar_xpos(len(METHOD_ORDER), len(conds_order), width=0.13)
    for i, m in enumerate(METHOD_ORDER):
        ax.bar(pos + offsets[i], piv.loc[m].values, width=0.13,
               color=METHOD_COLOR[m], label=METHOD_LABEL[m],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds_order], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Final Accuracy", fontsize=11)
    ax.set_title("Accuracy của 6 methods qua các điều kiện attack\n"
                 "(gộp K2 + K3@α=0.1, 2 datasets) — Sign-flip phá tất cả methods",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", ncol=3, fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    out = OUT_DIR / "fig5_accuracy_robustness.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Fairness pareto (Jain vs leakage)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_jain_leakage(df: pd.DataFrame):
    sub = df[(df.attack_label == "attack") &
             (df.scenario_variant == "K3 (dirichlet=0.1)")]
    sub = sub[(sub.reward_policy != "csra") | (sub.beta == 0.5)]
    grp = (
        sub.groupby(["method"])
        .agg(jain=("jain", "mean"),
             leakage=("reward_leakage", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in grp.iterrows():
        m = r["method"]
        ax.scatter(r["leakage"], r["jain"], s=230, c=METHOD_COLOR[m],
                   edgecolors="white", linewidth=1.5, zorder=3)
        # Offset annotation
        dx, dy = 0.007, 0.005
        if m == "fedavg+equal":
            dx, dy = 0.007, -0.015
        elif m == "csra_dcd+equal":
            dx, dy = -0.04, 0.015
        ax.annotate(METHOD_LABEL[m], (r["leakage"] + dx, r["jain"] + dy),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Reward Leakage (↓ better)", fontsize=11)
    ax.set_ylabel("Jain Fairness Index (↑ better)", fontsize=11)
    ax.set_title("Pareto: Fairness vs Leakage — Attack runs trên K3@α=0.1\n"
                 "Góc trên trái = lý tưởng (công bằng + ít leak)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Ideal corner annotation
    ax.annotate("", xy=(0.01, 1.02), xytext=(0.18, 0.78),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(0.18, 0.76, "Ideal", color="green", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig6_pareto_jain_vs_leakage.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Missing {SUMMARY_CSV}. Run analyze_results.py first.")
    df = pd.read_csv(SUMMARY_CSV)
    log.info("Loaded %d summary rows", len(df))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_fairness_jain(df)
    plot_leakage_by_attack(df)
    plot_beta_sweep(df)
    plot_detection_limitation(df)
    plot_accuracy_robustness(df)
    plot_pareto_jain_leakage(df)
    log.info("All 6 report plots saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
