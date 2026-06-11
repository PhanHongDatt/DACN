"""
gen_system_diagram.py — Sơ đồ Triển khai Hệ thống cho báo cáo Ch.4.
Output: results/plots/report/system_implementation.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "plots" / "report" / "system_implementation.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def cell(ax, x, y, w, h, title, body, color, fs_title=10, fs_body=8.5,
         edge="#333"):
    """Hộp với title in đậm ở trên, body text ở dưới."""
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor=edge, facecolor=color, alpha=0.97,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top",
            fontsize=fs_title, fontweight="bold", color="#1a1a1a")
    if body:
        ax.text(x + w / 2, y + (h - 2.5) / 2, body, ha="center", va="center",
                fontsize=fs_body, color="#1a1a1a")


def container(ax, x, y, w, h, title, color="#f6f6f6", edge="#888"):
    """Container rỗng có title ở góc trên trái."""
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=color, alpha=0.5,
    )
    ax.add_patch(rect)
    ax.text(x + 0.6, y + h - 0.6, title, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#444")


def arrow(ax, p1, p2, color="#444", lw=1.5, label=None, label_xy=None,
          style="->", ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, color=color, lw=lw, mutation_scale=16,
        linestyle=ls,
    ))
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center",
                fontsize=8.5, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc",
                          alpha=0.95, lw=0.5))


def section_band(ax, y, text, color="#fcecd9"):
    ax.add_patch(plt.Rectangle((4, y), 96, 1.5, facecolor=color,
                                edgecolor="none"))
    ax.text(5, y + 0.75, text, ha="left", va="center", fontsize=10.5,
            fontweight="bold", color="#a04000")


def main():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ── TITLE ──
    ax.text(50, 97.5, "Sơ đồ Triển khai Hệ thống FL + Blockchain",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(50, 95, "Python 3.11 · Node 20 · Hardhat 2.22 · Solidity 0.8.19",
            ha="center", fontsize=10.5, style="italic", color="#555")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 1 — CONTROL
    # ═══════════════════════════════════════════════════════════════
    section_band(ax, 89, "1.  TẦNG ĐIỀU KHIỂN  (Orchestration)")

    cell(ax, 5, 80, 26, 7, "CLI Launcher",
         "experiments/run_all.sh\n--smoke / --quick / --full",
         "#fdebd0")
    cell(ax, 34, 80, 32, 7, "Unified Runner",
         "experiments/run_experiment.py\n--aggregation × --reward-policy --seed --attack",
         "#fdebd0")
    cell(ax, 69, 80, 26, 7, "Analysis Pipeline",
         "analyze_results.py + analysis/*\nLoader → Stats → Plots → Report",
         "#fdebd0")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 2 — PYTHON FL CORE
    # ═══════════════════════════════════════════════════════════════
    section_band(ax, 77, "2.  TẦNG FL CORE  (Python — fl/)")

    container(ax, 5, 59, 27, 16, "Server-side")
    cell(ax, 6.5, 67, 24, 6, "FLUnifiedStrategy",
         "server_base.py\nsimulation_local.py (sequential)",
         "#fff9e6", fs_title=9.5, fs_body=8)
    cell(ax, 6.5, 60, 11, 6, "aggregation",
         "fedavg / trimmed\n/ csra_dcd (MAD-z)",
         "#fff9e6", fs_title=9, fs_body=7.5)
    cell(ax, 19.5, 60, 11, 6, "reward policy",
         "equal / data /\nquality / csra",
         "#fff9e6", fs_title=9, fs_body=7.5)

    container(ax, 35, 59, 31, 16, "Clients  (client_attacks.py)")
    cell(ax, 36.5, 68, 28, 5, "HonestClient",
         "local train · report quality + Δ-norm",
         "#e8f6e8", fs_title=9.5, fs_body=8)
    cell(ax, 36.5, 60, 6.5, 7, "FreeRider", "noise Δ",
         "#fde2e2", fs_title=8.5, fs_body=7.5)
    cell(ax, 43.5, 60, 6.5, 7, "Lazy", "1 epoch",
         "#fde2e2", fs_title=8.5, fs_body=7.5)
    cell(ax, 50.5, 60, 7, 7, "LabelNoise", "30% flip",
         "#fde2e2", fs_title=8.5, fs_body=7.5)
    cell(ax, 58, 60, 6.5, 7, "SignFlip", "−Δ Byzantine",
         "#fde2e2", fs_title=8.5, fs_body=7.5)

    container(ax, 69, 59, 26, 16, "Data + Logger")
    cell(ax, 70.5, 68, 23, 5, "data_utils.py",
         "K1 IID · K2 Weak · K3 Dirichlet",
         "#eef5ff", fs_title=9.5, fs_body=8)
    cell(ax, 70.5, 64, 23, 3.5, "models.py",
         "CNN MNIST / Fashion-MNIST",
         "#eef5ff", fs_title=9, fs_body=7.5)
    cell(ax, 70.5, 60, 23, 3.5, "logger.py",
         "CSV schema v2",
         "#eef5ff", fs_title=9, fs_body=7.5)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 3 — BLOCKCHAIN
    # ═══════════════════════════════════════════════════════════════
    section_band(ax, 56, "3.  TẦNG BLOCKCHAIN  (Solidity + Hardhat)")

    container(ax, 5, 33, 25, 21, "Smart Contracts (contracts/)", color="#ede7f5")
    cell(ax, 6.5, 47, 22, 5.5, "ContributionStore.sol",
         "Circular buffer W=10\nEWMA reputation (β=0.9)",
         "#d2b4de", fs_title=9, fs_body=7.5)
    cell(ax, 6.5, 40.5, 22, 5.5, "RewardDistributor.sol",
         "distributeRewards(addrs, weights)\npayable · transfer ETH",
         "#d2b4de", fs_title=9, fs_body=7.5)
    cell(ax, 6.5, 34, 22, 5.5, "FLRegistry.sol",
         "start/endExperiment\nOn-chain audit metadata",
         "#d2b4de", fs_title=9, fs_body=7.5)

    container(ax, 33, 33, 28, 21, "Hardhat Local Chain", color="#dde9f5")
    cell(ax, 34.5, 47, 25, 5.5, "Hardhat Node",
         "npx hardhat node\nRPC :8545 · chain_id 31337",
         "#aed6f1", fs_title=9.5, fs_body=8)
    cell(ax, 34.5, 40, 25, 5.5, "20 Accounts × 10,000 ETH",
         "[0]=owner · [1..10]=clients · [11]=buffer",
         "#aed6f1", fs_title=9, fs_body=7.5)
    cell(ax, 34.5, 34, 25, 4.5, "Deploy Scripts",
         "scripts/{deploy,fund,healthcheck}.js",
         "#aed6f1", fs_title=9, fs_body=7.5)

    container(ax, 64, 33, 17, 21, "Web3.py Bridge", color="#dde9f5")
    cell(ax, 65, 47, 15, 5.5, "BlockchainBridge",
         "fl/blockchain.py",
         "#aed6f1", fs_title=9, fs_body=7.5)
    cell(ax, 65, 39, 15, 6.5, "Bridge API",
         "submit_contribution()\nget_reputation()\ndistribute_audit()",
         "#aed6f1", fs_title=9, fs_body=7.5)
    cell(ax, 65, 34, 15, 4, "addresses.json",
         "loaded after deploy",
         "#fad7a0", fs_title=9, fs_body=7.5)

    container(ax, 84, 33, 11, 21, "Tests")
    cell(ax, 85, 45, 9, 8, "Unit (Python)",
         "136 PASS\npytest\ntests/unit -q",
         "#e8f6e8", fs_title=8.5, fs_body=7.5)
    cell(ax, 85, 34, 9, 9, "Contract (JS)",
         "tests/contracts/\nreward_flow.js\nHardhat+Mocha",
         "#e8f6e8", fs_title=8.5, fs_body=7.5)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 4 — DATA & ARTIFACTS
    # ═══════════════════════════════════════════════════════════════
    section_band(ax, 30, "4.  TẦNG DỮ LIỆU & KẾT QUẢ")

    container(ax, 5, 14, 28, 14, "Output (results/)", color="#fef7eb")
    cell(ax, 6.5, 22, 25, 4.5, "results/logs/",
         "354 CSV (schema v2) + .log",
         "#fae5d3", fs_title=8.5, fs_body=7.5)
    cell(ax, 6.5, 18.5, 25, 3, "Summary CSVs",
         "summary_metrics · fairness_metrics · stat_tests",
         "#fae5d3", fs_title=8.5, fs_body=7.5)
    cell(ax, 6.5, 15, 25, 3, "Plots + LaTeX",
         "results/plots/ (15+ PNG) · results/latex/",
         "#fae5d3", fs_title=8.5, fs_body=7.5)

    container(ax, 36, 14, 28, 14, "Input (data/)", color="#fef7eb")
    cell(ax, 37.5, 22, 25, 4.5, "torchvision.datasets",
         "MNIST 60k+10k · Fashion-MNIST 60k+10k",
         "#fae5d3", fs_title=8.5, fs_body=7.5)
    cell(ax, 37.5, 15, 25, 6.5, "Partition Strategies",
         "K1 IID · K2 Weak Non-IID\nK3 Dirichlet(α) · lognormal size imbalance",
         "#fae5d3", fs_title=8.5, fs_body=7.5)

    container(ax, 67, 14, 28, 14, "Experiment Matrix", color="#fef7eb")
    cell(ax, 68.5, 22, 25, 4.5, "Design",
         "2 datasets × 4 scenarios × 6 methods (M1–M6)",
         "#fae5d3", fs_title=8.5, fs_body=7.5)
    cell(ax, 68.5, 18, 11.5, 3.5, "Clean", "162 runs",
         "#d4efdf", fs_title=8.5, fs_body=8)
    cell(ax, 81.5, 18, 12, 3.5, "Attack", "192 runs",
         "#fadbd8", fs_title=8.5, fs_body=8)
    cell(ax, 68.5, 15, 25, 2.5, "Total",
         "354 runs · 30 rounds · ~26h compute",
         "#fae5d3", fs_title=8.5, fs_body=7.5)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 5 — DEPS
    # ═══════════════════════════════════════════════════════════════
    section_band(ax, 11, "5.  DEPENDENCIES & MÔI TRƯỜNG")

    cell(ax, 5, 2, 22, 8, "Python Stack",
         "flwr 1.8.0  ·  torch ≥ 2.1\nweb3 ≥ 6.15  ·  pandas\nmatplotlib · scipy",
         "#eaecee", fs_title=9.5, fs_body=8)
    cell(ax, 29, 2, 22, 8, "Node.js Stack",
         "hardhat ^2.22\nhardhat-toolbox · ethers v6\nMocha + Chai",
         "#eaecee", fs_title=9.5, fs_body=8)
    cell(ax, 53, 2, 22, 8, "Smart Contract Spec",
         "Solidity ^0.8.19\nSCALE = 1e6 (fixed-point)\nGas 150–500k · onlyOwner",
         "#eaecee", fs_title=9.5, fs_body=8)
    cell(ax, 77, 2, 18, 8, "Reproducibility",
         "Seeds {42, 123, 2024}\nDeterministic partition\nFilename schema v2",
         "#eaecee", fs_title=9.5, fs_body=8)

    # ═══════════════════════════════════════════════════════════════
    # ARROWS — data flow
    # ═══════════════════════════════════════════════════════════════
    # CLI → Runner → Analysis
    arrow(ax, (31, 83.5), (34, 83.5))
    arrow(ax, (66, 83.5), (69, 83.5))

    # Runner → FL Core
    arrow(ax, (50, 80), (50, 75.5), color="#0066aa", lw=2,
          label="config + seed", label_xy=(58, 77.5))

    # FL Core ↔ Bridge
    arrow(ax, (66, 67), (69, 67), color="#888", style="<->",
          label="metadata", label_xy=(67.5, 65))
    # Bridge sits in layer 3 — connect via vertical line from FL Core
    arrow(ax, (82, 59), (72, 54), color="#aa0066", lw=1.6,
          label="quality, data\n→ submit", label_xy=(89, 56))

    # Bridge ↔ Hardhat
    arrow(ax, (64, 47), (61, 47), color="#aa0066", lw=1.8, style="<->",
          label="RPC :8545", label_xy=(62.5, 50))

    # Hardhat → Contracts
    arrow(ax, (33, 43), (30, 43), color="#aa0066", lw=1.8,
          label="tx", label_xy=(31.5, 45.5))

    # Logger → Output
    arrow(ax, (82, 60), (19, 28), color="#0066aa", lw=1.4, ls="--",
          label="CSV per round", label_xy=(40, 47))

    # Output → Analysis (load)
    arrow(ax, (33, 21), (82, 80), color="#888", lw=1.2, ls="--",
          label="load CSV", label_xy=(58, 52))

    # Data → Clients
    arrow(ax, (50, 28), (50, 59), color="#0066aa", lw=1.3,
          label="DataLoader", label_xy=(56, 44))

    # ═══════════════════════════════════════════════════════════════
    # LEGEND
    # ═══════════════════════════════════════════════════════════════
    handles = [
        mpatches.Patch(color="#fdebd0", label="Orchestration"),
        mpatches.Patch(color="#fff9e6", label="Python FL module"),
        mpatches.Patch(color="#e8f6e8", label="Honest client / Tests"),
        mpatches.Patch(color="#fde2e2", label="Attack client"),
        mpatches.Patch(color="#d2b4de", label="Solidity contract"),
        mpatches.Patch(color="#aed6f1", label="Blockchain runtime"),
        mpatches.Patch(color="#fae5d3", label="Artifacts / Data"),
        mpatches.Patch(color="#eaecee", label="Dependencies"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005), frameon=False)

    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
