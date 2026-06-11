"""
gen_advanced_plots.py — Advanced visualizations sâu hơn cho báo cáo.

Sinh các plot phân tích chi tiết per-client per-round, vượt qua các bar chart đơn giản:
  fig7  — Per-client cumulative reward heatmap (6 methods × 10 clients)
  fig8  — Attacker reward share trajectory (timeline 30 rounds)
  fig9  — Lorenz curve (reward inequality, có Gini overlay)
  fig10 — CSRA composite weight decomposition (β·q + γ·d + δ·rep stacked)
  fig11 — Detection scatter: robust_z honest vs attacker (with threshold)
  fig12 — Per-round Jain trajectory với seed-CI
  fig13 — Top-k reward concentration curve
  fig14 — Statistical significance heatmap (p-values)

Dùng raw logs trong results/logs/.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
OUT_DIR = ROOT / "results" / "plots" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "fedavg+equal", "fedavg+data", "fedavg+quality",
    "fedavg+csra", "csra_dcd+equal", "csra_dcd+csra",
]
METHOD_LABEL = {
    "fedavg+equal":   "M1\nFedAvg\nEqual",
    "fedavg+data":    "M2\nFedAvg\nData",
    "fedavg+quality": "M3\nFedAvg\nQuality",
    "fedavg+csra":    "M4\nFedAvg\nCSRA",
    "csra_dcd+equal": "M5\nCSRA-DCD\nEqual",
    "csra_dcd+csra":  "M6\nCSRA-DCD\nCSRA",
}
METHOD_SHORT = {
    "fedavg+equal": "M1", "fedavg+data": "M2", "fedavg+quality": "M3",
    "fedavg+csra": "M4", "csra_dcd+equal": "M5", "csra_dcd+csra": "M6",
}
METHOD_COLOR = {
    "fedavg+equal": "#9aa0a6", "fedavg+data": "#86c986",
    "fedavg+quality": "#f3b860", "fedavg+csra": "#c14b6e",
    "csra_dcd+equal": "#7a6cc6", "csra_dcd+csra": "#1a5a8c",
}

# Regex parse filename
FNAME_RE = re.compile(
    r"^(?P<dataset>fashion_mnist|cifar10|mnist)_(?P<scenario>K\d+)"
    r"(?:_da(?P<dirichlet_raw>\d{3}))?_(?P<agg>fedavg|trimmed|csra_dcd)"
    r"_(?P<reward>equal|data|quality|csra)"
    r"_b(?P<beta>\d{2})g(?P<gamma>\d{2})d(?P<delta>\d{2})"
    r"_s(?P<seed>\d+)_(?P<attack>clean|free_rider|lazy|label_noise|sign_flip)"
    r"_(?P<ts>\d{8}_\d{6})\.csv$"
)


def load_runs(dataset: str, scenario: str, attack: str,
              dirichlet: str | None = None, beta: int = 50,
              gamma: int = 30, delta: int = 20) -> dict[str, pd.DataFrame]:
    """Load all 6 methods for a specific (dataset, scenario, attack) combo, averaged across seeds."""
    out: dict[str, list[pd.DataFrame]] = {}
    for f in LOG_DIR.glob("*.csv"):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        if m["dataset"] != dataset or m["scenario"] != scenario or m["attack"] != attack:
            continue
        if dirichlet is not None and (m["dirichlet_raw"] != dirichlet):
            continue
        method = f"{m['agg']}+{m['reward']}"
        # For CSRA reward, restrict to canonical β=0.5 (avoid sweep contamination)
        if m["reward"] == "csra":
            if int(m["beta"]) != beta:
                continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        out.setdefault(method, []).append(df)
    # Concat across seeds
    return {m: pd.concat(dfs, ignore_index=True) for m, dfs in out.items() if dfs}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Per-client cumulative reward heatmap (6 methods × 10 clients)
# ─────────────────────────────────────────────────────────────────────────────

def fig7_per_client_heatmap():
    """Heatmap: total reward each client received over 30 rounds, per method.
    Attackers (client 8, 9) highlighted by red border."""
    runs = load_runs("mnist", "K3", "free_rider", dirichlet="010")
    if not runs:
        log.warning("fig7: no runs found")
        return

    # Compute per-client total reward (mean across seeds)
    matrix = []
    methods_present = [m for m in METHOD_ORDER if m in runs]
    for method in methods_present:
        df = runs[method]
        per_client = df.groupby(["seed", "client_id"])["reward_eth"].sum().reset_index()
        avg = per_client.groupby("client_id")["reward_eth"].mean()
        matrix.append([avg.get(c, 0.0) for c in range(10)])

    M = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    cmap = LinearSegmentedColormap.from_list("rwd", ["#f5f5f5", "#1a5a8c"], N=256)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=M.max())

    # Annotate values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "white" if M[i, j] > M.max() * 0.6 else "#222"
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"C{i}" for i in range(10)], fontsize=10)
    ax.set_yticks(range(len(methods_present)))
    ax.set_yticklabels([METHOD_SHORT[m] for m in methods_present], fontsize=11)

    # Highlight attackers (clients 8, 9) with red rectangle on top
    for atk in (8, 9):
        ax.add_patch(plt.Rectangle((atk - 0.5, -0.5), 1, len(methods_present),
                                    fill=False, edgecolor="red", lw=2.5,
                                    linestyle="--"))
    ax.text(8.5, -0.85, "ATTACKERS", ha="center", color="red",
            fontweight="bold", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Total reward (ETH) trên 30 rounds", fontsize=10)

    ax.set_title("Phân phối reward tích lũy per-client × method — MNIST K3@α=0.1, free-rider attack\n"
                 "M2 và M4 phạt attacker mạnh nhất (cột 8, 9 sáng nhạt); M1, M5 chia đều mọi client",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Client ID", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "fig7_per_client_heatmap.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Attacker reward share trajectory
# ─────────────────────────────────────────────────────────────────────────────

def fig8_attacker_timeline():
    """Time-series: reward_share của clients 8+9 (attackers) qua 30 rounds, 6 methods overlay."""
    runs = load_runs("mnist", "K3", "free_rider", dirichlet="010")
    if not runs:
        log.warning("fig8: no runs found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # Per-round attacker share = sum(reward of clients 8+9) / sum(reward all clients) per round
    for ax, scenario_attack in zip(axes, [("free_rider", "Free-Rider"), ("label_noise", "Label-Noise")]):
        atk, title = scenario_attack
        runs2 = load_runs("mnist", "K3", atk, dirichlet="010")
        for method in METHOD_ORDER:
            if method not in runs2:
                continue
            df = runs2[method]
            # Per round attacker share averaged across seeds
            def attacker_share(g):
                total = g["reward_eth"].sum()
                if total <= 0:
                    return 0.0
                atks = g[g["client_id"].isin([8, 9])]["reward_eth"].sum()
                return atks / total
            ts = (
                df.groupby(["seed", "round"]).apply(attacker_share, include_groups=False)
                .reset_index(name="share")
                .groupby("round")["share"]
                .agg(["mean", "std"]).reset_index()
            )
            ax.plot(ts["round"], ts["mean"], color=METHOD_COLOR[method],
                    label=METHOD_SHORT[method], lw=2)
            ax.fill_between(ts["round"], ts["mean"] - ts["std"], ts["mean"] + ts["std"],
                             color=METHOD_COLOR[method], alpha=0.12)
        ax.axhline(0.2, color="red", ls="--", lw=1, alpha=0.6,
                   label="Random share = 20%")
        ax.set_title(f"{title} attackers (client 8, 9)", fontsize=11)
        ax.set_xlabel("Round", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.35)
    axes[0].set_ylabel("Attacker reward share", fontsize=11)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("Trajectory của reward đến tay attacker qua từng round — MNIST K3@α=0.1\n"
                 "M4 (CSRA) liên tục duy trì share < 10% trong khi M1, M5 giữ ở mức 20%",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig8_attacker_timeline.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — Lorenz curve (cumulative reward distribution)
# ─────────────────────────────────────────────────────────────────────────────

def fig9_lorenz_curve():
    """Lorenz curve: x = % of clients ranked by reward (ascending), y = cumulative % of reward.
    Diagonal = perfect equality."""
    runs = load_runs("mnist", "K3", "clean", dirichlet="010")
    if not runs:
        log.warning("fig9: no clean runs")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect equality")

    for method in METHOD_ORDER:
        if method not in runs:
            continue
        df = runs[method]
        # Sum reward per client (across rounds + seeds)
        per_client = df.groupby("client_id")["reward_eth"].sum().sort_values().values
        if per_client.sum() <= 0:
            continue
        cum = np.cumsum(per_client) / per_client.sum()
        x = np.linspace(0, 1, len(cum) + 1)
        cum = np.insert(cum, 0, 0)
        gini = 1 - 2 * np.trapezoid(cum, x)  # discrete Gini approx
        ax.plot(x, cum, color=METHOD_COLOR[method], lw=2.5,
                label=f"{METHOD_SHORT[method]}  (Gini={gini:.3f})")

    ax.set_xlabel("Tỷ lệ client tích lũy (sắp xếp theo reward tăng)", fontsize=11)
    ax.set_ylabel("Tỷ lệ reward tích lũy", fontsize=11)
    ax.set_title("Lorenz Curve — phân phối reward giữa 10 clients\n"
                 "MNIST K3@α=0.1, clean runs. Càng cong xa đường chéo = càng bất bình đẳng",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    out = OUT_DIR / "fig9_lorenz_curve.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — CSRA composite weight decomposition
# ─────────────────────────────────────────────────────────────────────────────

def fig10_csra_decomposition():
    """Stacked bar: β·q̃ + γ·d̃ + δ·ρ̃ for each client of M4, gộp các round."""
    runs = load_runs("mnist", "K3", "clean", dirichlet="010")
    if "fedavg+csra" not in runs:
        log.warning("fig10: no M4 runs")
        return

    df = runs["fedavg+csra"]
    # Normalize each component per round (same as inside the policy)
    rows = []
    for (seed, rnd), grp in df.groupby(["seed", "round"]):
        q = grp["quality_score"].clip(lower=0).values
        d = grp["data_size"].astype(float).values
        r = grp["reputation"].clip(lower=0).values
        q_norm = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
        d_norm = d / d.sum() if d.sum() > 0 else np.ones_like(d) / len(d)
        r_norm = r / r.sum() if r.sum() > 0 else np.ones_like(r) / len(r)
        beta, gamma, delta = 0.5, 0.3, 0.2
        for i, cid in enumerate(grp["client_id"].values):
            rows.append({"client_id": cid,
                         "q_contrib": beta * q_norm[i],
                         "d_contrib": gamma * d_norm[i],
                         "r_contrib": delta * r_norm[i]})

    decomp = pd.DataFrame(rows).groupby("client_id").mean()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    clients = decomp.index.values
    width = 0.7
    p1 = ax.bar(clients, decomp["q_contrib"], width=width,
                color="#c14b6e", label=f"β·q̃ (β=0.5, quality)")
    p2 = ax.bar(clients, decomp["d_contrib"], width=width,
                bottom=decomp["q_contrib"], color="#86c986",
                label=f"γ·d̃ (γ=0.3, data size)")
    p3 = ax.bar(clients, decomp["r_contrib"], width=width,
                bottom=decomp["q_contrib"] + decomp["d_contrib"],
                color="#7a6cc6", label=f"δ·ρ̃ (δ=0.2, reputation)")

    # Annotate total
    totals = decomp.sum(axis=1)
    for cid, total in totals.items():
        ax.text(cid, total + 0.005, f"{total:.3f}", ha="center", fontsize=8.5)

    ax.set_xticks(clients)
    ax.set_xticklabels([f"C{c}" for c in clients])
    ax.set_xlabel("Client ID", fontsize=11)
    ax.set_ylabel("Composite weight W_i (avg qua các round)", fontsize=11)
    ax.set_title("Phân rã trọng số CSRAReward 3-chiều — M4 trên MNIST K3@α=0.1 (clean)\n"
                 "Thấy rõ thành phần quality dominant, data size đóng vai trò ổn định, reputation cố định 0.5",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig10_csra_decomposition.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — Detection scatter (robust_z honest vs malicious)
# ─────────────────────────────────────────────────────────────────────────────

def fig11_detection_scatter():
    """Scatter plot of robust_z values: honest clients vs each attacker type. Threshold line at z=3."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    attacks = ["free_rider", "lazy", "label_noise", "sign_flip"]
    for ax, atk in zip(axes, attacks):
        all_runs = load_runs("mnist", "K3", atk, dirichlet="010")
        # Only use csra_dcd methods (they have robust_z)
        if "csra_dcd+csra" not in all_runs:
            continue
        df = all_runs["csra_dcd+csra"].dropna(subset=["robust_z"])
        honest = df[df["client_type"] == "honest"]["robust_z"].values
        mal = df[df["client_type"] == atk]["robust_z"].values

        # Jitter x
        rng = np.random.default_rng(42)
        x_h = rng.normal(0, 0.05, len(honest))
        x_m = rng.normal(1, 0.05, len(mal))
        ax.scatter(x_h, honest, alpha=0.35, s=15, color="#2a8a4e", label="Honest")
        ax.scatter(x_m, mal, alpha=0.6, s=25, color="#c14b6e",
                   marker="^", label=atk.replace("_", " ").title())
        ax.axhline(3.0, color="red", ls="--", lw=1.5,
                   label="Threshold z=3.0" if atk == "free_rider" else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Honest", atk.replace("_", "\n")], fontsize=10)
        ax.set_title(atk.replace("_", " "), fontsize=11)
        ax.grid(True, alpha=0.3)
        if atk == "free_rider":
            ax.set_ylabel("MAD Robust z-score", fontsize=11)

    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Phân bố robust z-score — Honest vs Attacker (M6 trên MNIST K3@α=0.1)\n"
                 "Chỉ sign-flip có z vượt ngưỡng 3.0; 3 attack còn lại nằm trong phân phối honest",
                 fontsize=11, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = OUT_DIR / "fig11_detection_scatter.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 — Per-round Jain trajectory với seed-CI
# ─────────────────────────────────────────────────────────────────────────────

def fig12_jain_trajectory():
    """Per-round Jain index, shaded by ±std across seeds. So sánh M1/M4/M6 on K3@0.1 clean."""
    runs = load_runs("mnist", "K3", "clean", dirichlet="010")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        df = runs[method]
        # Per round Jain: J = (sum r)^2 / (n * sum r^2)
        def jain(rewards):
            r = np.asarray(rewards, dtype=float)
            if r.sum() <= 0:
                return np.nan
            return r.sum()**2 / (len(r) * (r**2).sum())
        per_round = (
            df.groupby(["seed", "round"])["reward_eth"].apply(jain)
            .reset_index(name="jain")
            .groupby("round")["jain"]
            .agg(["mean", "std"]).reset_index()
        )
        ax.plot(per_round["round"], per_round["mean"], color=METHOD_COLOR[method],
                lw=2.2, label=METHOD_SHORT[method])
        if per_round["std"].notna().any():
            ax.fill_between(per_round["round"],
                             per_round["mean"] - per_round["std"],
                             per_round["mean"] + per_round["std"],
                             color=METHOD_COLOR[method], alpha=0.15)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Jain Index per round", fontsize=11)
    ax.set_title("Diễn biến Fairness Jain qua từng round — MNIST K3@α=0.1, clean\n"
                 "M3 (Quality) dao động mạnh do quality score nhiễu; M4 (CSRA) ổn định hơn",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right", ncol=3, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig12_jain_trajectory.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13 — Top-k reward concentration (Pareto principle)
# ─────────────────────────────────────────────────────────────────────────────

def fig13_topk_concentration():
    """Curve: top-k clients chiếm bao nhiêu % tổng reward. 80-20 line as reference."""
    runs = load_runs("mnist", "K3", "clean", dirichlet="010")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_clients = 10
    ks = np.arange(1, n_clients + 1)
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        df = runs[method]
        per_client = df.groupby("client_id")["reward_eth"].sum().sort_values(ascending=False).values
        total = per_client.sum()
        if total <= 0:
            continue
        topk_share = np.cumsum(per_client) / total
        ax.plot(ks / n_clients * 100, topk_share * 100,
                "o-", color=METHOD_COLOR[method], lw=2,
                label=METHOD_SHORT[method])

    # 80-20 reference
    ax.axhline(80, color="red", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(20, color="red", ls=":", lw=0.8, alpha=0.6)
    ax.text(22, 81, "80-20 Pareto", color="red", fontsize=9)

    ax.set_xlabel("Top-k% clients (sắp xếp theo reward giảm dần)", fontsize=11)
    ax.set_ylabel("Tỷ lệ reward được nhóm top-k% nắm giữ (%)", fontsize=11)
    ax.set_title("Tập trung reward (Pareto curve) — MNIST K3@α=0.1, clean\n"
                 "Đường gần đường chéo = phân phối đều; cong càng nhanh = tập trung càng cao",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out = OUT_DIR / "fig13_topk_concentration.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14 — Statistical significance heatmap (p-values)
# ─────────────────────────────────────────────────────────────────────────────

def fig14_significance_heatmap():
    """Heatmap p-values Mann-Whitney U trên Jain Index giữa các method pairs."""
    summary = pd.read_csv(ROOT / "results" / "summary_metrics.csv")
    clean = summary[summary.attack_label == "clean"].copy()
    clean = clean[(clean.reward_policy != "csra") | (clean.beta == 0.5)]

    methods = METHOD_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metrics = [("jain", "Jain Index (clean)"),
               ("reward_leakage", "Reward Leakage (attack)")]

    for ax, (metric, title) in zip(axes, metrics):
        src = clean if metric == "jain" else summary[summary.attack_label == "attack"]
        src = src if metric == "jain" else src[(src.reward_policy != "csra") | (src.beta == 0.5)]
        P = np.full((len(methods), len(methods)), np.nan)
        for i, ma in enumerate(methods):
            for j, mb in enumerate(methods):
                if i == j:
                    continue
                va = src[src.method == ma][metric].dropna().values
                vb = src[src.method == mb][metric].dropna().values
                if len(va) < 2 or len(vb) < 2:
                    continue
                try:
                    _, p = mannwhitneyu(va, vb, alternative="two-sided")
                    P[i, j] = p
                except Exception:
                    pass
        # Display: -log10(p) so significant = bright
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = -np.log10(np.where(P > 0, P, np.nan))
        im = ax.imshow(logp, cmap="YlOrRd", vmin=0, vmax=4)
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="grey")
                    continue
                p = P[i, j]
                if np.isnan(p):
                    continue
                txt = f"{p:.2g}"
                color = "white" if logp[i, j] > 2 else "#333"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels([METHOD_SHORT[m] for m in methods], fontsize=10)
        ax.set_yticklabels([METHOD_SHORT[m] for m in methods], fontsize=10)
        ax.set_title(f"{title}\nMann-Whitney U p-values  (đậm = significant)", fontsize=11)

    fig.suptitle("Statistical significance giữa các method pairs",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig14_significance_heatmap.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig7_per_client_heatmap()
    fig8_attacker_timeline()
    fig9_lorenz_curve()
    fig10_csra_decomposition()
    fig11_detection_scatter()
    fig12_jain_trajectory()
    fig13_topk_concentration()
    fig14_significance_heatmap()
    log.info("All advanced plots saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
