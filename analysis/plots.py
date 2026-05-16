"""
plots.py — All plotting functions for FL + Blockchain analysis.

Each public function:
  - accepts the master DataFrame + common kwargs (out_dir, dpi, smooth, ci)
  - iterates over datasets automatically
  - saves PNG files to out_dir
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from analysis.style import (
    apply_style, save_fig, moving_avg, ci95,
    CONFIG_COLORS, CONFIG_LABELS, ALPHA_COLORS, TYPE_COLORS,
    ordered_configs,
)
from fl.metrics import (
    jain_index, gini_coefficient,
    contribution_reward_correlation, fairness_gap,
)

log = logging.getLogger(__name__)
apply_style()

# ── helpers ───────────────────────────────────────────────────────────────────

def _datasets(df):
    return sorted(df["dataset"].unique())


def _scenarios(df):
    return sorted(df["scenario"].unique())


def _scenario_variants(df):
    """Return scenario labels without merging K3 dirichlet settings."""
    col = "scenario_variant" if "scenario_variant" in df.columns else "scenario"
    if col not in df.columns:
        return []
    order = {"K1": 1, "K2": 2, "K3": 3}
    cols = [c for c in ["scenario", col, "dirichlet_alpha"] if c in df.columns]
    variants = df[cols].drop_duplicates()
    if "scenario" in variants.columns:
        variants["_scenario_order"] = variants["scenario"].map(order).fillna(99)
    else:
        variants["_scenario_order"] = 99
    if "dirichlet_alpha" not in variants.columns:
        variants["dirichlet_alpha"] = 0.0
    variants = variants.sort_values(["_scenario_order", "dirichlet_alpha", col])
    return variants[col].astype(str).tolist()


def _variant_col(df):
    return "scenario_variant" if "scenario_variant" in df.columns else "scenario"


def _clean_view(df):
    if "attack_label" not in df.columns:
        return df
    clean = df[df["attack_label"] == "clean"]
    return clean if not clean.empty else df


def _plot_label_suffix(df):
    return " (clean runs)" if "attack_label" in df.columns and (df["attack_label"] == "attack").any() else ""


def _safe_name(value):
    return (
        str(value)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "p")
    )


def _acc_by_round(grp: pd.DataFrame) -> pd.Series:
    """Mean global_accuracy per round, dropping rounds with all-NaN values."""
    return (
        grp.groupby("round_num")["global_accuracy"]
        .mean()
        .sort_index()
        .dropna()
    )


def _plot_line_with_ci(ax, x, mean, lo, hi, color, label, smooth=0):
    if smooth > 1:
        mean = mean.rolling(smooth, min_periods=1, center=True).mean()
        lo   = lo.rolling(smooth, min_periods=1, center=True).mean()
        hi   = hi.rolling(smooth, min_periods=1, center=True).mean()
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, lo, hi, alpha=0.15, color=color)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_baseline_comparison(df, out_dir, dpi=200, smooth=0, ci=False, **_):
    """
    1a. Line chart: global accuracy vs round  (one subplot per scenario)
    1b. Grouped bar: final accuracy by config (one chart per dataset)
    """
    out_dir = Path(out_dir)

    for ds in _datasets(df):
        sub = _clean_view(df[df["dataset"] == ds])
        scenario_col = _variant_col(sub)
        scenarios = _scenario_variants(sub)
        configs   = ordered_configs(sub["config"].unique())

        # ── 1a: accuracy curves ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
        fig.suptitle(f"Global Accuracy vs Round — {ds}{_plot_label_suffix(df[df['dataset'] == ds])}", fontweight="bold")

        for col_i, sc in enumerate(scenarios):
            ax = axes[0][col_i]
            ax.set_title(str(sc))
            ax.set_xlabel("Round")
            ax.set_ylabel("Global Accuracy")

            sc_df = sub[sub[scenario_col].astype(str) == str(sc)]
            for cfg in configs:
                cfg_df = sc_df[sc_df["config"] == cfg]
                if cfg_df.empty:
                    continue
                by_round = cfg_df.groupby("round_num")["global_accuracy"]
                mean_s   = by_round.mean()
                rounds   = mean_s.index.values

                if ci:
                    # Smooth mean trước, rồi tính CI trên cùng window
                    if smooth > 1:
                        mean_smooth = mean_s.rolling(smooth, min_periods=1, center=True).mean()
                        std_s = by_round.std()
                        count_s = by_round.count().clip(lower=1)
                        std_smooth = std_s.rolling(smooth, min_periods=1, center=True).mean()
                        count_smooth = count_s.rolling(smooth, min_periods=1, center=True).mean()
                        margin = 1.96 * std_smooth / np.sqrt(count_smooth)
                        lo = mean_smooth - margin
                        hi = mean_smooth + margin
                        ax.plot(rounds, mean_smooth, color=CONFIG_COLORS.get(cfg, "#888888"),
                                label=CONFIG_LABELS.get(cfg, cfg))
                        ax.fill_between(rounds, lo, hi, alpha=0.15, color=CONFIG_COLORS.get(cfg, "#888888"))
                    else:
                        margin = 1.96 * by_round.std() / np.sqrt(by_round.count().clip(lower=1))
                        lo = mean_s - margin
                        hi = mean_s + margin
                        _plot_line_with_ci(ax, rounds, mean_s, lo, hi,
                                           CONFIG_COLORS.get(cfg, "#888888"),
                                           CONFIG_LABELS.get(cfg, cfg), 0)
                else:
                    y = moving_avg(mean_s, smooth) if smooth > 1 else mean_s
                    ax.plot(rounds, y, color=CONFIG_COLORS.get(cfg, "#888888"),
                            label=CONFIG_LABELS.get(cfg, cfg))

            ax.legend()
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        save_fig(fig, out_dir / f"baseline_accuracy_curve_{ds}.png", dpi)

        # 1b: final accuracy bar chart
        def _last_valid_acc(g):
            s = g.groupby("round_num")["global_accuracy"].mean().sort_index().dropna()
            return float(s.iloc[-1]) if not s.empty else np.nan

        final_acc = (
            sub.groupby([scenario_col, "config"])
            .apply(_last_valid_acc)
            .reset_index(name="final_accuracy")
        )

        x = np.arange(len(scenarios))
        width = 0.8 / max(len(configs), 1)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.set_title(f"Final Accuracy by Config — {ds}", fontweight="bold")
        ax2.set_ylabel("Final Global Accuracy")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios)
        ax2.set_ylim(0, 1.05)

        for i, cfg in enumerate(configs):
            vals = [
                final_acc[(final_acc[scenario_col].astype(str) == str(sc)) & (final_acc["config"] == cfg)]["final_accuracy"].values
                for sc in scenarios
            ]
            heights = [v[0] if len(v) > 0 else 0 for v in vals]
            offset  = (i - len(configs) / 2 + 0.5) * width
            bars = ax2.bar(x + offset, heights, width * 0.9,
                           color=CONFIG_COLORS.get(cfg, "#888888"),
                           label=CONFIG_LABELS.get(cfg, cfg))
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                             f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax2.legend()
        plt.tight_layout()
        save_fig(fig2, out_dir / f"baseline_final_accuracy_{ds}.png", dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FAIRNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fairness_analysis(df, out_dir, dpi=200, smooth=0, ci=False, **_):
    """
    2a. Reward distribution boxplot per config
    2b. Reward histogram
    2c. Jain fairness index bar chart
    2d. Reward vs Quality scatter plot
    """
    out_dir = Path(out_dir)

    for ds in _datasets(df):
        sub     = _clean_view(df[df["dataset"] == ds])
        scenario_col = _variant_col(sub)
        scenarios = _scenario_variants(sub)
        honest  = sub[sub["is_honest"]]
        configs = ordered_configs(sub["config"].unique())

        # ── 2a: boxplot ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
        fig.suptitle(f"Reward Distribution (Honest Clients) — {ds}{_plot_label_suffix(df[df['dataset'] == ds])}", fontweight="bold")

        for col_i, sc in enumerate(scenarios):
            ax = axes[0][col_i]
            ax.set_title(str(sc))
            ax.set_xlabel("Config")
            ax.set_ylabel("Reward (ETH)")
            data = [
                honest[(honest[scenario_col].astype(str) == str(sc)) & (honest["config"] == cfg)]["reward_eth"].dropna().values
                for cfg in configs
            ]
            bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)
            for patch, cfg in zip(bp["boxes"], configs):
                patch.set_facecolor(CONFIG_COLORS.get(cfg, "#AAAAAA"))
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(configs) + 1))
            ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=8)

        plt.tight_layout()
        save_fig(fig, out_dir / f"fairness_boxplot_{ds}.png", dpi)

        # ── 2b: reward histogram ──────────────────────────────────────────────
        fig2, axes2 = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4), squeeze=False)
        fig2.suptitle(f"Reward Histogram — {ds}", fontweight="bold")

        for col_i, cfg in enumerate(configs):
            ax = axes2[0][col_i]
            ax.set_title(CONFIG_LABELS.get(cfg, cfg))
            ax.set_xlabel("Reward (ETH)")
            ax.set_ylabel("Count")
            vals = honest[honest["config"] == cfg]["reward_eth"].dropna()
            if not vals.empty:
                ax.hist(vals, bins=30, color=CONFIG_COLORS.get(cfg, "#888888"), alpha=0.75, edgecolor="white")

        plt.tight_layout()
        save_fig(fig2, out_dir / f"fairness_histogram_{ds}.png", dpi)

        # ── 2c: Jain index bar chart ──────────────────────────────────────────
        jain_records = []
        for (sc, cfg, alpha), grp in honest.groupby([scenario_col, "config", "alpha"]):
            r = grp["reward_eth"].dropna().values
            jain_records.append({
                scenario_col: sc, "config": cfg, "alpha": alpha,
                "jain": jain_index(r),
                "gini": gini_coefficient(r),
            })
        if not jain_records:
            continue

        jdf = pd.DataFrame(jain_records)
        # Average over alpha for the bar chart
        jmean = jdf.groupby([scenario_col, "config"])[["jain", "gini"]].mean().reset_index()

        scenarios = sorted(jmean[scenario_col].astype(str).unique())
        x = np.arange(len(scenarios))
        width = 0.8 / max(len(configs), 1)

        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
        fig3.suptitle(f"Jain Index & Gini Coefficient — {ds}", fontweight="bold")

        for i, cfg in enumerate(configs):
            offset = (i - len(configs) / 2 + 0.5) * width
            sub_c  = jmean[jmean["config"] == cfg]
            j_vals = [sub_c[sub_c[scenario_col].astype(str) == str(sc)]["jain"].values for sc in scenarios]
            g_vals = [sub_c[sub_c[scenario_col].astype(str) == str(sc)]["gini"].values for sc in scenarios]
            j_h = [v[0] if len(v) > 0 else 0 for v in j_vals]
            g_h = [v[0] if len(v) > 0 else 0 for v in g_vals]
            ax3a.bar(x + offset, j_h, width * 0.9, color=CONFIG_COLORS.get(cfg, "#888"), label=CONFIG_LABELS.get(cfg, cfg))
            ax3b.bar(x + offset, g_h, width * 0.9, color=CONFIG_COLORS.get(cfg, "#888"), label=CONFIG_LABELS.get(cfg, cfg))

        for ax, title, ylabel in [
            (ax3a, "Jain Fairness Index (↑ better)", "Jain Index"),
            (ax3b, "Gini Coefficient (↓ better)", "Gini"),
        ]:
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend()
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        save_fig(fig3, out_dir / f"fairness_jain_gini_{ds}.png", dpi)

        # ── 2d: Reward vs Quality scatter ─────────────────────────────────────
        fig4, axes4 = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
        fig4.suptitle(f"Reward vs Quality (Honest Clients) — {ds}", fontweight="bold")

        for col_i, sc in enumerate(scenarios):
            ax = axes4[0][col_i]
            ax.set_title(str(sc))
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Reward (ETH)")
            sc_honest = honest[honest[scenario_col].astype(str) == str(sc)]
            for cfg in configs:
                cfg_data = sc_honest[sc_honest["config"] == cfg]
                if cfg_data.empty:
                    continue
                q = cfg_data["quality"].dropna()
                r = cfg_data["reward_eth"].dropna()
                # Align by index
                common = q.index.intersection(r.index)
                if len(common) > 0:
                    ax.scatter(q[common], r[common], alpha=0.3, s=10,
                               color=CONFIG_COLORS.get(cfg, "#888"),
                               label=CONFIG_LABELS.get(cfg, cfg))
            ax.legend(fontsize=7)

        plt.tight_layout()
        save_fig(fig4, out_dir / f"fairness_reward_vs_quality_{ds}.png", dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 3. REPUTATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_reputation_analysis(df, out_dir, dpi=200, smooth=0, **_):
    """
    3a. Average reputation over rounds per config
    3b. Honest vs malicious reputation trends
    """
    out_dir = Path(out_dir)

    for ds in _datasets(df):
        sub_all = df[df["dataset"] == ds]
        sub     = _clean_view(sub_all)
        scenario_col = _variant_col(sub)
        configs = ordered_configs(sub["config"].unique())

        for sc in _scenario_variants(sub):
            sc_df = sub[sub[scenario_col].astype(str) == str(sc)]

            # ── 3a: avg reputation per config ─────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title(f"Avg Reputation Over Rounds — {ds} / {sc}", fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Reputation")

            for cfg in configs:
                cfg_df  = sc_df[sc_df["config"] == cfg]
                by_r    = cfg_df.groupby("round_num")["reputation"].mean().sort_index()
                y       = moving_avg(by_r, smooth) if smooth > 1 else by_r
                ax.plot(by_r.index, y, color=CONFIG_COLORS.get(cfg, "#888"),
                        label=CONFIG_LABELS.get(cfg, cfg))

            ax.legend()
            plt.tight_layout()
            save_fig(fig, out_dir / f"reputation_avg_{ds}_{_safe_name(sc)}.png", dpi)

            # ── 3b: honest vs malicious ────────────────────────────────────────
            attack_sc_df = sub_all[
                (sub_all[_variant_col(sub_all)].astype(str) == str(sc))
                & (sub_all.get("has_attack", False) == True)
            ]
            if "client_type" not in attack_sc_df.columns:
                continue
            types = attack_sc_df["client_type"].unique()
            if len(types) < 2:
                continue

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.set_title(f"Reputation: Honest vs Malicious — {ds} / {sc}", fontweight="bold")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Reputation")

            for ctype in ["honest", "free_rider", "lazy"]:
                type_df = attack_sc_df[attack_sc_df["client_type"] == ctype]
                if type_df.empty:
                    continue
                by_r = type_df.groupby("round_num")["reputation"].mean().sort_index()
                y    = moving_avg(by_r, smooth) if smooth > 1 else by_r
                ax2.plot(by_r.index, y, color=TYPE_COLORS.get(ctype, "#888"),
                         label=ctype.replace("_", " ").title(),
                         linestyle="--" if ctype != "honest" else "-")

            ax2.legend()
            plt.tight_layout()
            save_fig(fig2, out_dir / f"reputation_honest_vs_malicious_{ds}_{_safe_name(sc)}.png", dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ATTACK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_attack_analysis(df, out_dir, dpi=200, smooth=0, **_):
    """
    4a. Accuracy: attack run vs clean run
    4b. Reward share: honest vs malicious
    """
    out_dir = Path(out_dir)

    if "has_attack" not in df.columns:
        log.info("No attack detection column — skipping attack analysis.")
        return

    for ds in _datasets(df):
        sub = df[df["dataset"] == ds]
        attack_df = sub[sub["has_attack"]]
        clean_df  = sub[~sub["has_attack"]]

        if attack_df.empty:
            log.info("[%s] No attack runs detected — skipping.", ds)
            continue

        scenario_col = _variant_col(sub)
        attack_pairs = (
            attack_df[[scenario_col, "config"]]
            .drop_duplicates()
            .sort_values([scenario_col, "config"])
            .to_dict("records")
        )

        # ── 4a: accuracy comparison ────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(attack_pairs), figsize=(5 * len(attack_pairs), 4), squeeze=False)
        fig.suptitle(f"Accuracy: Clean vs Attack Runs — {ds}", fontweight="bold")

        for col_i, pair in enumerate(attack_pairs):
            sc = pair[scenario_col]
            cfg = pair["config"]
            ax = axes[0][col_i]
            ax.set_title(f"{sc} / {CONFIG_LABELS.get(cfg, cfg)}")
            ax.set_xlabel("Round")
            ax.set_ylabel("Global Accuracy")
            ax.set_ylim(0, 1.05)

            for label, src, color, ls in [
                ("Clean",  clean_df,  "#4A90D9", "-"),
                ("Attack", attack_df, "#E07070", "--"),
            ]:
                c_df = src[
                    (src["config"] == cfg)
                    & (src[scenario_col].astype(str) == str(sc))
                ]
                if c_df.empty:
                    continue
                by_r = (
                    c_df.groupby("round_num")["global_accuracy"]
                    .mean().sort_index().dropna()
                )
                y = moving_avg(by_r, smooth) if smooth > 1 else by_r
                ax.plot(by_r.index, y, color=color, linestyle=ls, label=label)

            ax.legend()

        plt.tight_layout()
        save_fig(fig, out_dir / f"attack_accuracy_{ds}.png", dpi)

        # ── 4b: reward share ───────────────────────────────────────────────────
        if "client_type" not in attack_df.columns:
            continue

        reward_share = (
            attack_df.groupby([scenario_col, "config", "client_type"])["reward_eth"]
            .sum()
            .reset_index()
        )

        labels = [f"{p[scenario_col]}\n{p['config']}" for p in attack_pairs]
        x = np.arange(len(attack_pairs))
        bottom = np.zeros(len(attack_pairs), dtype=float)
        fig2, ax2 = plt.subplots(figsize=(max(6, 1.8 * len(attack_pairs)), 4))
        fig2.suptitle(f"Reward Share by Client Type — {ds}", fontweight="bold")

        for ctype in ["honest", "free_rider", "lazy"]:
            shares = []
            for pair in attack_pairs:
                sub_rs = reward_share[
                    (reward_share[scenario_col].astype(str) == str(pair[scenario_col]))
                    & (reward_share["config"] == pair["config"])
                ]
                total = float(sub_rs["reward_eth"].sum())
                val = float(sub_rs.loc[sub_rs["client_type"] == ctype, "reward_eth"].sum())
                shares.append(val / total if total > 0 else 0.0)
            ax2.bar(x, shares, bottom=bottom, color=TYPE_COLORS.get(ctype, "#AAAAAA"),
                    label=ctype.replace("_", " ").title())
            bottom += np.asarray(shares)

        ax2.set_ylabel("Reward Share")
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()

        plt.tight_layout()
        save_fig(fig2, out_dir / f"attack_reward_share_{ds}.png", dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALPHA SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_alpha_sensitivity(df, out_dir, dpi=200, **_):
    """
    5a. Alpha vs final accuracy (line per config)
    5b. Alpha vs Jain fairness index
    5c. Alpha vs reward std (stability)
    """
    out_dir = Path(out_dir)

    for ds in _datasets(df):
        sub     = _clean_view(df[df["dataset"] == ds])
        configs = ordered_configs(sub["config"].unique())
        honest  = sub[sub["is_honest"]]

        # Per-(config, alpha) final accuracy — use last non-NaN round
        def _last_valid(g):
            s = g.groupby("round_num")["global_accuracy"].mean().sort_index().dropna()
            return float(s.iloc[-1]) if not s.empty else np.nan

        final_acc = (
            sub.groupby(["config", "alpha"])
            .apply(_last_valid)
            .reset_index(name="final_accuracy")
        )

        # Per-(config, alpha) fairness & stability
        fairness_rows = []
        for (cfg, alpha), grp in honest.groupby(["config", "alpha"]):
            r = grp["reward_eth"].dropna().values
            fairness_rows.append({
                "config": cfg, "alpha": alpha,
                "jain": jain_index(r),
                "reward_std": float(np.std(r)) if len(r) > 1 else np.nan,
            })
        fdf = pd.DataFrame(fairness_rows)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Alpha Sensitivity — {ds}", fontweight="bold")

        metrics = [
            (axes[0], final_acc, "final_accuracy", "Final Accuracy", "Accuracy"),
            (axes[1], fdf,       "jain",           "Jain Fairness Index (↑ better)", "Jain Index"),
            (axes[2], fdf,       "reward_std",     "Reward Std (↓ = stable)", "Reward Std"),
        ]

        for ax, src, col, title, ylabel in metrics:
            ax.set_title(title)
            ax.set_xlabel("Alpha (α)")
            ax.set_ylabel(ylabel)
            for cfg in configs:
                c_src = src[src["config"] == cfg].sort_values("alpha")
                if c_src.empty or col not in c_src.columns:
                    continue
                ax.plot(
                    c_src["alpha"], c_src[col],
                    marker="o", color=CONFIG_COLORS.get(cfg, "#888"),
                    label=CONFIG_LABELS.get(cfg, cfg),
                )
            ax.legend()

        plt.tight_layout()
        save_fig(fig, out_dir / f"alpha_sensitivity_{ds}.png", dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CONVERGENCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_comparison(df, out_dir, dpi=200, **_):
    """
    6a. Convergence round bar chart per config/scenario
    6b. Final accuracy vs convergence round scatter
    """
    out_dir = Path(out_dir)

    for ds in _datasets(df):
        sub = _clean_view(df[df["dataset"] == ds])
        scenario_col = _variant_col(sub)
        scenarios = _scenario_variants(sub)
        configs = ordered_configs(sub["config"].unique())

        # Compute convergence per (scenario variant, config, alpha)
        conv_records = []
        for (sc, cfg, alpha), grp in sub.groupby([scenario_col, "config", "alpha"]):
            acc = grp.groupby("round_num")["global_accuracy"].mean().sort_index().dropna()
            if acc.empty:
                continue
            from fl.metrics import convergence_round as _cr
            rounds_list = [int(r) for r in acc.index]
            cr = _cr(acc.to_list(), rounds=rounds_list, peak_ratio=0.95, patience=5)
            conv_records.append({
                scenario_col: sc, "config": cfg, "alpha": alpha,
                "convergence_round": cr if cr is not None else np.nan,
                "final_accuracy": float(acc.iloc[-1]),
            })

        if not conv_records:
            continue

        cdf = pd.DataFrame(conv_records)
        # Mean over alpha
        cmean = cdf.groupby([scenario_col, "config"])[["convergence_round", "final_accuracy"]].mean().reset_index()

        # ── 6a: convergence round bar chart ───────────────────────────────────
        x = np.arange(len(scenarios))
        width = 0.8 / max(len(configs), 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"Convergence Round by Config — {ds}", fontweight="bold")
        ax.set_ylabel("Convergence Round (95% peak, patience=5)")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)

        for i, cfg in enumerate(configs):
            sub_c = cmean[cmean["config"] == cfg]
            vals = [sub_c[sub_c[scenario_col].astype(str) == str(sc)]["convergence_round"].values for sc in scenarios]
            heights = [v[0] if len(v) > 0 and not np.isnan(v[0]) else 0 for v in vals]
            offset = (i - len(configs) / 2 + 0.5) * width
            ax.bar(x + offset, heights, width * 0.9,
                   color=CONFIG_COLORS.get(cfg, "#888"), label=CONFIG_LABELS.get(cfg, cfg))

        ax.legend()
        plt.tight_layout()
        save_fig(fig, out_dir / f"convergence_round_{ds}.png", dpi)

        # ── 6b: final accuracy vs convergence round scatter ───────────────────
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.set_title(f"Accuracy vs Convergence Speed — {ds}", fontweight="bold")
        ax2.set_xlabel("Convergence Round (lower = faster)")
        ax2.set_ylabel("Final Accuracy")

        for cfg in configs:
            sub_c = cdf[cdf["config"] == cfg].dropna(subset=["convergence_round", "final_accuracy"])
            if sub_c.empty:
                continue
            ax2.scatter(sub_c["convergence_round"], sub_c["final_accuracy"],
                        alpha=0.6, s=30, color=CONFIG_COLORS.get(cfg, "#888"),
                        label=CONFIG_LABELS.get(cfg, cfg))

        ax2.legend()
        plt.tight_layout()
        save_fig(fig2, out_dir / f"convergence_scatter_{ds}.png", dpi)
