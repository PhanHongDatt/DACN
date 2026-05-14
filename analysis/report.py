"""
report.py — Auto-generate a Markdown analysis report with full B-vs-C comparison.
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONFIG_LABELS   = {
    "A": "Traditional FL",
    "B": "Blockchain Baseline",
    "C": "CSRA Reward",
    "C-CSRA": "CSRA-DCD Reward",
    "C-CSRA-Opt": "CSRA-DCD Reward (Optimized)",
    "TrimmedMean": "TrimmedMean Robust FL",
}
SCENARIO_LABELS = {"K1": "IID", "K2": "Weak Non-IID", "K3": "Dirichlet Non-IID"}


def _md_table(df: pd.DataFrame, float_fmt=".4f") -> str:
    """Convert DataFrame to Markdown table, NaN shown as '—'."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _build_config_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparison table: for each (dataset, scenario, alpha),
    show final_accuracy and mean_reward for configs A, B, C side-by-side.
    Also compute delta columns: (C - B) and (C - A).
    """
    if summary.empty:
        return pd.DataFrame()

    pivot = summary.pivot_table(
        index=[c for c in ["dataset", "scenario", "dirichlet_alpha", "alpha"] if c in summary.columns],
        columns="config",
        values=["final_accuracy", "mean_reward", "reward_std", "mean_reputation"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{cfg}" for metric, cfg in pivot.columns]
    pivot = pivot.reset_index()

    # Delta metrics: CSRA improvement over Blockchain baseline
    for metric in ["final_accuracy", "mean_reward"]:
        col_b = f"{metric}_B"
        col_c = f"{metric}_C"
        col_a = f"{metric}_A"
        if col_b in pivot.columns and col_c in pivot.columns:
            pivot[f"delta_C_minus_B_{metric}"] = pivot[col_c] - pivot[col_b]
        if col_a in pivot.columns and col_c in pivot.columns:
            pivot[f"delta_C_minus_A_{metric}"] = pivot[col_c] - pivot[col_a]

    return pivot.round(4)


def generate_markdown_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    fairness: pd.DataFrame,
    report_path: Path,
    plot_dir: Path,
):
    """Write a self-contained Markdown research report."""
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    datasets  = sorted(df["dataset"].unique())
    scenarios = sorted(df["scenario"].unique())
    configs   = sorted(df["config"].unique())
    alphas    = sorted(df["alpha"].unique())

    lines = [
        "# FL + Blockchain Reward — Experiment Analysis Report",
        "",
        f"**Generated:** {now}  ",
        f"**Total records:** {len(df):,}  ",
        f"**Datasets:** {', '.join(datasets)}  ",
        f"**Scenarios:** {', '.join(f'{s} ({SCENARIO_LABELS.get(s, s)})' for s in scenarios)}  ",
        f"**Configs:** {', '.join(f'{c} — {CONFIG_LABELS.get(c, c)}' for c in configs)}  ",
        f"**Alpha values:** {', '.join(str(a) for a in alphas)}  ",
        "",
        "---",
        "",
        "## 1. Baseline Comparison (A vs B vs C)",
        "",
        "| Symbol | Config | Description |",
        "| --- | --- | --- |",
        "| **A** | Traditional FL | Standard FedAvg, no blockchain, no reward |",
        "| **B** | Blockchain Baseline | FedAvg + on-chain reputation + proportional reward |",
        "| **TrimmedMean** | Robust FL Baseline | Coordinate-wise trimmed mean aggregation, no blockchain |",
        "| **C** | CSRA Reward | B + CSRA-DCD anomaly detection before aggregation |",
        "| **C-CSRA** | CSRA-DCD Reward | Dedicated CSRA runner with MAD-based update-delta filtering |",
        "",
    ]

    # 1a. Full summary table per config
    if not summary.empty:
        valid = summary.dropna(subset=["final_accuracy"])
        show_cols = [c for c in ["dataset", "scenario", "config", "alpha",
                                  "dirichlet_alpha",
                                  "final_accuracy", "peak_accuracy", "convergence_round",
                                  "mean_reward", "reward_std", "mean_reputation",
                                  "reward_leakage", "false_positive_rate",
                                  "attack_detection_rate"] if c in summary.columns]
        lines += [
            "### Full Summary (all configs, all datasets)",
            "",
            _md_table(summary[show_cols].sort_values(["dataset", "scenario", "config", "alpha"])),
            "",
        ]

        # Best config per dataset/scenario
        if not valid.empty:
            idx  = valid.groupby(["dataset", "scenario"])["final_accuracy"].idxmax()
            best = summary.loc[idx, ["dataset", "scenario", "config", "alpha", "final_accuracy"]]
            lines += [
                "### Best Config per Dataset / Scenario",
                "",
                _md_table(best.round(4)),
                "",
            ]

    # 1b. B vs C detailed comparison
    comp = _build_config_comparison(summary)
    if not comp.empty:
        lines += [
            "### Config B vs C — Side-by-Side Comparison",
            "",
            "> `delta_C_minus_B` = improvement of CSRA (C) over Blockchain Baseline (B).  ",
            "> Positive = CSRA is better.",
            "",
        ]
        # Select most informative columns
        keep = [c for c in comp.columns if any(
            tok in c for tok in ["dataset", "scenario", "dirichlet_alpha", "alpha",
                                  "final_accuracy_A", "final_accuracy_B", "final_accuracy_C",
                                  "mean_reward_B", "mean_reward_C",
                                  "delta_C_minus_B_final_accuracy",
                                  "delta_C_minus_A_final_accuracy"]
        )]
        lines += [
            _md_table(comp[keep]),
            "",
        ]

    lines += ["### Plots", ""]
    for ds in datasets:
        lines.append(f"- `baseline_accuracy_curve_{ds}.png`")
        lines.append(f"- `baseline_final_accuracy_{ds}.png`")

    # --- Fairness ---
    lines += [
        "",
        "---",
        "",
        "## 2. Fairness Analysis",
        "",
        "| Metric | Formula | Interpretation |",
        "| --- | --- | --- |",
        "| Jain Index | (sum r)^2 / (n * sum r^2) | 1 = perfectly equal; closer to 0 = one client dominates |",
        "| Gini Coeff | area above Lorenz curve | 0 = equal; 1 = maximally unequal |",
        "| Fairness Gap | mean|r_i/R - q_i/Q| | 0 = reward proportional to quality |",
        "",
    ]

    if not fairness.empty:
        fair_agg = (
            fairness.groupby("config")[["jain", "gini", "fairness_gap", "reward_variance"]]
            .mean()
            .reset_index()
            .round(4)
        )
        fair_agg["config_label"] = fair_agg["config"].map(CONFIG_LABELS)
        lines += [
            "### Average Fairness Metrics by Config (across all rounds, datasets, scenarios)",
            "",
            _md_table(fair_agg[["config", "config_label", "jain", "gini", "fairness_gap", "reward_variance"]]),
            "",
            "> Jain Index closer to 1 = more fair.  ",
            "> Gini closer to 0 = more equal reward distribution.  ",
            "> Fairness Gap closer to 0 = reward proportional to contribution.  ",
            "",
        ]

        # Per-scenario breakdown
        fair_sc = (
            fairness.groupby(["scenario", "config"])[["jain", "gini", "fairness_gap"]]
            .mean()
            .reset_index()
            .round(4)
        )
        fair_sc["config_label"] = fair_sc["config"].map(CONFIG_LABELS)
        lines += [
            "### Fairness by Scenario",
            "",
            _md_table(fair_sc[["scenario", "config", "config_label", "jain", "gini", "fairness_gap"]]),
            "",
        ]

    lines += ["### Plots", ""]
    for ds in datasets:
        lines.append(f"- `fairness_boxplot_{ds}.png`")
        lines.append(f"- `fairness_histogram_{ds}.png`")
        lines.append(f"- `fairness_jain_gini_{ds}.png`")

    # --- Reputation ---
    lines += [
        "",
        "---",
        "",
        "## 3. Reputation Analysis",
        "",
        "Reputation is updated each round based on `quality_score` and `data_size`.",
        "In Config B, it gates which clients join `P_honest` (binary threshold).",
        "In Config C (CSRA), clients are **additionally** filtered by MAD-based update-delta anomaly detection before aggregation —",
        "malicious clients flagged by CSRA-DCD have their quality zeroed before on-chain submission.",
        "",
        "### Plots",
        "",
    ]
    for ds in datasets:
        for sc in scenarios:
            lines.append(f"- `reputation_avg_{ds}_{sc}.png`")
            lines.append(f"- `reputation_honest_vs_malicious_{ds}_{sc}.png`")

    # --- Attack ---
    lines += [
        "",
        "---",
        "",
        "## 4. Attack Analysis",
        "",
    ]

    has_attack_col = "has_attack" in df.columns
    if has_attack_col and df["has_attack"].any():
        attack_df = df[df["has_attack"]]
        n_attack  = attack_df["run_id"].nunique()

        # Reward leakage: % reward going to malicious clients per config
        if "client_type" in attack_df.columns:
            reward_by_type = (
                attack_df.groupby(["config", "client_type"])["reward_eth"]
                .sum()
                .reset_index()
            )
            leakage_rows = []
            for cfg in sorted(attack_df["config"].unique()):
                cfg_df  = reward_by_type[reward_by_type["config"] == cfg]
                total   = cfg_df["reward_eth"].sum()
                mal_r   = cfg_df[cfg_df["client_type"].isin(["free_rider", "lazy"])]["reward_eth"].sum()
                hon_r   = cfg_df[cfg_df["client_type"] == "honest"]["reward_eth"].sum()
                leakage_rows.append({
                    "config": cfg,
                    "config_label": CONFIG_LABELS.get(cfg, cfg),
                    "total_reward": total,
                    "honest_reward": hon_r,
                    "malicious_reward": mal_r,
                    "leakage_%": round(100 * mal_r / total, 2) if total > 0 else 0.0,
                })
            leak_df = pd.DataFrame(leakage_rows)
            lines += [
                f"**Attack runs detected:** {n_attack}  ",
                "",
                "### Reward Leakage by Config",
                "",
                "> `leakage_%` = % of total reward captured by malicious clients.  ",
                "> CSRA (C) should show lower leakage than Blockchain Baseline (B).",
                "",
                _md_table(leak_df),
                "",
            ]

        lines += ["### Plots", ""]
        for ds in datasets:
            lines.append(f"- `attack_accuracy_{ds}.png`")
            lines.append(f"- `attack_reward_share_{ds}.png`")
    else:
        lines += ["*No attack runs detected in this dataset.*", ""]

    # --- Alpha sensitivity ---
    lines += [
        "",
        "---",
        "",
        "## 5. Alpha Sensitivity",
        "",
        "Alpha (alpha) controls the weight of quality vs data-size in the reward formula:",
        "",
        "```",
        "W_new = alpha * quality_norm + (1 - alpha) * data_size_norm",
        "```",
        "",
        "| alpha | Interpretation |",
        "| --- | --- |",
        "| 0.0 | Pure data-quantity reward (ignores quality) |",
        "| 0.5 | Balanced quality + quantity |",
        "| 1.0 | Pure quality-based reward |",
        "",
    ]

    if not summary.empty:
        alpha_tbl = (
            summary.groupby(["config", "alpha"])[["final_accuracy", "mean_reward", "reward_std"]]
            .mean()
            .reset_index()
            .round(4)
        )
        alpha_tbl["config_label"] = alpha_tbl["config"].map(CONFIG_LABELS)
        lines += [
            "### Accuracy & Reward by Alpha (averaged over datasets/scenarios)",
            "",
            _md_table(alpha_tbl[["config", "config_label", "alpha", "final_accuracy", "mean_reward", "reward_std"]]),
            "",
        ]

    lines += ["### Plots", ""]
    for ds in datasets:
        lines.append(f"- `alpha_sensitivity_{ds}.png`")

    # --- Summary tables ---
    lines += [
        "",
        "---",
        "",
        "## 6. Output Files",
        "",
        "| File | Description |",
        "| --- | --- |",
        "| `summary_metrics.csv` | Per-(dataset, scenario, config, alpha): accuracy, stable convergence round, reward stats, reputation, FPR, reward leakage |",
        "| `fairness_metrics.csv` | Per-(dataset, scenario, config, alpha, round): Jain, Gini, fairness_gap, reward_quality_corr |",
        "| `analysis_report.md` | This report |",
        "",
        "---",
        "",
        "*Report auto-generated by `analyze_results.py`.*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved: %s", report_path)
