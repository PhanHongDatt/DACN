"""
report.py — Auto-generate a Markdown analysis report with full B-vs-C comparison,
statistical significance tests, and LaTeX table export.
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONFIG_LABELS = {
    "A": "Traditional FL",
    "B": "Blockchain Baseline",
    "C": "CSRA Reward",
    "C-CSRA": "CSRA-DCD Reward",
    "C-CSRA-Opt": "CSRA-DCD Reward (Optimized)",
    "TrimmedMean": "TrimmedMean Robust FL",
}
SCENARIO_LABELS = {"K1": "IID", "K2": "Weak Non-IID", "K3": "Dirichlet Non-IID"}
CONFIG_ORDER = ["A", "B", "TrimmedMean", "C", "C-CSRA", "C-CSRA-Opt"]


def _ordered_configs(configs) -> list:
    present = list(dict.fromkeys(configs))
    known = [cfg for cfg in CONFIG_ORDER if cfg in present]
    extra = sorted(cfg for cfg in present if cfg not in CONFIG_ORDER)
    return known + extra


def _csra_config(configs) -> str | None:
    for cfg in ["C-CSRA-Opt", "C-CSRA", "C"]:
        if cfg in set(configs):
            return cfg
    return None


def _scenario_display_values(df: pd.DataFrame) -> list[str]:
    col = "scenario_variant" if "scenario_variant" in df.columns else "scenario"
    order = {"K1": 1, "K2": 2, "K3": 3}
    cols = [c for c in ["scenario", col, "dirichlet_alpha"] if c in df.columns]
    variants = df[cols].drop_duplicates()
    if "scenario" in variants.columns:
        variants["_order"] = variants["scenario"].map(order).fillna(99)
    else:
        variants["_order"] = 99
    if "dirichlet_alpha" not in variants.columns:
        variants["dirichlet_alpha"] = 0.0
    return variants.sort_values(["_order", "dirichlet_alpha", col])[col].astype(str).tolist()


def _safe_name(value) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "p")
    )


def _append_plot(lines: list[str], plot_dir: Path, filename: str) -> None:
    """Append plot reference only when the PNG was actually generated."""
    if (plot_dir / filename).exists():
        lines.append(f"- `{filename}`")


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


def _df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "", float_fmt: str = ".4f") -> str:
    """Convert DataFrame to LaTeX table string."""
    cols = df.columns.tolist()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{'l' * len(cols)}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            elif isinstance(v, bool):
                cells.append("\\ding{51}" if v else "\\ding{55}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _build_config_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparison table per dataset/scenario/condition.

    Alpha is intentionally treated as a method parameter, not an index column:
    A, B, and C-CSRA use different fixed alpha values in this project, so
    indexing by alpha would prevent side-by-side baseline comparison.
    """
    if summary.empty:
        return pd.DataFrame()

    index_cols = [
        c for c in ["dataset", "scenario", "scenario_variant", "dirichlet_alpha", "attack_label"]
        if c in summary.columns
    ]
    value_cols = [c for c in ["final_accuracy", "mean_reward", "reward_std", "mean_reputation",
                               "reward_leakage", "jain", "gini", "fairness_gap", "alpha"] if c in summary.columns]

    pivot = summary.pivot_table(
        index=index_cols,
        columns="config",
        values=value_cols,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{cfg}" for metric, cfg in pivot.columns]
    pivot = pivot.reset_index()

    csra_cfg = _csra_config(summary["config"].unique())
    if csra_cfg is None:
        return pivot.round(4)

    # Delta metrics: CSRA improvement over Blockchain baseline and FedAvg.
    for metric in ["final_accuracy", "mean_reward"]:
        col_b = f"{metric}_B"
        col_c = f"{metric}_{csra_cfg}"
        col_a = f"{metric}_A"
        if col_b in pivot.columns and col_c in pivot.columns:
            pivot[f"delta_CSRA_minus_B_{metric}"] = pivot[col_c] - pivot[col_b]
        if col_a in pivot.columns and col_c in pivot.columns:
            pivot[f"delta_CSRA_minus_A_{metric}"] = pivot[col_c] - pivot[col_a]

    return pivot.round(4)


def generate_markdown_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    fairness: pd.DataFrame,
    report_path: Path,
    plot_dir: Path,
    stat_tests: pd.DataFrame = None,
):
    """Write a self-contained Markdown research report."""
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    datasets  = sorted(df["dataset"].unique())
    scenarios = sorted(df["scenario"].unique())
    scenario_variants = _scenario_display_values(df)
    configs   = _ordered_configs(df["config"].unique())
    alphas    = sorted(df["alpha"].unique())
    attack_counts = df.groupby("attack_label")["run_id"].nunique().to_dict() if "attack_label" in df.columns else {"clean": df["run_id"].nunique()}

    lines = [
        "# FL + Blockchain Reward — Experiment Analysis Report",
        "",
        f"**Generated:** {now}  ",
        f"**Total records:** {len(df):,}  ",
        f"**Datasets:** {', '.join(datasets)}  ",
        f"**Scenarios:** {', '.join(f'{s} ({SCENARIO_LABELS.get(s, s)})' for s in scenarios)}  ",
        f"**Scenario variants:** {', '.join(scenario_variants)}  ",
        f"**Configs:** {', '.join(f'{c} — {CONFIG_LABELS.get(c, c)}' for c in configs)}  ",
        f"**Alpha values:** {', '.join(str(a) for a in alphas)}  ",
        f"**Run conditions:** {', '.join(f'{k}={v}' for k, v in attack_counts.items())}  ",
        "",
        "---",
        "",
        "## 1. Baseline Comparison (A vs B vs CSRA)",
        "",
        "| Symbol | Config | Description |",
        "| --- | --- | --- |",
        "| **A** | Traditional FL | Standard FedAvg, no blockchain, no reward |",
        "| **B** | Blockchain Baseline | FedAvg + on-chain reputation + proportional reward |",
        "| **TrimmedMean** | Robust FL Baseline | Coordinate-wise trimmed mean aggregation, no blockchain |",
        "| **C** | CSRA Reward | Legacy CSRA naming, if present in older logs |",
        "| **C-CSRA** | CSRA-DCD Reward | Dedicated CSRA runner with MAD-based update-delta filtering |",
        "| **C-CSRA-Opt** | CSRA-DCD Reward (Optimized) | Runtime config name used by the CSRA strategy |",
        "",
    ]

    # 1a. Full summary table per config
    if not summary.empty:
        valid = summary.dropna(subset=["final_accuracy"])
        show_cols = [c for c in [
            "dataset", "scenario", "scenario_variant", "attack_label",
            "config", "alpha", "dirichlet_alpha", "n_runs",
            "rounds_observed", "max_round",
            "final_accuracy", "peak_accuracy", "convergence_round",
            "mean_reward", "reward_std", "mean_reputation",
            "reward_leakage", "reward_ratio", "eii",
            "false_positive_rate", "attack_detection_rate", "fdr",
            "false_positive_count", "attack_detected_count", "malicious_rows",
            "jain", "gini", "fairness_gap",
        ] if c in summary.columns]
        sort_cols = [c for c in ["dataset", "scenario", "dirichlet_alpha", "attack_label", "config", "alpha"] if c in summary.columns]
        lines += [
            "### Full Summary (all configs, all datasets)",
            "",
            _md_table(summary[show_cols].sort_values(sort_cols)),
            "",
        ]

        # Best config per dataset/scenario
        if not valid.empty:
            best_group = [c for c in ["dataset", "scenario_variant", "attack_label"] if c in valid.columns]
            if not best_group:
                best_group = ["dataset", "scenario"]
            idx  = valid.groupby(best_group)["final_accuracy"].idxmax()
            best_cols = [c for c in best_group + ["config", "alpha", "final_accuracy"] if c in summary.columns]
            best = summary.loc[idx, best_cols]
            lines += [
                "### Best Config per Dataset / Scenario Variant",
                "",
                _md_table(best.round(4)),
                "",
            ]

    # 1b. B vs C detailed comparison
    comp = _build_config_comparison(summary)
    if not comp.empty:
        csra_cfg = _csra_config(summary["config"].unique()) or "CSRA"
        lines += [
            f"### Config B vs {csra_cfg} — Side-by-Side Comparison",
            "",
            "> `delta_CSRA_minus_B` = improvement of CSRA over Blockchain Baseline (B).  ",
            "> Positive = CSRA is better.",
            "",
        ]
        keep = [c for c in comp.columns if any(
            tok in c for tok in ["dataset", "scenario", "scenario_variant", "attack_label",
                                  "dirichlet_alpha",
                                  "alpha_A", "alpha_B", f"alpha_{csra_cfg}",
                                  "final_accuracy_A", "final_accuracy_B", f"final_accuracy_{csra_cfg}",
                                  "mean_reward_B", f"mean_reward_{csra_cfg}",
                                  "jain_B", f"jain_{csra_cfg}", "gini_B", f"gini_{csra_cfg}",
                                  "fairness_gap_B", f"fairness_gap_{csra_cfg}",
                                  "delta_CSRA_minus_B", "delta_CSRA_minus_A"]
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
        "| Reward Ratio | mean(honest) / mean(malicious) | Higher = better discrimination |",
        "| EII | (r_honest - r_lazy) / delta_cost | >1 = honest behavior economically worthwhile |",
        "",
    ]

    if not fairness.empty:
        fair_group_cols = [c for c in ["attack_label", "config"] if c in fairness.columns]
        fair_agg = (
            fairness.groupby(fair_group_cols)[["jain", "gini", "fairness_gap", "reward_variance"]]
            .mean()
            .reset_index()
            .round(4)
        )
        fair_agg["config_label"] = fair_agg["config"].map(CONFIG_LABELS)
        fair_show_cols = [c for c in ["attack_label", "config", "config_label", "jain", "gini", "fairness_gap", "reward_variance"] if c in fair_agg.columns]
        lines += [
            "### Average Fairness Metrics by Config",
            "",
            _md_table(fair_agg[fair_show_cols]),
            "",
            "> Jain Index closer to 1 = more fair.  ",
            "> Gini closer to 0 = more equal reward distribution.  ",
            "> Fairness Gap closer to 0 = reward proportional to contribution.  ",
            "",
        ]

        # Per-scenario breakdown
        fair_scenario_col = "scenario_variant" if "scenario_variant" in fairness.columns else "scenario"
        fair_sc_group = [c for c in [fair_scenario_col, "attack_label", "config"] if c in fairness.columns]
        fair_sc = (
            fairness.groupby(fair_sc_group)[["jain", "gini", "fairness_gap"]]
            .mean()
            .reset_index()
            .round(4)
        )
        fair_sc["config_label"] = fair_sc["config"].map(CONFIG_LABELS)
        fair_sc_show = [c for c in [fair_scenario_col, "attack_label", "config", "config_label", "jain", "gini", "fairness_gap"] if c in fair_sc.columns]
        lines += [
            "### Fairness by Scenario",
            "",
            _md_table(fair_sc[fair_sc_show]),
            "",
        ]

    lines += ["### Plots", ""]
    for ds in datasets:
        lines.append(f"- `fairness_boxplot_{ds}.png`")
        lines.append(f"- `fairness_histogram_{ds}.png`")
        lines.append(f"- `fairness_jain_gini_{ds}.png`")
        lines.append(f"- `fairness_reward_vs_quality_{ds}.png`")

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
        for sc in _scenario_display_values(df[df["dataset"] == ds]):
            safe_sc = _safe_name(sc)
            _append_plot(lines, plot_dir, f"reputation_avg_{ds}_{safe_sc}.png")
            _append_plot(lines, plot_dir, f"reputation_honest_vs_malicious_{ds}_{safe_sc}.png")

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
            attack_scenario_col = "scenario_variant" if "scenario_variant" in attack_df.columns else "scenario"
            attack_df = attack_df.copy()
            attack_df["is_malicious_tmp"] = attack_df.get(
                "is_malicious",
                attack_df["client_type"].isin(["free_rider", "lazy", "label_noise", "malicious"]),
            )
            reward_by_type = (
                attack_df.groupby([attack_scenario_col, "config", "client_type"])["reward_eth"]
                .sum()
                .reset_index()
            )
            leakage_rows = []
            for (scenario_value, cfg), raw_grp in attack_df.groupby([attack_scenario_col, "config"]):
                cfg_df  = reward_by_type[
                    (reward_by_type[attack_scenario_col] == scenario_value)
                    & (reward_by_type["config"] == cfg)
                ]
                total   = cfg_df["reward_eth"].sum()
                mal_r   = raw_grp.loc[raw_grp["is_malicious_tmp"], "reward_eth"].sum()
                hon_r   = cfg_df[cfg_df["client_type"] == "honest"]["reward_eth"].sum()
                leakage_rows.append({
                    "scenario": scenario_value,
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
                "### Reward Leakage by Scenario / Config",
                "",
                "> `leakage_%` = % of total reward captured by malicious clients.  ",
                "> CSRA should show lower leakage than Blockchain Baseline (B).",
                "",
                _md_table(leak_df),
                "",
            ]

        lines += ["### Plots", ""]
        for ds in datasets:
            _append_plot(lines, plot_dir, f"attack_accuracy_{ds}.png")
            _append_plot(lines, plot_dir, f"attack_reward_share_{ds}.png")
    else:
        lines += ["*No attack runs detected in the loaded logs.*", ""]

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
        alpha_src = summary
        if "attack_label" in alpha_src.columns and (alpha_src["attack_label"] == "clean").any():
            alpha_src = alpha_src[alpha_src["attack_label"] == "clean"]
        alpha_tbl = (
            alpha_src.groupby(["config", "alpha"])[["final_accuracy", "mean_reward", "reward_std"]]
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

    # --- Convergence ---
    lines += [
        "",
        "---",
        "",
        "## 6. Convergence Analysis",
        "",
        "Convergence round = first round reaching 95% of peak accuracy, sustained for 5 rounds.",
        "",
        "### Plots",
        "",
    ]
    for ds in datasets:
        lines.append(f"- `convergence_round_{ds}.png`")
        lines.append(f"- `convergence_scatter_{ds}.png`")

    # --- Statistical significance ---
    lines += [
        "",
        "---",
        "",
        "## 7. Statistical Significance Tests",
        "",
        "Mann-Whitney U test (unpaired, two-sided) between config pairs.",
        "Effect size: rank-biserial correlation. α = 0.05.",
        "",
    ]

    if stat_tests is not None and not stat_tests.empty:
        show_cols = [c for c in ["config_a", "config_b", "n_a", "n_b", "mean_a", "mean_b",
                                  "diff", "p_value", "effect_size", "significant"] if c in stat_tests.columns]
        lines += [
            "### Pairwise Config Comparison (Final Accuracy)",
            "",
            _md_table(stat_tests[show_cols].round(6)),
            "",
            "> **significant** = p < 0.05.  ",
            "> Positive **diff** = config_b has higher accuracy than config_a.  ",
            "> **effect_size** close to ±1 = large effect, close to 0 = negligible.",
            "",
        ]
    else:
        lines += ["*Insufficient runs for statistical tests (need ≥2 per config).*", ""]

    # --- Output files ---
    lines += [
        "",
        "---",
        "",
        "## 8. Output Files",
        "",
        "| File | Description |",
        "| --- | --- |",
        "| `summary_metrics.csv` | Per-(dataset, scenario variant, config, alpha, clean/attack): accuracy, convergence, reward, reputation, FPR, leakage, RR, EII, fairness |",
        "| `fairness_metrics.csv` | Per-(dataset, scenario variant, config, alpha, clean/attack, round): Jain, Gini, fairness_gap, reward_quality_corr |",
        "| `stat_tests.csv` | Pairwise Mann-Whitney U test results with p-values and effect sizes |",
        "| `analysis_report.md` | This report |",
        "",
        "---",
        "",
        "*Report auto-generated by `analyze_results.py`.*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved: %s", report_path)


def export_latex_tables(summary: pd.DataFrame, fairness: pd.DataFrame, out_dir: Path):
    """Export key tables as standalone LaTeX files for paper inclusion."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        # Summary table
        cols = [c for c in ["dataset", "scenario", "scenario_variant", "attack_label",
                             "config", "alpha", "dirichlet_alpha",
                             "final_accuracy", "convergence_round", "mean_reward",
                             "reward_leakage", "jain", "gini"] if c in summary.columns]
        sort_cols = [c for c in ["dataset", "scenario", "dirichlet_alpha", "attack_label", "config", "alpha"] if c in summary.columns]
        tbl = summary[cols].sort_values(sort_cols).round(4)
        latex = _df_to_latex(tbl, caption="Summary metrics across all experiment configurations",
                             label="tab:summary")
        (out_dir / "table_summary.tex").write_text(latex, encoding="utf-8")
        log.info("LaTeX table saved: %s/table_summary.tex", out_dir)

    if not fairness.empty:
        # Fairness per scenario
        fair_scenario_col = "scenario_variant" if "scenario_variant" in fairness.columns else "scenario"
        fair_group = [c for c in [fair_scenario_col, "attack_label", "config"] if c in fairness.columns]
        fair_agg = (
            fairness.groupby(fair_group)[["jain", "gini", "fairness_gap"]]
            .mean().reset_index().round(4)
        )
        latex = _df_to_latex(fair_agg, caption="Fairness metrics by scenario and configuration",
                             label="tab:fairness")
        (out_dir / "table_fairness.tex").write_text(latex, encoding="utf-8")
        log.info("LaTeX table saved: %s/table_fairness.tex", out_dir)
