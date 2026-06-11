"""Markdown and LaTeX report generation for schema v2 analysis.

Schema v2 compares FL methods as:
    aggregation_method x reward_policy

Blockchain is treated as the audit/distribution layer, not as an experimental
baseline. This report focuses on M1-M6 method comparisons.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.style import METHOD_LABELS, METHOD_ORDER

log = logging.getLogger(__name__)

DETECTOR_PARAM_COLS = [
    "mad_threshold", "cosine_threshold", "direction_min_norm_z",
    "min_honest_ratio", "fallback_hard_z",
    "suspicion_decay", "suspicion_threshold", "low_quality_z_threshold",
    "low_quality_suspicion", "zero_data_suspicion", "anomaly_suspicion",
    "authenticity_suspicion", "low_authenticity_threshold",
    "high_update_norm_z_threshold", "inefficient_update_suspicion",
]
_TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", ""}


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _ordered_methods(methods) -> list[str]:
    present = list(dict.fromkeys(str(m) for m in methods))
    known = [m for m in METHOD_ORDER if m in present]
    extra = sorted(m for m in present if m not in METHOD_ORDER)
    return known + extra


def _scenario_values(df: pd.DataFrame) -> list[str]:
    col = "scenario_variant" if "scenario_variant" in df.columns else "scenario"
    if col not in df.columns:
        return []
    order = {"K1": 1, "K2": 2, "K3": 3}
    cols = [c for c in ["scenario", col, "dirichlet_alpha"] if c in df.columns]
    variants = df[cols].drop_duplicates()
    variants["_order"] = variants["scenario"].map(order).fillna(99) if "scenario" in variants else 99
    if "dirichlet_alpha" not in variants:
        variants["dirichlet_alpha"] = 0.0
    variants = variants.sort_values(["_order", "dirichlet_alpha", col])
    return variants[col].astype(str).tolist()


def _safe_name(value) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "p")
    )


def _nullable_bool_series(series: pd.Series) -> pd.Series:
    """Parse bool-like Series while preserving missing/unparseable values."""
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_present = numeric.notna()
    result.loc[numeric_present] = numeric.loc[numeric_present].astype(float) != 0.0
    text = series.astype("string").str.strip().str.lower()
    result.loc[text.isin(_TRUE_STRINGS)] = True
    result.loc[text.isin(_FALSE_STRINGS)] = False
    return result.astype("boolean")


def _malicious_mask(df: pd.DataFrame) -> pd.Series:
    """Ground-truth malicious mask with fallback for direct report tests."""
    if "client_type" in df.columns:
        type_fallback = df["client_type"].astype(str).isin([
            "free_rider", "stealth_free_rider", "lazy", "label_noise",
            "sign_flip", "malicious",
        ])
    else:
        type_fallback = pd.Series(False, index=df.index, dtype=bool)

    if "is_malicious" in df.columns:
        is_malicious = _nullable_bool_series(df["is_malicious"])
        fallback = is_malicious.fillna(type_fallback).astype(bool)
    else:
        fallback = type_fallback

    if "ground_truth_honest" in df.columns:
        truth = _nullable_bool_series(df["ground_truth_honest"])
        truth_present = truth.notna()
        truth_malicious = ~truth.fillna(True).astype(bool)
        return fallback.where(~truth_present, truth_malicious)
    return fallback


def _participating_mask(df: pd.DataFrame) -> pd.Series:
    if "detection_reason" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    return df["detection_reason"].astype(str).ne("not_participating")


def _bool_mask(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """Return a boolean Series aligned to df.index, parsing bool-like strings."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return _nullable_bool_series(df[col]).fillna(default).astype(bool)


def _md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _df_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_fmt: str = ".4f",
) -> str:
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
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _last_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    scenario_col = "scenario_variant" if "scenario_variant" in df.columns else "scenario"
    detector_cols = [c for c in DETECTOR_PARAM_COLS if c in df.columns]

    def _last_valid(g: pd.DataFrame) -> float:
        s = g.groupby("round_num")["global_accuracy"].mean().sort_index().dropna()
        return float(s.iloc[-1]) if not s.empty else np.nan

    out = (
        df.groupby(["dataset", scenario_col, "method", *detector_cols], dropna=False)
        .apply(_last_valid)
        .reset_index(name="final_accuracy")
    )
    out["method_label"] = out["method"].map(_method_label)
    return out.sort_values(["dataset", scenario_col, "method"])


def _attack_table(df: pd.DataFrame) -> pd.DataFrame:
    if "has_attack" not in df.columns:
        return pd.DataFrame()
    attack_df = df[df["has_attack"]].copy()
    if attack_df.empty:
        return pd.DataFrame()

    scenario_col = "scenario_variant" if "scenario_variant" in attack_df.columns else "scenario"
    detector_cols = [c for c in DETECTOR_PARAM_COLS if c in attack_df.columns]
    group_cols = ["dataset", scenario_col, "attack_type", "method", *detector_cols]
    records = []
    for keys, grp in attack_df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        dataset = key_map["dataset"]
        scenario = key_map[scenario_col]
        attack_type = key_map["attack_type"]
        method = key_map["method"]
        acc = grp.groupby("round_num")["global_accuracy"].mean().sort_index().dropna()
        reward = pd.to_numeric(
            grp.get("reward_eth", pd.Series(0.0, index=grp.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        malicious = _malicious_mask(grp)
        anomaly = _bool_mask(grp, "is_anomaly")
        data_commitment = _bool_mask(grp, "data_commitment_anomaly")
        inefficient_update = _bool_mask(grp, "inefficient_update")
        suspicion_quarantine = _bool_mask(grp, "suspicion_quarantine")
        participating = _participating_mask(grp)
        if "reward_blocked" in grp.columns:
            reward_blocked = _bool_mask(grp, "reward_blocked")
        elif "reward_eligible" in grp.columns:
            reward_blocked = (~_bool_mask(grp, "reward_eligible")) & participating
        elif "is_honest" in grp.columns:
            reward_blocked = (~_bool_mask(grp, "is_honest")) & participating
        else:
            reward_source = grp.get(
                "reward_eth", pd.Series(0.0, index=grp.index, dtype=float)
            )
            reward = pd.to_numeric(reward_source, errors="coerce").fillna(0.0)
            reward_blocked = reward.le(1e-12) & participating
        mal_part = malicious & participating
        honest_part = (~malicious) & participating
        total_reward = float(reward.loc[participating].sum())
        rec = {
            "dataset": dataset,
            "scenario": scenario,
            "attack_type": attack_type,
            "method": method,
            "method_label": _method_label(method),
            "final_accuracy": float(acc.iloc[-1]) if not acc.empty else np.nan,
            "reward_leakage": (
                float(reward.loc[mal_part].sum() / total_reward)
                if total_reward > 1e-12 else 0.0
            ),
            "attack_detection_rate": (
                float(anomaly[mal_part].mean()) if mal_part.any() else np.nan
            ),
            "attack_reward_block_rate": (
                float(reward_blocked[mal_part].mean()) if mal_part.any() else np.nan
            ),
            "false_positive_detection_rate": (
                float(anomaly[honest_part].mean()) if honest_part.any() else 0.0
            ),
            "false_positive_quarantine_rate": (
                float(reward_blocked[honest_part].mean()) if honest_part.any() else 0.0
            ),
            "false_positive_rate": (
                float(anomaly[honest_part].mean()) if honest_part.any() else 0.0
            ),
            "data_commitment_anomaly_rate": (
                float(data_commitment[participating].mean())
                if participating.any() else np.nan
            ),
            "inefficient_update_rate": (
                float(inefficient_update[participating].mean())
                if participating.any() else np.nan
            ),
            "suspicion_quarantine_rate": (
                float(suspicion_quarantine[participating].mean())
                if participating.any() else np.nan
            ),
        }
        for col in detector_cols:
            rec[col] = key_map[col]
        records.append(rec)
    return pd.DataFrame(records).sort_values(
        ["dataset", "scenario", "attack_type", "method", *detector_cols]
    )


def _fairness_table(fairness: pd.DataFrame) -> pd.DataFrame:
    if fairness.empty:
        return pd.DataFrame()
    scenario_col = "scenario_variant" if "scenario_variant" in fairness.columns else "scenario"
    detector_cols = [c for c in DETECTOR_PARAM_COLS if c in fairness.columns]
    group = [c for c in ["dataset", scenario_col, "method", *detector_cols] if c in fairness.columns]
    if "method" not in fairness.columns and "config" in fairness.columns:
        fairness = fairness.copy()
        fairness["method"] = fairness["config"]
        group = [c for c in ["dataset", scenario_col, "method", *detector_cols] if c in fairness.columns]
    metric_cols = [
        c for c in [
            "jain", "gini", "fairness_gap", "fairness_gap_alignment",
            "reward_quality_corr", "reward_alignment_corr", "reward_variance",
        ]
        if c in fairness.columns
    ]
    out = fairness.groupby(group, dropna=False)[metric_cols].mean().reset_index()
    out["method_label"] = out["method"].map(_method_label)
    return out.sort_values(group)


def _reward_risk_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return reward-risk diagnostics that should be visible in reports."""
    required = {
        "reward_eligible_honest_rows",
        "honest_reward_starvation_count",
        "honest_reward_starvation_rate",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()

    scenario_col = "scenario_variant" if "scenario_variant" in summary.columns else "scenario"
    detector_cols = [c for c in DETECTOR_PARAM_COLS if c in summary.columns]
    cols = [
        c for c in [
            "dataset", scenario_col, "attack_type", "attack_label",
            "method", "method_label", *detector_cols,
            "n_runs", "reward_eligible_honest_rows",
            "honest_reward_starvation_count",
            "honest_reward_starvation_rate",
            "reward_leakage", "false_positive_quarantine_rate",
        ] if c in summary.columns
    ]
    out = summary[cols].copy()
    if "method_label" not in out.columns and "method" in out.columns:
        out["method_label"] = out["method"].map(_method_label)
    sort_cols = [
        c for c in [
            "dataset", scenario_col, "attack_type", "method",
            *detector_cols,
        ] if c in out.columns
    ]
    return out.sort_values(sort_cols).reset_index(drop=True)


def generate_markdown_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    fairness: pd.DataFrame,
    report_path: Path,
    plot_dir: Path,
    stat_tests: pd.DataFrame | None = None,
    paired_stat_tests: pd.DataFrame | None = None,
):
    """Write schema-v2 analysis report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    methods = _ordered_methods(df["method"].unique() if "method" in df.columns else df["config"].unique())
    datasets = sorted(df["dataset"].unique())
    scenarios = _scenario_values(df)
    attack_counts = (
        df.groupby("attack_type")["run_id"].nunique().to_dict()
        if "attack_type" in df.columns else {"clean": df["run_id"].nunique()}
    )

    clean_df = df[df.get("attack_type", "clean") == "clean"] if "attack_type" in df.columns else df
    acc_tbl = _last_accuracy_table(clean_df).round(4)
    fair_tbl = _fairness_table(fairness).round(4)
    atk_tbl = _attack_table(df).round(4)
    risk_tbl = _reward_risk_table(summary).round(4)

    lines = [
        "# FL Reward - Schema v2 Experiment Report",
        "",
        f"**Generated:** {now}  ",
        f"**Rows:** {len(df):,}  ",
        f"**Runs:** {df['run_id'].nunique():,}  ",
        f"**Datasets:** {', '.join(datasets)}  ",
        f"**Scenarios:** {', '.join(scenarios)}  ",
        f"**Conditions:** {', '.join(f'{k}={v}' for k, v in attack_counts.items())}  ",
        "",
        "## Method Matrix",
        "",
        "Blockchain is treated as an audit and reward-distribution layer. It is not used as a separate experimental baseline.",
        "",
        "| ID | Method | Role |",
        "| --- | --- | --- |",
        "| M1 | FedAvg + EqualSplit | Minimal baseline |",
        "| M2 | FedAvg + DataSize | Data-quantity reward baseline |",
        "| M3 | FedAvg + QualityOnly | One-round quality reward baseline |",
        "| M4 | FedAvg + CSRAReward | Reward-formula ablation |",
        "| M5 | CSRA-DCD + EqualSplit | Filtering-only ablation |",
        "| M6 | CSRA-DCD + CSRAReward | Proposed full system |",
        "",
        "## Clean Accuracy",
        "",
        _md_table(acc_tbl[[
            c for c in [
                "dataset", "scenario_variant", "method", "method_label",
                *DETECTOR_PARAM_COLS, "final_accuracy",
            ]
            if c in acc_tbl.columns
        ]]),
        "",
        "## Reward Fairness",
        "",
        _md_table(fair_tbl[[
            c for c in [
                "dataset", "scenario_variant", "method", "method_label",
                "jain", "gini", "fairness_gap", "fairness_gap_alignment",
                "reward_quality_corr", "reward_alignment_corr", "reward_variance",
            ]
            if c in fair_tbl.columns
        ]]),
        "",
        "## Reward Risk Diagnostics",
        "",
        "Honest reward starvation counts honest participating clients that were not reward-blocked but still received approximately zero reward.",
        "",
        _md_table(risk_tbl[[
            c for c in [
                "dataset", "scenario_variant", "attack_type", "method",
                "method_label", *DETECTOR_PARAM_COLS, "n_runs",
                "reward_eligible_honest_rows",
                "honest_reward_starvation_count",
                "honest_reward_starvation_rate",
                "reward_leakage", "false_positive_quarantine_rate",
            ]
            if c in risk_tbl.columns
        ]]),
        "",
        "## Attack Robustness",
        "",
    ]

    if atk_tbl.empty:
        lines += ["_No attack runs found._", ""]
    else:
        show_cols = [
            c for c in [
                "dataset", "scenario", "attack_type", "method", "method_label",
                *DETECTOR_PARAM_COLS,
                "final_accuracy", "reward_leakage",
                "attack_detection_rate", "attack_reward_block_rate",
                "false_positive_detection_rate", "false_positive_quarantine_rate",
                "data_commitment_anomaly_rate", "inefficient_update_rate",
                "suspicion_quarantine_rate",
            ] if c in atk_tbl.columns
        ]
        lines += [_md_table(atk_tbl[show_cols]), ""]

    lines += [
        "## Plots",
        "",
    ]
    for ds in datasets:
        for filename in [
            f"baseline_accuracy_curve_{ds}.png",
            f"baseline_final_accuracy_{ds}.png",
            f"fairness_boxplot_{ds}.png",
            f"fairness_jain_gini_{ds}.png",
            f"fairness_reward_vs_quality_{ds}.png",
            f"attack_accuracy_{ds}.png",
            f"attack_reward_share_{ds}.png",
            f"convergence_round_{ds}.png",
            f"convergence_scatter_{ds}.png",
            f"beta_sensitivity_{ds}.png",
        ]:
            if (plot_dir / filename).exists():
                lines.append(f"- `{filename}`")

    lines += [
        "",
        "## Statistical Tests",
        "",
        "### Mann-Whitney U (Unpaired)",
        "",
    ]
    if stat_tests is not None and not stat_tests.empty:
        show_cols = [
            c for c in [
                "config_a", "config_b", "n_a", "n_b", "mean_a", "mean_b",
                "diff", "p_value", "effect_size", "significant",
            ] if c in stat_tests.columns
        ]
        lines += [_md_table(stat_tests[show_cols].round(6)), ""]
    else:
        lines += ["_Insufficient runs for pairwise tests._", ""]

    lines += [
        "### Wilcoxon Signed-Rank (Paired by Seed)",
        "",
    ]
    if paired_stat_tests is not None and not paired_stat_tests.empty:
        show_cols = [
            c for c in [
                "config_a", "config_b", "n_pairs", "paired_on",
                "skipped_unpaired", "skipped_ambiguous",
                "mean_a", "mean_b", "mean_diff",
                "p_value", "statistic", "significant",
            ] if c in paired_stat_tests.columns
        ]
        lines += [_md_table(paired_stat_tests[show_cols].round(6)), ""]
    else:
        lines += ["_Insufficient same-seed pairs for Wilcoxon tests._", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved: %s", report_path)


def export_latex_tables(summary: pd.DataFrame, fairness: pd.DataFrame, out_dir: Path):
    """Export compact schema-v2 tables for paper/slides."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        cols = [
            c for c in [
                "dataset", "scenario_variant", "method", "method_label",
                "attack_type", "attack_label", "beta", "gamma", "delta",
                *DETECTOR_PARAM_COLS,
                "final_accuracy", "peak_accuracy", "convergence_round",
                "reward_leakage",
                "reward_eligible_honest_rows",
                "honest_reward_starvation_count",
                "honest_reward_starvation_rate",
                "attack_detection_rate", "attack_reward_block_rate",
                "false_positive_detection_rate", "false_positive_quarantine_rate",
                "data_commitment_anomaly_rate", "low_quality_outlier_rate",
                "inefficient_update_rate",
                "suspicion_quarantine_rate",
                "jain", "gini", "fairness_gap", "fairness_gap_alignment",
                "reward_quality_corr", "reward_alignment_corr",
            ] if c in summary.columns
        ]
        sort_cols = [
            c for c in [
                "dataset", "scenario_variant", "attack_type", "method", "beta",
                *DETECTOR_PARAM_COLS,
            ]
            if c in summary.columns
        ]
        tbl = summary[cols].sort_values(sort_cols).round(4)
        latex = _df_to_latex(
            tbl,
            caption="Schema v2 summary metrics by FL method",
            label="tab:summary_schema_v2",
        )
        (out_dir / "table_summary.tex").write_text(latex, encoding="utf-8")
        log.info("LaTeX table saved: %s/table_summary.tex", out_dir)

    if not fairness.empty:
        fair = fairness.copy()
        if "method" not in fair.columns and "config" in fair.columns:
            fair["method"] = fair["config"]
        scenario_col = "scenario_variant" if "scenario_variant" in fair.columns else "scenario"
        detector_cols = [c for c in DETECTOR_PARAM_COLS if c in fair.columns]
        group = [c for c in ["dataset", scenario_col, "method", *detector_cols] if c in fair.columns]
        fair_agg = (
            fair.groupby(group, dropna=False)[["jain", "gini", "fairness_gap", "reward_variance"]]
            .mean()
            .reset_index()
            .round(4)
        )
        fair_agg["method_label"] = fair_agg["method"].map(_method_label)
        latex = _df_to_latex(
            fair_agg,
            caption="Reward fairness metrics by scenario and method",
            label="tab:fairness_schema_v2",
        )
        (out_dir / "table_fairness.tex").write_text(latex, encoding="utf-8")
        log.info("LaTeX table saved: %s/table_fairness.tex", out_dir)
