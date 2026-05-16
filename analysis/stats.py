"""
stats.py — Tính toán các metrics thống kê: summary, fairness, và statistical tests.

Metrics sử dụng trực tiếp từ fl.metrics (không trùng lặp).
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

from fl.metrics import (
    jain_index,
    gini_coefficient,
    fairness_gap,
    contribution_reward_correlation,
    convergence_round as _convergence_round_impl,
    reward_leakage,
    reward_ratio,
    economic_incentive_index,
    false_positive_rate as fpr_metric,
    free_rider_detection_rate,
)

log = logging.getLogger(__name__)


# ── Metric helpers ────────────────────────────────────────────────────────────

_GROUP_KEY_CANDIDATES = [
    "dataset", "scenario", "scenario_variant", "config",
    "alpha", "dirichlet_alpha", "attack_label",
]


def _available_group_keys(df: pd.DataFrame, include_round: bool = False) -> List[str]:
    """Return grouping columns present in the loaded log DataFrame."""
    keys = [c for c in _GROUP_KEY_CANDIDATES if c in df.columns]
    if include_round and "round_num" in df.columns:
        keys.append("round_num")
    return keys


def _malicious_mask(df: pd.DataFrame) -> pd.Series:
    """Run-level attack metadata is derived by loader; keep a fallback for old logs."""
    if "is_malicious" in df.columns:
        return df["is_malicious"].fillna(False).astype(bool)
    client_types = df.get("client_type", pd.Series(["honest"] * len(df), index=df.index))
    return client_types.isin(["free_rider", "lazy", "label_noise", "malicious"])

def _reward_quality_correlation(quality: np.ndarray, reward: np.ndarray) -> float:
    """Pearson correlation giữa quality và reward (wrapper cho fl.metrics)."""
    return contribution_reward_correlation(quality, reward)


def convergence_round(acc_series: pd.Series, peak_ratio: float = 0.95, patience: int = 5) -> Optional[int]:
    """First stable round reaching peak_ratio * peak accuracy."""
    s = acc_series.dropna()
    if s.empty:
        return None
    return _convergence_round_impl(
        s.to_list(),
        rounds=[int(round_num) for round_num in s.index.to_list()],
        peak_ratio=peak_ratio,
        patience=patience,
    )


# ── Statistical significance tests ───────────────────────────────────────────

def statistical_tests(
    df: pd.DataFrame,
    metric_col: str = "global_accuracy",
    group_col: str = "config",
    alpha_sig: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise statistical significance tests between configs.

    Dùng Mann-Whitney U test (unpaired) cho mỗi cặp config.
    Trả về DataFrame với p-values và effect size.
    """
    configs = sorted(df[group_col].unique())
    if len(configs) < 2:
        return pd.DataFrame()

    # Lấy metric cuối cùng per run cho mỗi config
    run_metrics: Dict[str, List[float]] = {}
    for cfg in configs:
        cfg_df = df[df[group_col] == cfg]
        vals = []
        for _, run_grp in cfg_df.groupby("run_id"):
            by_round = run_grp.groupby("round_num")[metric_col].mean().sort_index().dropna()
            if not by_round.empty:
                vals.append(float(by_round.iloc[-1]))
        if vals:
            run_metrics[cfg] = vals

    results = []
    config_pairs = [(a, b) for i, a in enumerate(configs) for b in configs[i + 1:]]
    for cfg_a, cfg_b in config_pairs:
        if cfg_a not in run_metrics or cfg_b not in run_metrics:
            continue
        vals_a = run_metrics[cfg_a]
        vals_b = run_metrics[cfg_b]
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue
        try:
            stat_u, p_val = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        except ValueError:
            p_val = np.nan
            stat_u = np.nan
        # Effect size: rank-biserial correlation
        n_a, n_b = len(vals_a), len(vals_b)
        effect_size = 1 - (2 * stat_u) / (n_a * n_b) if (n_a * n_b) > 0 else np.nan
        results.append({
            "config_a": cfg_a,
            "config_b": cfg_b,
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": float(np.mean(vals_a)),
            "mean_b": float(np.mean(vals_b)),
            "diff": float(np.mean(vals_b) - np.mean(vals_a)),
            "U_stat": stat_u,
            "p_value": p_val,
            "effect_size": effect_size,
            "significant": bool(p_val < alpha_sig) if not np.isnan(p_val) else False,
        })
    return pd.DataFrame(results)


def wilcoxon_per_run(
    df: pd.DataFrame,
    config_a: str,
    config_b: str,
    metric_col: str = "global_accuracy",
    alpha_sig: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test (paired) cho cùng dataset/scenario nhưng khác config.
    Hữu ích khi so sánh cùng seed nhưng khác phương pháp.
    """
    group_keys = ["dataset", "scenario"]
    if "dirichlet_alpha" in df.columns:
        group_keys.append("dirichlet_alpha")

    pairs = []
    for keys, grp in df.groupby(group_keys):
        grp_a = grp[grp["config"] == config_a]
        grp_b = grp[grp["config"] == config_b]
        # Match by run_id suffix (timestamp may differ, match by seed/scenario)
        for run_id_a in grp_a["run_id"].unique():
            # Find matching run in B (same dataset/scenario/alpha but different config)
            alpha_val = grp_a[grp_a["run_id"] == run_id_a]["alpha"].iloc[0] if len(grp_a[grp_a["run_id"] == run_id_a]) > 0 else None
            if alpha_val is None:
                continue
            match_b = grp_b[grp_b["alpha"] == alpha_val]
            if match_b.empty:
                continue
            acc_a = grp_a[grp_a["run_id"] == run_id_a].groupby("round_num")[metric_col].mean().sort_index().dropna()
            acc_b = match_b.groupby("run_id").apply(
                lambda g: g.groupby("round_num")[metric_col].mean().sort_index().dropna().iloc[-1]
                if not g.groupby("round_num")[metric_col].mean().sort_index().dropna().empty else np.nan
            ).dropna()
            if not acc_a.empty and not acc_b.empty:
                pairs.append((float(acc_a.iloc[-1]), float(acc_b.mean())))

    if len(pairs) < 2:
        return {"test": "wilcoxon", "n_pairs": len(pairs), "statistic": np.nan, "p_value": np.nan, "significant": False}

    vals_a, vals_b = zip(*pairs)
    try:
        stat, p_val = wilcoxon(vals_a, vals_b, alternative="two-sided")
    except ValueError:
        stat, p_val = np.nan, np.nan

    return {
        "test": "wilcoxon",
        "config_a": config_a,
        "config_b": config_b,
        "n_pairs": len(pairs),
        "mean_diff": float(np.mean(vals_b) - np.mean(vals_a)),
        "statistic": stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha_sig) if not np.isnan(p_val) else False,
    }


# ── Main export functions ─────────────────────────────────────────────────────

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, config, alpha, attack_label) summary:
        - final_accuracy, peak_accuracy, convergence_round
        - mean_reward, reward_std, reward_cv
        - mean_reputation, n_clients, has_attack
        - reward_leakage, false_positive_rate, attack_detection_rate
        - reward_ratio, eii, jain, gini, fairness_gap
    """
    records = []
    group_keys = _available_group_keys(df)

    for keys, grp in df.groupby(group_keys, sort=True):
        rec = dict(zip(group_keys, keys))
        rec["n_runs"] = int(grp["run_id"].nunique()) if "run_id" in grp.columns else 1
        rec["n_rows"] = int(len(grp))
        rec["rounds_observed"] = int(grp["round_num"].nunique()) if "round_num" in grp.columns else 0
        rec["max_round"] = int(grp["round_num"].max()) if "round_num" in grp.columns and not grp.empty else 0

        # Per-run metrics, then aggregate
        run_ids = grp["run_id"].unique()
        run_metrics_list = []

        for run_id in run_ids:
            run_grp = grp[grp["run_id"] == run_id]
            run_rec: Dict = {}

            # Accuracy
            acc_by_round = (
                run_grp.groupby("round_num")["global_accuracy"]
                .mean().sort_index().dropna()
            )
            if not acc_by_round.empty:
                run_rec["final_accuracy"] = float(acc_by_round.iloc[-1])
                run_rec["peak_accuracy"] = float(acc_by_round.max())
            else:
                run_rec["final_accuracy"] = np.nan
                run_rec["peak_accuracy"] = np.nan
            run_rec["convergence_round"] = convergence_round(acc_by_round)

            # Rewards — honest clients only
            honest = run_grp[run_grp["is_honest"]]
            if not honest.empty:
                run_rec["mean_reward"] = float(honest["reward_eth"].mean())
                run_rec["reward_std"] = float(honest["reward_eth"].std())
                run_rec["reward_cv"] = (
                    run_rec["reward_std"] / (run_rec["mean_reward"] + 1e-12)
                    if not np.isnan(run_rec["mean_reward"]) else np.nan
                )
            else:
                run_rec["mean_reward"] = np.nan
                run_rec["reward_std"] = np.nan
                run_rec["reward_cv"] = np.nan

            # Reputation — last round
            rep_series = run_grp.groupby("round_num")["reputation"].mean().dropna()
            run_rec["mean_reputation"] = float(rep_series.iloc[-1]) if not rep_series.empty else np.nan

            run_rec["n_clients"] = int(run_grp["client_id"].nunique())
            run_rec["has_attack"] = bool(run_grp.get("has_attack", pd.Series([False])).any())

            # Reward leakage
            malicious_mask = _malicious_mask(run_grp)
            rewards_arr = run_grp["reward_eth"].fillna(0.0).to_numpy()
            run_rec["reward_leakage"] = reward_leakage(rewards_arr, malicious_mask.to_numpy())

            # Reward ratio (honest vs malicious)
            if run_rec["has_attack"] and "client_type" in run_grp.columns:
                honest_r = run_grp.loc[run_grp["client_type"] == "honest", "reward_eth"].dropna().values
                mal_r = run_grp.loc[malicious_mask, "reward_eth"].dropna().values
                has_reward_flow = (np.nansum(honest_r) + np.nansum(mal_r)) > 1e-12
                run_rec["reward_ratio"] = (
                    reward_ratio(honest_r, mal_r)
                    if has_reward_flow and len(mal_r) > 0 and len(honest_r) > 0
                    else np.nan
                )

                # EII
                honest_ds = run_grp.loc[run_grp["client_type"] == "honest", "data_size"].dropna().values
                mal_ds = run_grp.loc[malicious_mask, "data_size"].dropna().values
                if has_reward_flow and len(honest_r) > 0 and len(mal_r) > 0 and len(honest_ds) > 0 and len(mal_ds) > 0:
                    run_rec["eii"] = economic_incentive_index(
                        float(np.mean(honest_r)), float(np.mean(mal_r)),
                        float(np.mean(honest_ds)), float(np.mean(mal_ds)),
                    )
                else:
                    run_rec["eii"] = np.nan
            else:
                run_rec["reward_ratio"] = np.nan
                run_rec["eii"] = np.nan

            # Anomaly detection metrics
            if "is_anomaly" in run_grp.columns:
                anomaly = run_grp["is_anomaly"].fillna(False).astype(bool)
                honest_idx = run_grp.loc[~malicious_mask].index
                mal_idx = run_grp.loc[malicious_mask].index
                run_rec["false_positive_count"] = int(anomaly.loc[honest_idx].sum()) if len(honest_idx) > 0 else 0
                run_rec["attack_detected_count"] = int(anomaly.loc[mal_idx].sum()) if len(mal_idx) > 0 else 0
                run_rec["malicious_rows"] = int(len(mal_idx))
                run_rec["false_positive_rate"] = (
                    float(anomaly.loc[honest_idx].mean()) if len(honest_idx) > 0 else 0.0
                )
                run_rec["attack_detection_rate"] = (
                    float(anomaly.loc[mal_idx].mean()) if len(mal_idx) > 0 else np.nan
                )
                # FDR: detected malicious / total malicious
                if len(mal_idx) > 0:
                    detected_mal = set(run_grp.loc[anomaly].index) & set(mal_idx)
                    actual_mal = set(mal_idx)
                    run_rec["fdr"] = len(detected_mal) / len(actual_mal) if actual_mal else np.nan
                else:
                    run_rec["fdr"] = np.nan
            else:
                run_rec["false_positive_rate"] = np.nan
                run_rec["attack_detection_rate"] = np.nan
                run_rec["fdr"] = np.nan
                run_rec["false_positive_count"] = 0
                run_rec["attack_detected_count"] = 0
                run_rec["malicious_rows"] = 0

            # Fairness metrics per run
            if not honest.empty:
                paired = honest[["quality", "reward_eth"]].dropna()
                h_rewards = paired["reward_eth"].to_numpy(dtype=float)
                h_quality = paired["quality"].to_numpy(dtype=float)
                run_rec["jain"] = jain_index(h_rewards) if len(h_rewards) else np.nan
                run_rec["gini"] = gini_coefficient(h_rewards) if len(h_rewards) else np.nan
                run_rec["fairness_gap"] = fairness_gap(h_rewards, h_quality) if len(paired) else np.nan
            else:
                run_rec["jain"] = np.nan
                run_rec["gini"] = np.nan
                run_rec["fairness_gap"] = np.nan

            run_metrics_list.append(run_rec)

        # Aggregate across runs: mean of per-run metrics
        if len(run_metrics_list) == 1:
            rec.update(run_metrics_list[0])
        else:
            metric_keys = run_metrics_list[0].keys()
            for mk in metric_keys:
                vals = [r[mk] for r in run_metrics_list if not (isinstance(r[mk], float) and np.isnan(r[mk]))]
                if mk == "has_attack":
                    rec[mk] = any(vals)
                elif mk in {"false_positive_count", "attack_detected_count", "malicious_rows"}:
                    rec[mk] = int(np.sum(vals)) if vals else 0
                elif mk == "convergence_round":
                    valid = [v for v in vals if v is not None]
                    rec[mk] = int(round(np.mean(valid))) if valid else None
                elif mk == "n_clients":
                    rec[mk] = int(vals[0]) if vals else 0
                elif vals:
                    rec[mk] = float(np.mean(vals))
                else:
                    rec[mk] = np.nan

        records.append(rec)

    summary = pd.DataFrame(records)
    log.info("summary_metrics: %d rows", len(summary))
    return summary


def compute_fairness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, config, alpha, attack_label, round_num) fairness metrics:
        jain, gini, reward_quality_corr, fairness_gap, reward_variance
    """
    records = []
    group_keys = _available_group_keys(df, include_round=True)

    for keys, grp in df.groupby(group_keys, sort=True):
        honest = grp[grp["is_honest"]]
        if honest.empty:
            continue

        rec = dict(zip(group_keys, keys))
        paired = honest[["quality", "reward_eth"]].dropna()
        if paired.empty:
            continue

        rewards = paired["reward_eth"].to_numpy(dtype=float)
        quality = paired["quality"].to_numpy(dtype=float)

        rec["jain"] = jain_index(rewards)
        rec["gini"] = gini_coefficient(rewards)
        rec["reward_variance"] = float(np.var(rewards)) if len(rewards) > 1 else np.nan
        rec["reward_quality_corr"] = _reward_quality_correlation(quality, rewards)
        # fairness_gap(rewards, contributions) — đúng signature fl.metrics
        rec["fairness_gap"] = fairness_gap(rewards, quality)
        records.append(rec)

    fairness = pd.DataFrame(records)
    log.info("fairness_metrics: %d rows", len(fairness))
    return fairness
