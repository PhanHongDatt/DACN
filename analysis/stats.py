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
    "dataset", "scenario", "scenario_variant",
    "method", "method_label",
    "aggregation_method", "reward_policy",
    "beta", "gamma", "delta",
    "mad_threshold", "cosine_threshold", "direction_min_norm_z",
    "min_honest_ratio", "fallback_hard_z",
    "suspicion_decay", "suspicion_threshold", "low_quality_z_threshold",
    "low_quality_suspicion", "zero_data_suspicion", "anomaly_suspicion",
    "authenticity_suspicion", "low_authenticity_threshold",
    "high_update_norm_z_threshold", "inefficient_update_suspicion",
    "dirichlet_alpha", "attack_type", "attack_label",
]

_SUMMARY_TRACE_COLS = [
    "num_clients", "num_rounds", "local_epochs", "batch_size",
    "learning_rate", "client_fraction", "data_split", "data_imbalance",
    "persistent_clients",
]
_TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", ""}


def _available_group_keys(df: pd.DataFrame, include_round: bool = False) -> List[str]:
    """Return grouping columns present in the loaded log DataFrame."""
    keys = [c for c in _GROUP_KEY_CANDIDATES if c in df.columns]
    if include_round and "round_num" in df.columns:
        keys.append("round_num")
    return keys


def _format_unique_values(series: pd.Series) -> str:
    """Return a stable, compact representation of non-null values."""
    values = series.dropna().unique().tolist()
    if not values:
        return ""
    values = sorted(values, key=lambda v: str(v))
    return ",".join(str(v) for v in values)


def _single_or_mixed(series: pd.Series):
    """Return the single non-null value, or a stable mixed-value marker."""
    values = series.dropna().unique().tolist()
    if not values:
        return np.nan
    if len(values) == 1:
        return values[0]
    return "mixed:" + _format_unique_values(series)


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
    """Run-level attack metadata is derived by loader; keep a fallback for old logs."""
    client_types = df.get("client_type", pd.Series(["honest"] * len(df), index=df.index))
    type_fallback = client_types.isin([
        "free_rider", "stealth_free_rider", "lazy", "label_noise",
        "sign_flip", "malicious"
    ])
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


def _honest_mask(df: pd.DataFrame) -> pd.Series:
    """Ground-truth honest mask, independent from reward eligibility."""
    return ~_malicious_mask(df)


def _bool_mask(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """Return a boolean Series aligned to df.index, tolerating missing columns."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return _nullable_bool_series(df[col]).fillna(default).astype(bool)


def _participating_mask(df: pd.DataFrame) -> pd.Series:
    """Rows marked not_participating should not count in confusion denominators."""
    if "detection_reason" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    return df["detection_reason"].astype(str).ne("not_participating")


def _reward_blocked_mask(df: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    """
    Reward-quarantine mask for schema v2 logs.

    Prefer explicit reward_blocked. For old logs, infer from is_honest because
    it represented reward eligibility. Fall back to reward_eth≈0 only when no
    eligibility column exists.
    """
    if "reward_blocked" in df.columns:
        return _bool_mask(df, "reward_blocked")

    participating = _participating_mask(df)
    if "reward_eligible" in df.columns:
        return (~_bool_mask(df, "reward_eligible")) & participating
    if "is_honest" in df.columns:
        return (~_bool_mask(df, "is_honest")) & participating

    if "reward_eth" in df.columns:
        reward = pd.to_numeric(df["reward_eth"], errors="coerce").fillna(0.0)
        return reward.le(eps) & participating

    return pd.Series(False, index=df.index, dtype=bool)

def _reward_quality_correlation(quality: np.ndarray, reward: np.ndarray) -> float:
    """Pearson correlation giữa quality và reward (wrapper cho fl.metrics)."""
    return contribution_reward_correlation(quality, reward)


def _alignment_signal_col(df: pd.DataFrame) -> Optional[str]:
    """
    Return the server-side alignment signal used for reward diagnostics.

    Prefer cosine_to_reference because quality/csra reward currently uses
    server-side cosine. alignment_score is retained for FedLAW-style logs.
    """
    for col in ("cosine_to_reference", "alignment_score"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _nonnegative_signal(values: pd.Series) -> np.ndarray:
    """Match reward policy semantics by clipping negative alignment to zero."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.clip(arr, 0.0, None)


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
    group_col: str = "method",
    alpha_sig: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise statistical significance tests between schema-v2 methods.

    Dùng Mann-Whitney U test (unpaired) cho mỗi cặp method.
    Trả về DataFrame với p-values và effect size.
    """
    configs = sorted(df[group_col].unique())
    if len(configs) < 2:
        return pd.DataFrame()

    # Lấy metric cuối cùng per run cho mỗi method
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
    Wilcoxon signed-rank test (paired) cho cùng điều kiện và cùng seed.

    Mỗi pair là một run của config_a ghép với đúng một run của config_b có
    cùng dataset/scenario/dirichlet/attack/seed. Nếu một key có nhiều hơn một
    run ở một phía, key đó bị bỏ qua để tránh many-to-many pairing sai.
    """
    method_col = "method" if "method" in df.columns else "config"
    if method_col not in df.columns or "run_id" not in df.columns:
        return {
            "test": "wilcoxon",
            "config_a": config_a,
            "config_b": config_b,
            "n_pairs": 0,
            "paired_on": "",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    pair_cols = [
        c for c in [
            "dataset", "scenario", "scenario_variant", "dirichlet_alpha",
            "attack_type", "attack_label", "seed",
            "num_clients", "num_rounds", "local_epochs", "batch_size",
            "learning_rate", "client_fraction", "data_split",
            "data_imbalance", "persistent_clients",
        ] if c in df.columns
    ]

    def _last_metric(run_grp: pd.DataFrame) -> float:
        by_round = (
            run_grp.groupby("round_num")[metric_col]
            .mean().sort_index().dropna()
        )
        return float(by_round.iloc[-1]) if not by_round.empty else np.nan

    run_rows = []
    meta_cols = [method_col, "run_id", *pair_cols]
    for run_id, run_grp in df.groupby("run_id", dropna=False):
        metric = _last_metric(run_grp)
        if np.isnan(metric):
            continue
        first = run_grp.iloc[0]
        rec = {col: first[col] for col in meta_cols if col in run_grp.columns}
        rec["run_id"] = run_id
        rec["metric"] = metric
        run_rows.append(rec)

    if not run_rows:
        return {
            "test": "wilcoxon",
            "config_a": config_a,
            "config_b": config_b,
            "n_pairs": 0,
            "paired_on": ",".join(pair_cols),
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    run_df = pd.DataFrame(run_rows)
    a = run_df[run_df[method_col] == config_a].copy()
    b = run_df[run_df[method_col] == config_b].copy()
    if a.empty or b.empty or not pair_cols:
        return {
            "test": "wilcoxon",
            "config_a": config_a,
            "config_b": config_b,
            "n_pairs": 0,
            "paired_on": ",".join(pair_cols),
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    a_counts = a.groupby(pair_cols, dropna=False).size().rename("n_a").reset_index()
    b_counts = b.groupby(pair_cols, dropna=False).size().rename("n_b").reset_index()
    counts = a_counts.merge(b_counts, on=pair_cols, how="outer")
    counts[["n_a", "n_b"]] = counts[["n_a", "n_b"]].fillna(0).astype(int)
    valid_keys = counts[(counts["n_a"] == 1) & (counts["n_b"] == 1)][pair_cols]
    skipped_unpaired = int(((counts["n_a"] == 0) | (counts["n_b"] == 0)).sum())
    skipped_ambiguous = int(((counts["n_a"] > 1) | (counts["n_b"] > 1)).sum())

    if valid_keys.empty:
        pairs = []
    else:
        a_valid = valid_keys.merge(a, on=pair_cols, how="inner")
        b_valid = valid_keys.merge(b, on=pair_cols, how="inner")
        paired = a_valid.merge(
            b_valid,
            on=pair_cols,
            suffixes=("_a", "_b"),
            how="inner",
        )
        pairs = list(zip(paired["metric_a"].astype(float), paired["metric_b"].astype(float)))

    if len(pairs) < 2:
        return {
            "test": "wilcoxon",
            "config_a": config_a,
            "config_b": config_b,
            "n_pairs": len(pairs),
            "paired_on": ",".join(pair_cols),
            "skipped_unpaired": skipped_unpaired,
            "skipped_ambiguous": skipped_ambiguous,
            "mean_a": float(np.mean([p[0] for p in pairs])) if pairs else np.nan,
            "mean_b": float(np.mean([p[1] for p in pairs])) if pairs else np.nan,
            "mean_diff": float(np.mean([p[1] - p[0] for p in pairs])) if pairs else np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

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
        "paired_on": ",".join(pair_cols),
        "skipped_unpaired": skipped_unpaired,
        "skipped_ambiguous": skipped_ambiguous,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "mean_diff": float(np.mean(vals_b) - np.mean(vals_a)),
        "statistic": stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha_sig) if not np.isnan(p_val) else False,
    }


def paired_wilcoxon_tests(
    df: pd.DataFrame,
    metric_col: str = "global_accuracy",
    group_col: str = "method",
    alpha_sig: float = 0.05,
) -> pd.DataFrame:
    """Run paired Wilcoxon tests for all method/config pairs."""
    if group_col not in df.columns:
        return pd.DataFrame()
    configs = sorted(df[group_col].dropna().unique())
    if len(configs) < 2:
        return pd.DataFrame()

    records = []
    for i, cfg_a in enumerate(configs):
        for cfg_b in configs[i + 1:]:
            records.append(
                wilcoxon_per_run(
                    df,
                    str(cfg_a),
                    str(cfg_b),
                    metric_col=metric_col,
                    alpha_sig=alpha_sig,
                )
            )
    return pd.DataFrame(records)


# ── Main export functions ─────────────────────────────────────────────────────

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, method, beta/gamma/delta, attack_label) summary:
        - final_accuracy, peak_accuracy, convergence_round
        - mean_reward, reward_std, reward_cv
        - mean_reputation, n_clients, has_attack
        - reward_leakage, false_positive_rate, attack_detection_rate
        - reward_ratio, eii, jain, gini, fairness_gap
    """
    records = []
    group_keys = _available_group_keys(df)

    for keys, grp in df.groupby(group_keys, sort=True, dropna=False):
        rec = dict(zip(group_keys, keys))
        rec["n_runs"] = int(grp["run_id"].nunique()) if "run_id" in grp.columns else 1
        if "run_id" in grp.columns:
            rec["run_ids"] = _format_unique_values(grp["run_id"].astype(str))
        if "seed" in grp.columns:
            rec["seed_count"] = int(grp["seed"].dropna().nunique())
            rec["seeds"] = _format_unique_values(grp["seed"])
        for col in _SUMMARY_TRACE_COLS:
            if col in grp.columns:
                rec[col] = _single_or_mixed(grp[col])
        rec["n_rows"] = int(len(grp))
        rec["rounds_observed"] = int(grp["round_num"].nunique()) if "round_num" in grp.columns else 0
        rec["max_round"] = int(grp["round_num"].max()) if "round_num" in grp.columns and not grp.empty else 0

        # Per-run metrics, then aggregate
        run_ids = grp["run_id"].unique()
        run_metrics_list = []

        for run_id in run_ids:
            run_grp = grp[grp["run_id"] == run_id]
            run_rec: Dict = {}
            malicious_mask = _malicious_mask(run_grp)
            participating = _participating_mask(run_grp)
            honest_participating = (~malicious_mask) & participating

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

            if "global_loss" in run_grp.columns:
                loss_by_round = (
                    run_grp.groupby("round_num")["global_loss"]
                    .mean().sort_index().dropna()
                )
                if not loss_by_round.empty:
                    run_rec["final_loss"] = float(loss_by_round.iloc[-1])
                    run_rec["min_loss"] = float(loss_by_round.min())
                else:
                    run_rec["final_loss"] = np.nan
                    run_rec["min_loss"] = np.nan

            # Rewards — honest clients only
            honest = run_grp.loc[honest_participating]
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
            rewards_arr = run_grp["reward_eth"].fillna(0.0).to_numpy()
            run_rec["reward_leakage"] = reward_leakage(rewards_arr, malicious_mask.to_numpy())

            # Reward ratio (honest vs malicious)
            if run_rec["has_attack"]:
                malicious_participating = malicious_mask & participating
                honest_r = run_grp.loc[honest_participating, "reward_eth"].dropna().values
                mal_r = run_grp.loc[malicious_participating, "reward_eth"].dropna().values
                has_reward_flow = (np.nansum(honest_r) + np.nansum(mal_r)) > 1e-12
                run_rec["reward_ratio"] = (
                    reward_ratio(honest_r, mal_r)
                    if has_reward_flow and len(mal_r) > 0 and len(honest_r) > 0
                    else np.nan
                )

                # EII
                honest_ds = run_grp.loc[honest_participating, "data_size"].dropna().values
                mal_ds = run_grp.loc[malicious_participating, "data_size"].dropna().values
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

            # Detection and reward-quarantine metrics. Keep legacy aliases:
            # false_positive_rate == false_positive_detection_rate.
            anomaly = _bool_mask(run_grp, "is_anomaly") & participating
            reward_blocked = _reward_blocked_mask(run_grp) & participating
            data_commitment = _bool_mask(
                run_grp, "data_commitment_anomaly"
            ) & participating
            low_quality = _bool_mask(run_grp, "low_quality_outlier") & participating
            inefficient_update = _bool_mask(run_grp, "inefficient_update") & participating
            suspicion_quarantine = _bool_mask(
                run_grp, "suspicion_quarantine"
            ) & participating
            honest_idx = run_grp.loc[(~malicious_mask) & participating].index
            mal_idx = run_grp.loc[malicious_mask & participating].index

            run_rec["false_positive_detection_count"] = (
                int(anomaly.loc[honest_idx].sum()) if len(honest_idx) > 0 else 0
            )
            run_rec["attack_detected_count"] = (
                int(anomaly.loc[mal_idx].sum()) if len(mal_idx) > 0 else 0
            )
            run_rec["false_positive_quarantine_count"] = (
                int(reward_blocked.loc[honest_idx].sum()) if len(honest_idx) > 0 else 0
            )
            run_rec["attack_reward_block_count"] = (
                int(reward_blocked.loc[mal_idx].sum()) if len(mal_idx) > 0 else 0
            )
            run_rec["data_commitment_anomaly_count"] = int(data_commitment.sum())
            run_rec["low_quality_outlier_count"] = int(low_quality.sum())
            run_rec["inefficient_update_count"] = int(inefficient_update.sum())
            run_rec["suspicion_quarantine_count"] = int(suspicion_quarantine.sum())
            run_rec["malicious_rows"] = int(len(mal_idx))
            run_rec["honest_rows"] = int(len(honest_idx))
            run_rec["false_positive_detection_rate"] = (
                float(anomaly.loc[honest_idx].mean()) if len(honest_idx) > 0 else 0.0
            )
            run_rec["attack_detection_rate"] = (
                float(anomaly.loc[mal_idx].mean()) if len(mal_idx) > 0 else np.nan
            )
            run_rec["false_positive_quarantine_rate"] = (
                float(reward_blocked.loc[honest_idx].mean()) if len(honest_idx) > 0 else 0.0
            )
            run_rec["attack_reward_block_rate"] = (
                float(reward_blocked.loc[mal_idx].mean()) if len(mal_idx) > 0 else np.nan
            )
            denom = int(participating.sum())
            run_rec["data_commitment_anomaly_rate"] = (
                float(data_commitment.sum() / denom) if denom > 0 else np.nan
            )
            run_rec["low_quality_outlier_rate"] = (
                float(low_quality.sum() / denom) if denom > 0 else np.nan
            )
            run_rec["inefficient_update_rate"] = (
                float(inefficient_update.sum() / denom) if denom > 0 else np.nan
            )
            run_rec["suspicion_quarantine_rate"] = (
                float(suspicion_quarantine.sum() / denom) if denom > 0 else np.nan
            )
            run_rec["false_positive_count"] = run_rec["false_positive_detection_count"]
            run_rec["false_positive_rate"] = run_rec["false_positive_detection_rate"]

            reward = pd.to_numeric(
                run_grp.get("reward_eth", pd.Series(0.0, index=run_grp.index)),
                errors="coerce",
            ).fillna(0.0)
            reward_eligible_honest_idx = run_grp.loc[
                (~malicious_mask) & participating & (~reward_blocked)
            ].index
            starved_honest = reward.loc[reward_eligible_honest_idx].le(1e-12)
            run_rec["reward_eligible_honest_rows"] = int(
                len(reward_eligible_honest_idx)
            )
            run_rec["honest_reward_starvation_count"] = (
                int(starved_honest.sum())
                if len(reward_eligible_honest_idx) > 0 else 0
            )
            run_rec["honest_reward_starvation_rate"] = (
                float(starved_honest.mean())
                if len(reward_eligible_honest_idx) > 0 else 0.0
            )

            # FDR: detected malicious / total malicious rows.
            if len(mal_idx) > 0:
                detected_mal = set(run_grp.loc[anomaly].index) & set(mal_idx)
                actual_mal = set(mal_idx)
                run_rec["fdr"] = len(detected_mal) / len(actual_mal) if actual_mal else np.nan
            else:
                run_rec["fdr"] = np.nan

            # Fairness metrics per run
            if not honest.empty:
                paired = honest[["quality", "reward_eth"]].dropna()
                h_rewards = paired["reward_eth"].to_numpy(dtype=float)
                h_quality = paired["quality"].to_numpy(dtype=float)
                run_rec["jain"] = jain_index(h_rewards) if len(h_rewards) else np.nan
                run_rec["gini"] = gini_coefficient(h_rewards) if len(h_rewards) else np.nan
                run_rec["fairness_gap"] = fairness_gap(h_rewards, h_quality) if len(paired) else np.nan

                alignment_col = _alignment_signal_col(honest)
                if alignment_col:
                    align_paired = honest[[alignment_col, "reward_eth"]].dropna()
                    if not align_paired.empty:
                        a_rewards = align_paired["reward_eth"].to_numpy(dtype=float)
                        a_signal = _nonnegative_signal(align_paired[alignment_col])
                        run_rec["reward_alignment_corr"] = _reward_quality_correlation(
                            a_signal, a_rewards
                        )
                        run_rec["fairness_gap_alignment"] = fairness_gap(
                            a_rewards, a_signal
                        )
                    else:
                        run_rec["reward_alignment_corr"] = np.nan
                        run_rec["fairness_gap_alignment"] = np.nan
                else:
                    run_rec["reward_alignment_corr"] = np.nan
                    run_rec["fairness_gap_alignment"] = np.nan
            else:
                run_rec["jain"] = np.nan
                run_rec["gini"] = np.nan
                run_rec["fairness_gap"] = np.nan
                run_rec["reward_alignment_corr"] = np.nan
                run_rec["fairness_gap_alignment"] = np.nan

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
                elif mk in {
                    "false_positive_count",
                    "false_positive_detection_count",
                    "attack_detected_count",
                    "false_positive_quarantine_count",
                    "attack_reward_block_count",
                    "data_commitment_anomaly_count",
                    "low_quality_outlier_count",
                    "inefficient_update_count",
                    "suspicion_quarantine_count",
                    "malicious_rows",
                    "honest_rows",
                    "reward_eligible_honest_rows",
                    "honest_reward_starvation_count",
                }:
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


def compute_detection_confusion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-run, per-round confusion counts for two gates:
      - anomaly detection gate: is_anomaly
      - reward quarantine gate: reward_blocked

    This table is the traceable confusion matrix layer. It is derived from
    per-client logs, so the main training CSV does not need duplicated
    round-level columns on every client row.
    """
    records = []
    group_keys = [
        c for c in [
            "run_id", "dataset", "scenario", "scenario_variant",
            "method", "method_label", "aggregation_method", "reward_policy",
            "beta", "gamma", "delta",
            "mad_threshold", "cosine_threshold", "direction_min_norm_z",
            "min_honest_ratio", "fallback_hard_z",
            "suspicion_decay", "suspicion_threshold", "low_quality_z_threshold",
            "low_quality_suspicion", "zero_data_suspicion", "anomaly_suspicion",
            "authenticity_suspicion", "low_authenticity_threshold",
            "high_update_norm_z_threshold", "inefficient_update_suspicion",
            "dirichlet_alpha",
            "attack_type", "attack_label", "round_num",
        ] if c in df.columns
    ]
    if not group_keys:
        return pd.DataFrame()

    for keys, grp in df.groupby(group_keys, sort=True, dropna=False):
        rec = dict(zip(group_keys, keys))
        participating = _participating_mask(grp)
        malicious = _malicious_mask(grp) & participating
        honest = _honest_mask(grp) & participating
        anomaly = _bool_mask(grp, "is_anomaly") & participating
        reward_blocked = _reward_blocked_mask(grp) & participating
        data_commitment = _bool_mask(
            grp, "data_commitment_anomaly"
        ) & participating
        low_quality = _bool_mask(grp, "low_quality_outlier") & participating
        inefficient_update = _bool_mask(grp, "inefficient_update") & participating
        suspicion_quarantine = _bool_mask(
            grp, "suspicion_quarantine"
        ) & participating

        rec["n_participating"] = int(participating.sum())
        rec["n_malicious"] = int(malicious.sum())
        rec["n_honest"] = int(honest.sum())

        rec["detection_tp"] = int((malicious & anomaly).sum())
        rec["detection_fn"] = int((malicious & ~anomaly).sum())
        rec["detection_fp"] = int((honest & anomaly).sum())
        rec["detection_tn"] = int((honest & ~anomaly).sum())
        rec["quarantine_tp"] = int((malicious & reward_blocked).sum())
        rec["quarantine_fn"] = int((malicious & ~reward_blocked).sum())
        rec["quarantine_fp"] = int((honest & reward_blocked).sum())
        rec["quarantine_tn"] = int((honest & ~reward_blocked).sum())

        rec["attack_detection_rate"] = (
            rec["detection_tp"] / rec["n_malicious"]
            if rec["n_malicious"] > 0 else np.nan
        )
        rec["attack_reward_block_rate"] = (
            rec["quarantine_tp"] / rec["n_malicious"]
            if rec["n_malicious"] > 0 else np.nan
        )
        rec["false_positive_detection_rate"] = (
            rec["detection_fp"] / rec["n_honest"]
            if rec["n_honest"] > 0 else np.nan
        )
        rec["false_positive_quarantine_rate"] = (
            rec["quarantine_fp"] / rec["n_honest"]
            if rec["n_honest"] > 0 else np.nan
        )

        reward = pd.to_numeric(
            grp.get("reward_eth", pd.Series(0.0, index=grp.index)),
            errors="coerce",
        ).fillna(0.0)
        total_reward = float(reward.loc[participating].sum())
        malicious_reward = float(reward.loc[malicious].sum())
        honest_reward = float(reward.loc[honest].sum())
        rec["attacker_reward_share"] = (
            malicious_reward / total_reward if total_reward > 1e-12 else 0.0
        )
        rec["honest_reward_share"] = (
            honest_reward / total_reward if total_reward > 1e-12 else 0.0
        )
        rec["data_commitment_anomaly_count"] = int(data_commitment.sum())
        rec["data_commitment_anomaly_rate"] = (
            rec["data_commitment_anomaly_count"] / rec["n_participating"]
            if rec["n_participating"] > 0 else np.nan
        )
        rec["low_quality_outlier_count"] = int(low_quality.sum())
        rec["low_quality_outlier_rate"] = (
            rec["low_quality_outlier_count"] / rec["n_participating"]
            if rec["n_participating"] > 0 else np.nan
        )
        rec["inefficient_update_count"] = int(inefficient_update.sum())
        rec["inefficient_update_rate"] = (
            rec["inefficient_update_count"] / rec["n_participating"]
            if rec["n_participating"] > 0 else np.nan
        )
        rec["suspicion_quarantine_count"] = int(suspicion_quarantine.sum())
        rec["suspicion_quarantine_rate"] = (
            rec["suspicion_quarantine_count"] / rec["n_participating"]
            if rec["n_participating"] > 0 else np.nan
        )

        if "direction_anomaly" in grp.columns:
            direction = _bool_mask(grp, "direction_anomaly") & participating
            rec["direction_anomaly_count"] = int(direction.sum())
            rec["direction_anomaly_rate"] = (
                rec["direction_anomaly_count"] / rec["n_participating"]
                if rec["n_participating"] > 0 else np.nan
            )
        else:
            rec["direction_anomaly_count"] = 0
            rec["direction_anomaly_rate"] = np.nan

        if "detection_reason" in grp.columns:
            reasons = grp.loc[participating, "detection_reason"].astype(str)
            fallback_any = reasons.str.contains("fallback", na=False)
            fallback = reasons.str.contains("fallback_accept_all", na=False)
            fallback_soft = reasons.str.contains(
                "fallback_soft|fallback_hard_block", na=False
            )
            rec["fallback_triggered_count"] = int(fallback_any.sum())
            rec["fallback_triggered_rate"] = (
                float(fallback_any.mean()) if rec["n_participating"] > 0 else np.nan
            )
            rec["fallback_accept_all_count"] = int(fallback.sum())
            rec["fallback_accept_all_rate"] = (
                float(fallback.mean()) if rec["n_participating"] > 0 else np.nan
            )
            rec["fallback_soft_filter_count"] = int(fallback_soft.sum())
            rec["fallback_soft_filter_rate"] = (
                float(fallback_soft.mean()) if rec["n_participating"] > 0 else np.nan
            )
        else:
            rec["fallback_triggered_count"] = 0
            rec["fallback_triggered_rate"] = np.nan
            rec["fallback_accept_all_count"] = 0
            rec["fallback_accept_all_rate"] = np.nan
            rec["fallback_soft_filter_count"] = 0
            rec["fallback_soft_filter_rate"] = np.nan

        records.append(rec)

    out = pd.DataFrame(records)
    log.info("detection_confusion_metrics: %d rows", len(out))
    return out


def compute_fairness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, method, beta/gamma/delta, attack_label, round_num) fairness metrics:
        jain, gini, reward_quality_corr, fairness_gap, reward_variance
    """
    records = []
    group_keys = _available_group_keys(df, include_round=True)

    for keys, grp in df.groupby(group_keys, sort=True, dropna=False):
        participating = _participating_mask(grp)
        honest = grp.loc[(~_malicious_mask(grp)) & participating]
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

        alignment_col = _alignment_signal_col(honest)
        if alignment_col:
            align_paired = honest[[alignment_col, "reward_eth"]].dropna()
            if not align_paired.empty:
                a_rewards = align_paired["reward_eth"].to_numpy(dtype=float)
                a_signal = _nonnegative_signal(align_paired[alignment_col])
                rec["reward_alignment_corr"] = _reward_quality_correlation(
                    a_signal, a_rewards
                )
                rec["fairness_gap_alignment"] = fairness_gap(a_rewards, a_signal)
                rec["alignment_signal"] = alignment_col
            else:
                rec["reward_alignment_corr"] = np.nan
                rec["fairness_gap_alignment"] = np.nan
                rec["alignment_signal"] = ""
        else:
            rec["reward_alignment_corr"] = np.nan
            rec["fairness_gap_alignment"] = np.nan
            rec["alignment_signal"] = ""
        records.append(rec)

    fairness = pd.DataFrame(records)
    log.info("fairness_metrics: %d rows", len(fairness))
    return fairness
