"""
stats.py — Tính toán các metrics thống kê: summary và fairness.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from fl.metrics import convergence_round as stable_convergence_round
from fl.metrics import reward_leakage as reward_leakage_metric

log = logging.getLogger(__name__)

# ── Metric helpers ────────────────────────────────────────────────────────────

def jain_fairness_index(rewards: np.ndarray) -> float:
    """Jain's Fairness Index ∈ (0, 1]. 1 = perfectly fair."""
    r = np.asarray(rewards, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return np.nan
    denom = len(r) * np.sum(r ** 2)
    return float(np.sum(r) ** 2 / denom) if denom > 0 else 0.0


def gini_coefficient(rewards: np.ndarray) -> float:
    """Gini coefficient ∈ [0, 1]. 0 = perfect equality."""
    r = np.sort(np.asarray(rewards, dtype=float))
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0 or r.sum() == 0:
        return np.nan
    cumsum = np.cumsum(r)
    return float(
        (2 * np.sum(np.arange(1, n + 1) * r) - (n + 1) * cumsum[-1])
        / (n * cumsum[-1])
    )


def reward_quality_correlation(quality: np.ndarray, reward: np.ndarray) -> float:
    """Pearson correlation between contribution quality and reward received."""
    q = np.asarray(quality, dtype=float)
    r = np.asarray(reward, dtype=float)
    mask = ~(np.isnan(q) | np.isnan(r))
    q, r = q[mask], r[mask]
    if len(q) < 3 or np.std(q) < 1e-10 or np.std(r) < 1e-10:
        return np.nan
    corr, _ = pearsonr(q, r)
    return float(corr)


def fairness_gap(quality: np.ndarray, reward: np.ndarray) -> float:
    """
    Fairness Gap = mean |r_i/ΣR − q_i/ΣQ|.
    0 = reward perfectly proportional to quality.
    """
    q = np.asarray(quality, dtype=float)
    r = np.asarray(reward, dtype=float)
    mask = ~(np.isnan(q) | np.isnan(r))
    q, r = q[mask], r[mask]
    if len(q) == 0:
        return np.nan
    r_share = r / (r.sum() + 1e-12)
    q_share = q / (q.sum() + 1e-12)
    return float(np.abs(r_share - q_share).mean())


def convergence_round(acc_series: pd.Series, peak_ratio: float = 0.95, patience: int = 5) -> int | None:
    """First stable round reaching peak_ratio * peak accuracy."""
    s = acc_series.dropna()
    if s.empty:
        return None
    return stable_convergence_round(
        s.to_list(),
        rounds=[int(round_num) for round_num in s.index.to_list()],
        peak_ratio=peak_ratio,
        patience=patience,
    )


# ── Main export functions ─────────────────────────────────────────────────────

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, config, alpha) summary:
        - final_accuracy  : mean global_accuracy at last round
        - peak_accuracy   : max global_accuracy across all rounds
        - convergence_round : first stable round reaching 95% of peak acc
        - mean_reward     : mean reward_eth (honest clients)
        - reward_std      : std of reward_eth (honest clients)
        - mean_reputation : mean reputation (last round)
        - n_clients       : number of unique clients
        - has_attack      : any malicious client present
    """
    records = []
    group_keys = ["dataset", "scenario", "config", "alpha"]
    if "dirichlet_alpha" in df.columns:
        group_keys.append("dirichlet_alpha")

    for keys, grp in df.groupby(group_keys, sort=True):
        rec = dict(zip(group_keys, keys))

        # Global accuracy — mean per round, drop rounds where all clients have NaN
        acc_by_round = (
            grp.groupby("round_num")["global_accuracy"]
            .mean()
            .sort_index()
            .dropna()  # only keep rounds with real accuracy values
        )
        if not acc_by_round.empty:
            rec["final_accuracy"] = float(acc_by_round.iloc[-1])
            rec["peak_accuracy"]  = float(acc_by_round.max())
        else:
            rec["final_accuracy"] = np.nan
            rec["peak_accuracy"]  = np.nan
        rec["convergence_round"] = convergence_round(acc_by_round)

        # Rewards — honest clients only
        honest = grp[grp["is_honest"]]
        rec["mean_reward"] = float(honest["reward_eth"].mean()) if not honest.empty else np.nan
        rec["reward_std"]  = float(honest["reward_eth"].std())  if not honest.empty else np.nan
        rec["reward_cv"]   = (
            rec["reward_std"] / (rec["mean_reward"] + 1e-12)
            if not np.isnan(rec.get("mean_reward", np.nan)) else np.nan
        )

        # Reputation — last round with non-NaN values
        rep_series = grp.groupby("round_num")["reputation"].mean().dropna()
        rec["mean_reputation"] = float(rep_series.iloc[-1]) if not rep_series.empty else np.nan

        rec["n_clients"] = int(grp["client_id"].nunique())
        rec["has_attack"] = bool(grp.get("has_attack", pd.Series([False])).any())

        client_types = grp.get("client_type", pd.Series(["honest"] * len(grp), index=grp.index))
        malicious_mask = client_types.isin(["free_rider", "lazy"]).to_numpy()
        rewards = grp["reward_eth"].fillna(0.0).to_numpy()
        rec["reward_leakage"] = reward_leakage_metric(rewards, malicious_mask)

        if "is_anomaly" in grp.columns:
            anomaly = grp["is_anomaly"].fillna(False).astype(bool)
            honest_rows = grp.loc[~malicious_mask]
            malicious_rows = grp.loc[malicious_mask]
            rec["false_positive_rate"] = (
                float(anomaly.loc[honest_rows.index].mean()) if not honest_rows.empty else 0.0
            )
            rec["attack_detection_rate"] = (
                float(anomaly.loc[malicious_rows.index].mean()) if not malicious_rows.empty else np.nan
            )
        else:
            rec["false_positive_rate"] = np.nan
            rec["attack_detection_rate"] = np.nan

        records.append(rec)

    summary = pd.DataFrame(records)
    log.info("summary_metrics: %d rows", len(summary))
    return summary


def compute_fairness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(dataset, scenario, config, alpha, round_num) fairness metrics:
        jain, gini, reward_quality_corr, fairness_gap, reward_variance
    Also aggregated per group (mean over rounds).
    """
    records = []
    group_keys = ["dataset", "scenario", "config", "alpha"]
    if "dirichlet_alpha" in df.columns:
        group_keys.append("dirichlet_alpha")
    group_keys.append("round_num")

    for keys, grp in df.groupby(group_keys, sort=True):
        honest = grp[grp["is_honest"]]
        if honest.empty:
            continue

        rec = dict(zip(group_keys, keys))
        rewards  = honest["reward_eth"].dropna().values
        quality  = honest["quality"].dropna().values

        rec["jain"]                = jain_fairness_index(rewards)
        rec["gini"]                = gini_coefficient(rewards)
        rec["reward_variance"]     = float(np.var(rewards)) if len(rewards) > 1 else np.nan
        rec["reward_quality_corr"] = reward_quality_correlation(
            quality[:len(rewards)], rewards[:len(quality)]
        )
        rec["fairness_gap"] = fairness_gap(
            quality[:len(rewards)], rewards[:len(quality)]
        )
        records.append(rec)

    fairness = pd.DataFrame(records)
    log.info("fairness_metrics: %d rows", len(fairness))
    return fairness
