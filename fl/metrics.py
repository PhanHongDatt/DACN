"""
metrics.py — Tất cả metric cho 3 nhóm thực nghiệm.

Nhóm 1 (RQ1): AbC, AR, Global Accuracy
Nhóm 2 (RQ2): Jain's Index, CRC, Gini, Fairness Gap (FG) *metric mới*
Nhóm 3 (RQ3): FDR, RR, EII
"""
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import pearsonr


# ── Nhóm 1 ──────────────────────────────────────────────────
def abc_metric(acc_low: List[float], acc_high: List[float]) -> float:
    """
    Area between Curves (kế thừa FedCCEA).
    acc_low[k]  = accuracy sau khi loại k client có W_new thấp nhất
    acc_high[k] = accuracy sau khi loại k client có W_new cao nhất
    Lớn hơn = đo đóng góp chính xác hơn.
    """
    return float(sum(al - ah for al, ah in zip(acc_low, acc_high)))


def accuracy_reversal(acc_low: List[float], acc_high: List[float]) -> int:
    """AR: số lần acc_high > acc_low (hiện tượng đảo ngược — lỗi nghiêm trọng)."""
    return sum(1 for al, ah in zip(acc_low, acc_high) if ah > al)


# ── Nhóm 2 ──────────────────────────────────────────────────
def jain_index(rewards: np.ndarray) -> float:
    """Jain's Fairness Index. J=1: đồng đều; J=1/n: một client nhận tất cả."""
    rewards = np.array(rewards, dtype=float)
    denom = len(rewards) * np.sum(rewards ** 2)
    return float(np.sum(rewards) ** 2 / denom) if denom > 0 else 0.0


def gini_coefficient(rewards: np.ndarray) -> float:
    """Gini Coefficient. G=0: bình đẳng hoàn toàn; G→1: bất bình đẳng cao."""
    rewards = np.sort(np.array(rewards, dtype=float))
    n = len(rewards)
    cumsum = np.cumsum(rewards)
    return float((2 * np.sum((np.arange(1, n+1)) * rewards) - (n+1) * cumsum[-1])
                 / (n * cumsum[-1])) if cumsum[-1] > 0 else 0.0


def contribution_reward_correlation(
    contributions: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Pearson correlation giữa đóng góp thực tế và phần thưởng nhận được.
    CRC → 1: phần thưởng bám sát đóng góp; CRC ≈ 0: ngẫu nhiên.
    """
    c = np.array(contributions, dtype=float)
    r = np.array(rewards, dtype=float)
    if np.std(c) < 1e-10 or np.std(r) < 1e-10:
        return 1.0 if np.allclose(c, r) else 0.0
    corr, _ = pearsonr(c, r)
    return float(corr)


def fairness_gap(rewards: np.ndarray, contributions: np.ndarray) -> float:
    """
    Fairness Gap (FG) — metric đề xuất mới.
    FG = (1/n) * Σ|r_i/Σr - c_i/Σc|
    FG=0: phần thưởng tỷ lệ hoàn hảo với đóng góp.
    FG thấp hơn = công bằng hơn.
    """
    r = np.array(rewards, dtype=float)
    c = np.array(contributions, dtype=float)
    r_ratio = r / (r.sum() + 1e-10)
    c_ratio = c / (c.sum() + 1e-10)
    return float(np.abs(r_ratio - c_ratio).mean())


# ── Nhóm 3 ──────────────────────────────────────────────────
def free_rider_detection_rate(
    detected_free_riders: List[int],
    actual_free_riders: List[int]
) -> float:
    """FDR = |phát hiện đúng| / |tổng free-rider thực sự|"""
    if not actual_free_riders:
        return 1.0
    detected = set(detected_free_riders)
    actual   = set(actual_free_riders)
    return len(detected & actual) / len(actual)


def false_positive_rate(
    detected_clients: List[int],
    malicious_clients: List[int],
    all_clients: Optional[List[int]] = None,
) -> float:
    """FPR = honest clients incorrectly detected as anomalies / total honest clients."""
    detected = set(detected_clients)
    malicious = set(malicious_clients)
    if all_clients is None:
        all_clients = sorted(detected | malicious)
    honest = set(all_clients) - malicious
    if not honest:
        return 0.0
    return len(detected & honest) / len(honest)


def reward_leakage(rewards: np.ndarray, malicious_mask: np.ndarray) -> float:
    """Reward leakage = reward captured by malicious clients / total reward."""
    r = np.asarray(rewards, dtype=float)
    mask = np.asarray(malicious_mask, dtype=bool)
    total = float(np.nansum(r))
    if total <= 1e-12 or len(r) == 0:
        return 0.0
    return float(np.nansum(r[mask]) / total)


def convergence_round(
    accuracies: List[float],
    rounds: Optional[List[int]] = None,
    peak_ratio: float = 0.95,
    patience: int = 5,
) -> Optional[int]:
    """
    First round where accuracy reaches peak_ratio * peak_accuracy and remains
    above that target for the next `patience` rounds.
    """
    values = np.asarray(accuracies, dtype=float)
    valid = ~np.isnan(values)
    values = values[valid]
    if rounds is None:
        round_values = np.arange(1, len(valid) + 1)[valid]
    else:
        round_values = np.asarray(rounds, dtype=int)[valid]
    if len(values) == 0:
        return None

    target = float(np.max(values) * peak_ratio)
    window = max(1, int(patience))
    for idx in range(len(values)):
        if values[idx] < target:
            continue
        end = min(idx + window, len(values))
        if np.all(values[idx:end] >= target):
            return int(round_values[idx])
    return None


def reward_ratio(
    honest_rewards: np.ndarray,
    freerider_rewards: np.ndarray
) -> float:
    """RR = mean(honest reward) / mean(free-rider reward). Cao = tốt."""
    mean_fr = np.mean(freerider_rewards)
    return float(np.mean(honest_rewards) / mean_fr) if mean_fr > 1e-10 else float('inf')


def economic_incentive_index(
    r_honest: float,
    r_lazy: float,
    d_honest: float,
    d_lazy: float
) -> float:
    """
    EII = (r_honest - r_lazy) / delta_cost
    delta_cost = d_honest / d_lazy (chi phí tương đối dựa trên data size)
    EII > 1: trung thực về kinh tế là đáng giá.
    """
    delta_cost = d_honest / (d_lazy + 1e-10)
    delta_reward = r_honest - r_lazy
    return float(delta_reward / delta_cost) if delta_cost > 1e-10 else 0.0


# ── Utility ─────────────────────────────────────────────────
def compute_all_group2(
    rewards: np.ndarray,
    contributions: np.ndarray
) -> Dict[str, float]:
    """Tính toàn bộ metric nhóm 2 trong một lần gọi."""
    return {
        "jain":  jain_index(rewards),
        "gini":  gini_coefficient(rewards),
        "crc":   contribution_reward_correlation(contributions, rewards),
        "fg":    fairness_gap(rewards, contributions),
    }
