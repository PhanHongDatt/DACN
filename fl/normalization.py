"""
normalization.py — Hybrid Normalization và tính W_new.
Giải quyết LH1: chuẩn hóa suy biến khi mọi client có cùng data size (IID).
"""
import numpy as np
from fl.config import ContributionConfig


def hybrid_normalize(
    values: np.ndarray,
    mean_val: float = None,
    cfg: ContributionConfig = None
) -> np.ndarray:
    """
    Hybrid Normalization:
      - Nếu (max - min) >= tau: dùng Min-Max + epsilon smoothing
      - Ngược lại: fallback sang Standard Scale (kế thừa FedCCEA)

    Args:
        values:   array 1D các giá trị cần chuẩn hóa
        mean_val: giá trị trung bình hệ thống (|D̄|) dùng cho fallback
        cfg:      ContributionConfig (tau, epsilon)
    Returns:
        array đã chuẩn hóa về [0, 1]
    """
    if cfg is None:
        cfg = ContributionConfig()

    tau, eps = cfg.tau, cfg.epsilon
    v_min, v_max = float(values.min()), float(values.max())

    if (v_max - v_min) >= tau:
        # Trường hợp bình thường: Min-Max với epsilon smoothing
        return (values - v_min) / (v_max - v_min + eps)
    else:
        # Fallback Standard Scale (FedCCEA style)
        denom = mean_val if (mean_val and mean_val > eps) else values.mean()
        if denom < eps:
            return np.ones_like(values, dtype=float)
        return np.clip(values.astype(float) / denom, 0.0, 1.0)


def compute_w_new(
    quality_scores: np.ndarray,
    data_sizes: np.ndarray,
    alpha: float = 0.5,
    mean_data_size: float = None,
    cfg: ContributionConfig = None
) -> np.ndarray:
    """
    Tính W_new đa chiều:
        W_new_i = (alpha * q̂_i + (1-alpha) * d̂_i) / Σ(...)
    """
    if cfg is None:
        cfg = ContributionConfig()

    q_hat = hybrid_normalize(np.array(quality_scores, dtype=float), cfg=cfg)
    d_hat = hybrid_normalize(np.array(data_sizes, dtype=float), mean_val=mean_data_size, cfg=cfg)

    composite = alpha * q_hat + (1.0 - alpha) * d_hat
    total = composite.sum()

    if total < cfg.epsilon:
        return np.ones(len(quality_scores)) / len(quality_scores)

    return composite / total
