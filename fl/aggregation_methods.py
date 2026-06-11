"""
aggregation_methods.py — 3 aggregation strategies cho ablation study.

Mỗi hàm aggregation nhận parameters của các client và trả về parameters đã hợp nhất
cùng với metadata (anomaly_mask cho CSRA-DCD). Tách logic aggregation khỏi Flower
strategy class để có thể unit-test độc lập.

Reference: docs/PLAN.md §3.1, §7.3.

Note design (so với code cũ):
  - CSRA-DCD chỉ làm filtering (MAD robust z-score), KHÔNG làm reputation-weighted
    aggregation. Reputation chỉ ảnh hưởng tới REWARD (thông qua δ trong CSRAReward).
    Quyết định này giúp ablation matrix sạch hơn (M5 = filter alone, M6 = filter + reward).
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

ParameterLayers = list[np.ndarray]  # list of layer arrays (one client)
ClientParameters = list[ParameterLayers]  # outer list = clients

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float] | None = None,
) -> None:
    if len(client_params) == 0:
        raise ValueError("client_params must not be empty")

    n_layers = len(client_params[0])
    for i, params in enumerate(client_params):
        if len(params) != n_layers:
            raise ValueError(
                f"client {i} has {len(params)} layers, expected {n_layers}"
            )

    if weights is not None:
        if len(weights) != len(client_params):
            raise ValueError(
                f"weights length {len(weights)} != n_clients {len(client_params)}"
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {list(weights)}")


def _weighted_average(
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float],
) -> ParameterLayers:
    """Trung bình có trọng số coordinate-wise theo từng layer."""
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total < _EPS:
        # Fallback: trung bình thuần
        log.warning("weighted_average: all weights ~ 0, using uniform mean")
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / total

    n_layers = len(client_params[0])
    result: ParameterLayers = []
    for layer_idx in range(n_layers):
        # Stack: shape (n_clients, *layer_shape)
        stacked = np.stack(
            [params[layer_idx] for params in client_params], axis=0
        )
        # Reshape weights: (n_clients, 1, 1, ..., 1) để broadcast
        broadcast_shape = (len(client_params),) + (1,) * (stacked.ndim - 1)
        w_broadcast = w.reshape(broadcast_shape)
        agg = np.sum(stacked * w_broadcast, axis=0)
        result.append(agg.astype(client_params[0][layer_idx].dtype))
    return result


def _flatten_delta(params: ParameterLayers, base_params: ParameterLayers) -> np.ndarray:
    """Flatten one client update delta into a single vector."""
    if len(params) != len(base_params):
        raise ValueError(
            f"params has {len(params)} layers, base_params has {len(base_params)}"
        )
    return np.concatenate([
        (
            np.asarray(local, dtype=np.float64).ravel()
            - np.asarray(base, dtype=np.float64).ravel()
        )
        for local, base in zip(params, base_params)
    ])


def compute_update_features(
    client_params: Sequence[ParameterLayers],
    base_params: ParameterLayers,
) -> dict:
    """
    Compute server-side update features from received parameters.

    Features:
      - update_norms: ||local_params_i - global_params||_2
      - update_stds: Standard deviation của vector delta_i (STD-DAGMM logic)
      - cosine_to_reference: cosine(delta_i, median_delta)

    The reference delta is coordinate-wise median across participating clients.
    This is more robust than mean when a minority sends sign-flipped updates.
    """
    _validate_inputs(client_params)
    n_layers = len(client_params[0])
    if len(base_params) != n_layers:
        raise ValueError(
            f"base_params has {len(base_params)} layers, expected {n_layers}"
        )

    deltas = [_flatten_delta(params, base_params) for params in client_params]
    if not deltas:
        return {"update_norms": [], "cosine_to_reference": []}

    delta_matrix = np.stack(deltas, axis=0)
    update_norms = np.linalg.norm(delta_matrix, axis=1)
    update_stds = np.std(delta_matrix, axis=1)

    reference = np.median(delta_matrix, axis=0)
    reference_norm = float(np.linalg.norm(reference))
    cosines: list[float] = []
    for delta, norm in zip(delta_matrix, update_norms):
        denom = float(norm) * reference_norm
        if denom < _EPS:
            cosines.append(0.0)
        else:
            cosines.append(float(np.dot(delta, reference) / denom))

    return {
        "update_norms": [float(v) for v in update_norms],
        "update_stds": [float(v) for v in update_stds],
        "cosine_to_reference": cosines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregation(
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float],
) -> ParameterLayers:
    """
    FedAvg chuẩn (McMahan et al. 2017): trung bình có trọng số theo data size.

    Args:
        client_params: list các client, mỗi client là list các layer (np.ndarray)
        weights: trọng số cho mỗi client (thường là num_examples)

    Returns:
        Aggregated parameters (cùng cấu trúc layer như client_params[0]).
    """
    _validate_inputs(client_params, weights)
    return _weighted_average(client_params, weights)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: TrimmedMean
# ─────────────────────────────────────────────────────────────────────────────

def trimmed_mean_aggregation(
    client_params: Sequence[ParameterLayers],
    trim_ratio: float = 0.1,
) -> ParameterLayers:
    """
    Coordinate-wise trimmed mean (Yin et al. 2018): robust Byzantine-tolerant
    aggregation. Bỏ k client cao nhất và k client thấp nhất ở MỖI coordinate
    rồi lấy trung bình phần còn lại.

    Args:
        client_params: list client × list layer
        trim_ratio: tỷ lệ trim mỗi đầu, ∈ [0, 0.5)

    Returns:
        Aggregated parameters.
    """
    _validate_inputs(client_params)
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")

    n_clients = len(client_params)
    trim_k = int(n_clients * trim_ratio)
    if trim_k * 2 >= n_clients:
        log.warning(
            "trim_ratio %.3f too high for %d clients → fallback to plain mean",
            trim_ratio, n_clients,
        )
        trim_k = 0

    n_layers = len(client_params[0])
    result: ParameterLayers = []
    for layer_idx in range(n_layers):
        stacked = np.stack(
            [params[layer_idx] for params in client_params], axis=0
        )
        if trim_k > 0:
            sorted_layer = np.sort(stacked, axis=0)
            selected = sorted_layer[trim_k : n_clients - trim_k]
        else:
            selected = stacked
        agg = np.mean(selected, axis=0)
        result.append(agg.astype(client_params[0][layer_idx].dtype))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: CSRA-DCD — Robust z-score filtering + FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def mad_robust_zscore(
    scores: np.ndarray,
    mad_epsilon: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """
    Two-sided robust z-score dùng Median Absolute Deviation (MAD) làm scale.

    z_i = |score_i - median| / (1.4826 · MAD)

    Khi MAD ~ 0 (mọi score gần như bằng nhau), fallback sang mean absolute
    deviation để tránh chia 0.

    Args:
        scores: 1-D array of anomaly scores (e.g., ||Δᵢ||₂)
        mad_epsilon: ngưỡng coi MAD là 0

    Returns:
        (z_scores, detection_method) — z_scores cùng shape với scores,
        detection_method ∈ {"mad", "mean_abs_dev_fallback", "mad_zero_all_equal", "empty"}.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=float), "empty"

    median = float(np.median(scores))
    abs_dev = np.abs(scores - median)
    mad = float(np.median(abs_dev))

    if mad >= mad_epsilon:
        return abs_dev / (1.4826 * mad), "mad"

    max_dev = float(np.max(abs_dev)) if abs_dev.size else 0.0
    if max_dev < mad_epsilon:
        return np.zeros_like(scores, dtype=float), "mad_zero_all_equal"

    fallback_scale = max(float(np.mean(abs_dev)), mad_epsilon)
    return abs_dev / (1.4826 * fallback_scale), "mean_abs_dev_fallback"


def detect_anomalies(
    anomaly_scores: Sequence[float],
    mad_threshold: float = 3.0,
    mad_epsilon: float = 1e-8,
) -> tuple[list[bool], list[float], str]:
    """
    Phát hiện client bất thường bằng MAD robust z-score.

    Args:
        anomaly_scores: scores của các client (e.g., ||Δᵢ||₂)
        mad_threshold: ngưỡng z để gắn cờ anomaly
        mad_epsilon: dung sai cho MAD ~ 0

    Returns:
        (is_anomaly_mask, robust_z_values, detection_method)
    """
    scores_arr = np.asarray(anomaly_scores, dtype=float)
    robust_z, method = mad_robust_zscore(scores_arr, mad_epsilon=mad_epsilon)
    if robust_z.size == 0:
        return [], [], method
    is_anomaly = [bool(z > mad_threshold) for z in robust_z]
    return is_anomaly, [float(z) for z in robust_z], method


def csra_dcd_aggregation(
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float],
    anomaly_scores: Sequence[float],
    update_cosines: Sequence[float] | None = None,
    mad_threshold: float = 3.0,
    cosine_threshold: float = -0.8,
    direction_min_norm_z: float = 0.0,
    min_honest_ratio: float = 0.5,
    fallback_hard_z: float = 6.0,
    mad_epsilon: float = 1e-8,
) -> tuple[ParameterLayers, list[bool], list[float], str, dict]:
    """
    CSRA-DCD aggregation: phát hiện anomaly bằng MAD/hướng update, loại các
    client bị flag khỏi aggregation, rồi FedAvg trên phần còn lại.

    Failsafe: nếu số client còn lại sau filter < min_honest_ratio * n, fallback
    chấp nhận tất cả cho aggregation. Reward vẫn có thể bị chặn bởi
    reward_block_mask với tín hiệu direction anomaly có độ tin cậy cao.

    Args:
        client_params: list client × list layer
        weights: trọng số FedAvg (thường là num_examples)
        anomaly_scores: ||Δᵢ||₂ của mỗi client
        update_cosines: cosine(delta_i, robust_reference_delta), optional
        mad_threshold: ngưỡng z để flag anomaly
        cosine_threshold: ngưỡng hướng update để flag sign-flip rõ ràng
        direction_min_norm_z: chỉ flag hướng nếu norm z ít nhất mức này
        min_honest_ratio: tỷ lệ tối thiểu client còn lại sau filter
        fallback_hard_z: khi failsafe kích hoạt, vẫn filter/reward-block norm
            outlier nếu robust_z >= ngưỡng này. Đặt <= 0 để tắt soft hard-filter.
        mad_epsilon: dung sai MAD

    Returns:
        (aggregated_params, is_anomaly_mask, robust_z_values, detection_method, detection_meta)
        is_anomaly_mask[i] = True nếu client i bị filter khỏi aggregation.
        detection_meta["reward_block_mask"][i] = True nếu client i không đủ
        điều kiện nhận reward trong round này.
    """
    _validate_inputs(client_params, weights)
    if len(anomaly_scores) != len(client_params):
        raise ValueError(
            f"anomaly_scores length {len(anomaly_scores)} != "
            f"n_clients {len(client_params)}"
        )

    n = len(client_params)
    is_anomaly, robust_z, method = detect_anomalies(
        anomaly_scores, mad_threshold=mad_threshold, mad_epsilon=mad_epsilon,
    )

    direction_mask = [False] * n
    direction_risk = [0.0] * n
    cosines: list[float | None] = [None] * n
    if update_cosines is not None:
        if len(update_cosines) != n:
            raise ValueError(
                f"update_cosines length {len(update_cosines)} != n_clients {n}"
            )
        cosines = [float(c) for c in update_cosines]
        denom = max(cosine_threshold - (-1.0), _EPS)
        for i, cosine in enumerate(cosines):
            if cosine < cosine_threshold:
                direction_risk[i] = min(1.0, max(0.0, (cosine_threshold - cosine) / denom))
            direction_mask[i] = (
                bool(cosine < cosine_threshold)
                and bool(robust_z[i] >= direction_min_norm_z)
            )

    norm_mask = list(is_anomaly)
    is_anomaly = [bool(nm or dm) for nm, dm in zip(norm_mask, direction_mask)]
    detection_reasons: list[str] = []
    risk_scores: list[float] = []
    for i in range(n):
        reasons = []
        if norm_mask[i]:
            reasons.append("norm_mad")
        if direction_mask[i]:
            reasons.append("direction_cosine")
        detection_reasons.append("+".join(reasons) if reasons else "accepted")
        norm_risk = float(robust_z[i] / mad_threshold) if mad_threshold > 0 else 0.0
        risk_scores.append(float(max(norm_risk, direction_risk[i])))

    pre_fallback_anomaly_mask = list(is_anomaly)
    fallback_accept_all = False
    fallback_triggered = False
    fallback_soft_filter = False
    fallback_hard_norm_mask = [False] * n

    # Failsafe: nếu filter quá mạnh, accept all for aggregation.
    # Soft hard-filter: nếu có norm outlier cực mạnh, vẫn loại outlier đó.
    # Điều này giảm lỗ hổng "fallback accept all" nhưng tránh dùng ngưỡng thường
    # để hard-filter honest clients trong Non-IID mạnh.
    n_honest = sum(1 for a in is_anomaly if not a)
    min_honest = max(1, math.ceil(n * min_honest_ratio))
    if n_honest < min_honest:
        fallback_triggered = True
        log.warning(
            "csra_dcd: too few honest clients after filter (%d/%d) → "
            "failsafe this round",
            n_honest,
            n,
        )
        hard_z = float(fallback_hard_z)
        if hard_z > 0.0:
            fallback_hard_norm_mask = [
                bool(norm_mask[i] and robust_z[i] >= hard_z)
                for i in range(n)
            ]

        if any(fallback_hard_norm_mask):
            fallback_soft_filter = True
            is_anomaly = list(fallback_hard_norm_mask)
            method = f"{method}+fallback_soft_filter"
            detection_reasons = [
                "norm_mad+fallback_hard_block"
                if fallback_hard_norm_mask[i]
                else "fallback_soft_accept"
                for i in range(n)
            ]
        else:
            fallback_accept_all = True
            is_anomaly = [False] * n
            method = f"{method}+fallback_accept_all"
            detection_reasons = ["fallback_accept_all"] * n

    if fallback_triggered:
        reward_block_mask = [
            bool(direction_mask[i] or fallback_hard_norm_mask[i])
            for i in range(n)
        ]
    else:
        reward_block_mask = list(is_anomaly)

    # Filter và aggregate
    honest_params = [
        params for params, anom in zip(client_params, is_anomaly) if not anom
    ]
    honest_weights = [
        w for w, anom in zip(weights, is_anomaly) if not anom
    ]

    if not honest_params:
        # Không có ai để aggregate → trả về parameters của round trước (caller xử lý)
        # Ở đây trả về aggregate uniform để tránh crash
        log.error("csra_dcd: no honest clients at all, returning uniform mean")
        aggregated = _weighted_average(client_params, [1.0] * n)
    else:
        aggregated = _weighted_average(honest_params, honest_weights)

    return aggregated, is_anomaly, robust_z, method, {
        "norm_anomaly_mask": norm_mask,
        "direction_anomaly_mask": direction_mask,
        "pre_fallback_anomaly_mask": pre_fallback_anomaly_mask,
        "fallback_triggered": fallback_triggered,
        "fallback_accept_all": fallback_accept_all,
        "fallback_soft_filter": fallback_soft_filter,
        "fallback_hard_norm_mask": fallback_hard_norm_mask,
        "fallback_hard_z": float(fallback_hard_z),
        "reward_block_mask": reward_block_mask,
        "cosine_to_reference": cosines,
        "risk_score": risk_scores,
        "detection_reasons": detection_reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: FedLAW — Learnable Aggregation Weights (Byzantine Robust)
# ─────────────────────────────────────────────────────────────────────────────

def project_capped_simplex(h: np.ndarray, s: int, t: float) -> np.ndarray:
    """
    Toán tử chiếu Simplex với ràng buộc thưa s và chặn trên t:
        Proj_Δ+(h) = argmin_w ||w - h||^2
        s.t. sum(w) = 1, 0 <= w_i <= t, ||w||_0 <= s

    Thuật toán:
      1. Chọn s phần tử lớn nhất trong h. Các phần tử khác gán về 0.
      2. Chiếu s phần tử này lên unit-capped simplex (0 <= w_i <= t, sum = 1).
    """
    n = len(h)
    s = min(max(1, s), n)
    t = max(1.0 / s, t)

    # Bước 1: Sparsity constraint (giữ s phần tử lớn nhất)
    indices = np.argsort(h)[::-1]
    top_s_indices = indices[:s]
    w = np.zeros_like(h)
    v = h[top_s_indices]

    # Bước 2: Capped Simplex Projection (v ∈ R^s)
    # Thuật toán lặp/chia nhị phân để tìm ngưỡng λ sao cho sum(clip(v - λ, 0, t)) = 1
    low = np.min(v) - 1.0
    high = np.max(v)
    for _ in range(50):  # Đủ để đạt độ chính xác float
        mid = (low + high) / 2
        current_sum = np.sum(np.clip(v - mid, 0, t))
        if current_sum > 1.0:
            low = mid
        else:
            high = mid

    w[top_s_indices] = np.clip(v - high, 0, t)
    # Đảm bảo tổng chính xác bằng 1 (do sai số float)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    return w


def fedlaw_aggregation(
    client_params: Sequence[ParameterLayers],
    trial_params: Sequence[ParameterLayers],
    base_params: ParameterLayers,
    w_k: np.ndarray,
    local_losses: np.ndarray,
    alpha: float,
    beta_law: float,
    sparsity_s: int,
    capping_t: float,
) -> tuple[ParameterLayers, np.ndarray, np.ndarray, dict]:
    """
    FedLAW (Algorithm 2 trong bài báo):
      1. h_k = w_k + α·β_law·G_k^T·G_tilde·w_k - β_law·f_tilde
      2. w_k+1 = Proj_Δ+(h_k)
      3. θ_k+1 = θ_k - α·Σ w_i·g_i

    Args:
        client_params: G_k (gradients vòng 1)
        trial_params: G_tilde (gradients vòng 2 tại mô hình thử nghiệm)
        base_params: θ_k (mô hình toàn cục đầu epoch)
        w_k: trọng số aggregation hiện tại
        local_losses: f_tilde (losses tại mô hình thử nghiệm)
        alpha: learning rate của mô hình
        beta_law: learning rate của trọng số
        sparsity_s: số lượng client được giữ lại (s)
        capping_t: chặn trên trọng số của 1 client (t)

    Returns:
        (aggregated_params, w_next, alignment_scores, metadata)
    """
    _validate_inputs(client_params)
    _validate_inputs(trial_params)
    n = len(client_params)

    # Flatten gradients: G_k và G_tilde
    G_k = np.stack([_flatten_delta(p, base_params) / alpha for p in client_params], axis=1)  # (d, n)
    G_tilde = np.stack([_flatten_delta(p, base_params) / alpha for p in trial_params], axis=1)  # (d, n)

    # Gradient Alignment: G_k^T * G_tilde * w_k
    # (n, d) * (d, n) * (n, 1) -> (n, 1)
    # Chú ý: G_tilde * w_k là hướng đi chung của trial aggregation
    agg_trial_direction = G_tilde @ w_k
    alignment = G_k.T @ agg_trial_direction

    # Pre-projection score h_k
    # f_tilde cần được scale tương ứng để tránh áp đảo gradient alignment
    h_k = w_k + alpha * beta_law * alignment - beta_law * local_losses

    # Weight update via capped simplex projection
    w_next = project_capped_simplex(h_k, sparsity_s, capping_t)

    # Final aggregation: dùng G_k (gradients vòng 1) với trọng số mới
    # θ_k+1 = θ_k - α * Σ w_i * g_i
    # Tương đương FedAvg trên client_params với weights = w_next
    aggregated = _weighted_average(client_params, w_next)

    # Detection logic: client bị ép về 0 được coi là anomaly
    anomaly_mask = [bool(w < 1e-6) for w in w_next]

    return aggregated, w_next, alignment, {
        "anomaly_mask": anomaly_mask,
        "h_k": h_k,
        "alignment_scores": alignment,
        "w_next": w_next,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry & dispatcher
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATION_NAMES = ("fedavg", "trimmed", "csra_dcd", "fedlaw")


def apply_aggregation(
    method: str,
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float],
    *,
    trial_params: Sequence[ParameterLayers] | None = None,
    base_params: ParameterLayers | None = None,
    w_k: np.ndarray | None = None,
    local_losses: np.ndarray | None = None,
    alpha: float = 0.1,
    beta_law: float = 0.01,
    sparsity_s: int = 10,
    capping_t: float = 0.2,
    anomaly_scores: Sequence[float] | None = None,
    update_cosines: Sequence[float] | None = None,
    trim_ratio: float = 0.1,
    mad_threshold: float = 3.0,
    cosine_threshold: float = -0.8,
    direction_min_norm_z: float = 0.0,
    min_honest_ratio: float = 0.5,
    fallback_hard_z: float = 6.0,
    mad_epsilon: float = 1e-8,
) -> tuple[ParameterLayers, dict]:
    """
    Dispatcher cho server. Trả về (aggregated_params, metadata).

    metadata['anomaly_mask']: list[bool] cùng độ dài n_clients.
                              True = bị filter (nhận reward = 0).
    metadata['robust_z']:     list[float] | None (chỉ với csra_dcd)
    metadata['detection_method']: str | None (chỉ với csra_dcd)

    Args:
        method: "fedavg" | "trimmed" | "csra_dcd" | "fedlaw"

    Raises:
        ValueError: method không hợp lệ hoặc thiếu input cần thiết.
    """
    if method not in AGGREGATION_NAMES:
        raise ValueError(
            f"unknown aggregation method '{method}', "
            f"expected one of {AGGREGATION_NAMES}"
        )

    n = len(client_params)

    if method == "fedavg":
        params = fedavg_aggregation(client_params, weights)
        return params, {
            "anomaly_mask": [False] * n,
            "robust_z": None,
            "detection_method": None,
            "direction_anomaly_mask": [False] * n,
            "pre_fallback_anomaly_mask": [False] * n,
            "fallback_triggered": False,
            "fallback_accept_all": False,
            "fallback_soft_filter": False,
            "fallback_hard_norm_mask": [False] * n,
            "fallback_hard_z": None,
            "reward_block_mask": [False] * n,
            "cosine_to_reference": None,
            "risk_score": None,
            "detection_reasons": ["accepted"] * n,
        }

    if method == "trimmed":
        params = trimmed_mean_aggregation(client_params, trim_ratio=trim_ratio)
        return params, {
            "anomaly_mask": [False] * n,  # TrimmedMean trim coordinate-wise, không client-wise
            "robust_z": None,
            "detection_method": None,
            "direction_anomaly_mask": [False] * n,
            "pre_fallback_anomaly_mask": [False] * n,
            "fallback_triggered": False,
            "fallback_accept_all": False,
            "fallback_soft_filter": False,
            "fallback_hard_norm_mask": [False] * n,
            "fallback_hard_z": None,
            "reward_block_mask": [False] * n,
            "cosine_to_reference": None,
            "risk_score": None,
            "detection_reasons": ["accepted"] * n,
        }

    if method == "fedlaw":
        if trial_params is None or base_params is None or w_k is None or local_losses is None:
            raise ValueError("fedlaw needs trial_params, base_params, w_k, and local_losses")
        params, w_next, alignment, meta = fedlaw_aggregation(
            client_params=client_params,
            trial_params=trial_params,
            base_params=base_params,
            w_k=w_k,
            local_losses=local_losses,
            alpha=alpha,
            beta_law=beta_law,
            sparsity_s=sparsity_s,
            capping_t=capping_t,
        )
        return params, {
            **meta,
            "robust_z": None,
            "detection_method": "fedlaw_projection",
            "reward_block_mask": meta["anomaly_mask"],
            "detection_reasons": [
                "fedlaw_blocked" if a else "accepted" for a in meta["anomaly_mask"]
            ]
        }

    # method == "csra_dcd"
    if anomaly_scores is None:
        raise ValueError("anomaly_scores required for method='csra_dcd'")
    params, mask, z, det_method, det_meta = csra_dcd_aggregation(
        client_params=client_params,
        weights=weights,
        anomaly_scores=anomaly_scores,
        update_cosines=update_cosines,
        mad_threshold=mad_threshold,
        cosine_threshold=cosine_threshold,
        direction_min_norm_z=direction_min_norm_z,
        min_honest_ratio=min_honest_ratio,
        fallback_hard_z=fallback_hard_z,
        mad_epsilon=mad_epsilon,
    )
    return params, {
        "anomaly_mask": mask,
        "robust_z": z,
        "detection_method": det_method,
        **det_meta,
    }
