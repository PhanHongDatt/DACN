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
    mad_threshold: float = 3.0,
    min_honest_ratio: float = 0.5,
    mad_epsilon: float = 1e-8,
) -> tuple[ParameterLayers, list[bool], list[float], str]:
    """
    CSRA-DCD aggregation: phát hiện anomaly bằng MAD, loại các client bị flag,
    rồi FedAvg trên phần còn lại.

    Failsafe: nếu số client còn lại sau filter < min_honest_ratio * n,
    fallback chấp nhận tất cả (tránh trường hợp filter quá mạnh phá round).

    Args:
        client_params: list client × list layer
        weights: trọng số FedAvg (thường là num_examples)
        anomaly_scores: ||Δᵢ||₂ của mỗi client
        mad_threshold: ngưỡng z để flag anomaly
        min_honest_ratio: tỷ lệ tối thiểu client còn lại sau filter
        mad_epsilon: dung sai MAD

    Returns:
        (aggregated_params, is_anomaly_mask, robust_z_values, detection_method)
        is_anomaly_mask[i] = True nếu client i bị filter (và do đó nhận reward = 0).
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

    # Failsafe: nếu filter quá mạnh, accept all
    n_honest = sum(1 for a in is_anomaly if not a)
    min_honest = max(1, math.ceil(n * min_honest_ratio))
    if n_honest < min_honest:
        log.warning(
            "csra_dcd: too few honest clients after filter (%d/%d) → "
            "fallback accept all this round",
            n_honest, n,
        )
        is_anomaly = [False] * n
        method = f"{method}+fallback_accept_all"

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

    return aggregated, is_anomaly, robust_z, method


# ─────────────────────────────────────────────────────────────────────────────
# Registry & dispatcher
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATION_NAMES = ("fedavg", "trimmed", "csra_dcd")


def apply_aggregation(
    method: str,
    client_params: Sequence[ParameterLayers],
    weights: Sequence[float],
    *,
    anomaly_scores: Sequence[float] | None = None,
    trim_ratio: float = 0.1,
    mad_threshold: float = 3.0,
    min_honest_ratio: float = 0.5,
    mad_epsilon: float = 1e-8,
) -> tuple[ParameterLayers, dict]:
    """
    Dispatcher cho server. Trả về (aggregated_params, metadata).

    metadata['anomaly_mask']: list[bool] cùng độ dài n_clients.
                              True = bị filter (nhận reward = 0).
    metadata['robust_z']:     list[float] | None (chỉ với csra_dcd)
    metadata['detection_method']: str | None (chỉ với csra_dcd)

    Args:
        method: "fedavg" | "trimmed" | "csra_dcd"

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
        }

    if method == "trimmed":
        params = trimmed_mean_aggregation(client_params, trim_ratio=trim_ratio)
        return params, {
            "anomaly_mask": [False] * n,  # TrimmedMean trim coordinate-wise, không client-wise
            "robust_z": None,
            "detection_method": None,
        }

    # method == "csra_dcd"
    if anomaly_scores is None:
        raise ValueError("anomaly_scores required for method='csra_dcd'")
    params, mask, z, det_method = csra_dcd_aggregation(
        client_params=client_params,
        weights=weights,
        anomaly_scores=anomaly_scores,
        mad_threshold=mad_threshold,
        min_honest_ratio=min_honest_ratio,
        mad_epsilon=mad_epsilon,
    )
    return params, {
        "anomaly_mask": mask,
        "robust_z": z,
        "detection_method": det_method,
    }
