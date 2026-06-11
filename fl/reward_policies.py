"""
reward_policies.py — 4 reward policies cho ablation study.

Mỗi hàm policy nhận tập client hợp lệ và trả về dict {client_id: reward_eth}.
Semantic: reward chỉ phân phối cho valid_clients (đã qua filtering của aggregation).
Client bị flag sẽ nhận reward = 0 (xử lý ở caller).

Reference: docs/PLAN.md §4 (công thức CSRAReward 3-chiều).
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

from fl.normalization import hybrid_normalize
from fl.config import ContributionConfig

log = logging.getLogger(__name__)

_EPS = 1e-12


def _validate_total(total_reward: float) -> float:
    if total_reward < 0:
        raise ValueError(f"total_reward must be non-negative, got {total_reward}")
    return float(total_reward)


def _validate_ids(client_ids: Sequence[int]) -> list[int]:
    ids = [int(cid) for cid in client_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"client_ids must be unique, got {ids}")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Policy 1: EqualSplit — baseline tối thiểu
# ─────────────────────────────────────────────────────────────────────────────

def equal_split(
    client_ids: Sequence[int],
    total_reward: float,
) -> Dict[int, float]:
    """
    Mỗi valid client nhận R/n. Baseline đơn giản nhất — không nhìn vào quality
    hay data size. Vai trò: làm baseline trần để các policy khác cải thiện lên.

    Args:
        client_ids: ID các client hợp lệ (đã qua filtering)
        total_reward: tổng reward pool (ETH) cho round này

    Returns:
        {client_id: reward}. sum(rewards) == total_reward.
    """
    ids = _validate_ids(client_ids)
    R = _validate_total(total_reward)

    n = len(ids)
    if n == 0:
        log.warning("equal_split: no valid clients, skipping distribution")
        return {}

    share = R / n
    return {cid: share for cid in ids}


# ─────────────────────────────────────────────────────────────────────────────
# Policy 2: DataSize — bias theo lượng dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

def data_size_reward(
    client_ids: Sequence[int],
    data_sizes: Sequence[int],
    total_reward: float,
) -> Dict[int, float]:
    """
    r_i = R · d_i / Σ d_j. Thiên client nhiều dữ liệu, kể cả khi chất lượng kém.
    Vai trò: thể hiện hạn chế của reward dựa trên định lượng đơn thuần.

    Args:
        client_ids: ID các client hợp lệ
        data_sizes: data_size tương ứng (cùng thứ tự)
        total_reward: tổng pool

    Returns:
        {client_id: reward}. sum(rewards) ≈ total_reward (sai số làm tròn float).
    """
    ids = _validate_ids(client_ids)
    R = _validate_total(total_reward)
    sizes = np.asarray(data_sizes, dtype=float)

    if len(sizes) != len(ids):
        raise ValueError(
            f"data_sizes length {len(sizes)} != client_ids length {len(ids)}"
        )
    if (sizes < 0).any():
        raise ValueError(f"data_sizes must be non-negative, got {sizes.tolist()}")

    if len(ids) == 0:
        return {}

    total_size = float(sizes.sum())
    if total_size < _EPS:
        # Mọi client báo data_size = 0 → fallback EqualSplit
        log.warning("data_size_reward: all data_sizes ~ 0, fallback to equal_split")
        return equal_split(ids, R)

    weights = sizes / total_size
    return {cid: float(R * w) for cid, w in zip(ids, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Policy 3: AlignmentOnly (Thay thế QualityOnly) — bias theo sự đồng thuận gradient
# ─────────────────────────────────────────────────────────────────────────────

def quality_reward(
    client_ids: Sequence[int],
    quality_scores: Sequence[float],
    total_reward: float,
) -> Dict[int, float]:
    """
    (Lưu ý: tham số vẫn tên là quality_scores để tương thích với các module gọi hàm,
     nhưng nội dung thực tế là Alignment Score).
    r_i = R · a_i / Σ a_j. Dựa trên Gradient Alignment (Cosine Similarity hoặc Tích vô hướng).
    Thay thế cho Local Loss (dễ bị trick và bias IID).

    Args:
        client_ids: ID client hợp lệ
        quality_scores: alignment tương ứng (được tính tại Server), cần ≥ 0
        total_reward: tổng pool

    Returns:
        {client_id: reward}. sum(rewards) ≈ total_reward.
    """
    ids = _validate_ids(client_ids)
    R = _validate_total(total_reward)
    alignment = np.asarray(quality_scores, dtype=float)

    if len(alignment) != len(ids):
        raise ValueError(
            f"quality_scores length {len(alignment)} != client_ids length {len(ids)}"
        )

    if len(ids) == 0:
        return {}

    # Clip âm về 0 (các client đi ngược hướng với mô hình chung sẽ không nhận reward)
    alignment = np.clip(alignment, 0.0, None)
    total_a = float(alignment.sum())

    if total_a < _EPS:
        # Mọi client có alignment ~ 0 → fallback EqualSplit
        log.warning("quality_reward: all alignment_scores ~ 0, fallback to equal_split")
        return equal_split(ids, R)

    weights = alignment / total_a
    return {cid: float(R * w) for cid, w in zip(ids, weights)}



# ─────────────────────────────────────────────────────────────────────────────
# Policy 4: CSRAReward — đề xuất chính (3-chiều)
# ─────────────────────────────────────────────────────────────────────────────

def csra_reward(
    client_ids: Sequence[int],
    quality_scores: Sequence[float],
    data_sizes: Sequence[int],
    reputations: Sequence[float],
    total_reward: float,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
    mean_data_size: float | None = None,
    contrib_cfg: ContributionConfig | None = None,
) -> Dict[int, float]:
    """
    Công thức CSRAReward 3-chiều (đóng góp đa chiều):

        W_i = β · q̃_i + γ · d̃_i + δ · ρ̃_i
        r_i = R · W_i / Σ W_j

    với:
        - q̃_i, d̃_i, ρ̃_i là quality (Gradient Alignment) / data size / reputation đã hybrid-normalize
        - β + γ + δ = 1, mỗi tham số ∈ [0, 1]

    Args:
        client_ids: ID client hợp lệ
        quality_scores: Alignment score (cosine) của round, thay thế cho local loss, cần ≥ 0
        data_sizes: số sample local của round
        reputations: reputation tích lũy từ blockchain, ∈ [0, 1]
        total_reward: tổng pool (ETH)
        beta, gamma, delta: trọng số 3 chiều, phải có tổng = 1
        mean_data_size: mean data size hệ thống (dùng cho hybrid_normalize fallback)
        contrib_cfg: config normalization (tau, epsilon)

    Returns:
        {client_id: reward}. sum(rewards) ≈ total_reward.

    Raises:
        ValueError: nếu β+γ+δ ≠ 1 (sai số 1e-6), hoặc input không hợp lệ.
    """
    ids = _validate_ids(client_ids)
    R = _validate_total(total_reward)

    # Validate weights
    if not (0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0 and 0.0 <= delta <= 1.0):
        raise ValueError(
            f"beta, gamma, delta must be in [0, 1], got "
            f"beta={beta}, gamma={gamma}, delta={delta}"
        )
    weight_sum = beta + gamma + delta
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"beta + gamma + delta must equal 1.0, got {weight_sum:.8f} "
            f"(beta={beta}, gamma={gamma}, delta={delta})"
        )

    quality = np.asarray(quality_scores, dtype=float)
    sizes = np.asarray(data_sizes, dtype=float)
    rep = np.asarray(reputations, dtype=float)

    if not (len(quality) == len(sizes) == len(rep) == len(ids)):
        raise ValueError(
            f"input length mismatch: ids={len(ids)}, quality={len(quality)}, "
            f"sizes={len(sizes)}, rep={len(rep)}"
        )

    if len(ids) == 0:
        return {}

    # Clip âm về 0
    quality = np.clip(quality, 0.0, None)
    sizes = np.clip(sizes, 0.0, None)
    rep = np.clip(rep, 0.0, None)

    cfg = contrib_cfg or ContributionConfig()

    # Hybrid-normalize từng chiều (ổn định khi mọi client gần bằng nhau)
    q_hat = hybrid_normalize(quality, cfg=cfg)
    d_hat = hybrid_normalize(sizes, mean_val=mean_data_size, cfg=cfg)
    r_hat = hybrid_normalize(rep, cfg=cfg)

    # Composite weight 3-chiều
    composite = beta * q_hat + gamma * d_hat + delta * r_hat
    total = float(composite.sum())

    if total < _EPS:
        log.warning(
            "csra_reward: composite weights ~ 0 (β=%.2f γ=%.2f δ=%.2f), "
            "fallback to equal_split",
            beta, gamma, delta,
        )
        return equal_split(ids, R)

    weights = composite / total
    return {cid: float(R * w) for cid, w in zip(ids, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Registry & helpers
# ─────────────────────────────────────────────────────────────────────────────

#: Tên policy chuẩn hoá để dùng trong CLI / logger / filename.
POLICY_NAMES = ("equal", "data", "quality", "csra")


def apply_reward_policy(
    policy: str,
    client_ids: Sequence[int],
    total_reward: float,
    *,
    quality_scores: Sequence[float] | None = None,
    data_sizes: Sequence[int] | None = None,
    reputations: Sequence[float] | None = None,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
    mean_data_size: float | None = None,
    contrib_cfg: ContributionConfig | None = None,
) -> Dict[int, float]:
    """
    Dispatch tới hàm policy tương ứng. Dùng trong server.

    Args:
        policy: "equal" | "data" | "quality" | "csra"
        client_ids: valid clients (sau filtering)
        total_reward: pool ETH
        quality_scores, data_sizes, reputations: chỉ cần cho policy tương ứng
        beta, gamma, delta: chỉ cho csra
        mean_data_size, contrib_cfg: chỉ cho csra

    Raises:
        ValueError: policy không hợp lệ hoặc thiếu input cần thiết.
    """
    if policy not in POLICY_NAMES:
        raise ValueError(
            f"unknown reward policy '{policy}', expected one of {POLICY_NAMES}"
        )

    if policy == "equal":
        return equal_split(client_ids, total_reward)

    if policy == "data":
        if data_sizes is None:
            raise ValueError("data_sizes required for policy='data'")
        return data_size_reward(client_ids, data_sizes, total_reward)

    if policy == "quality":
        if quality_scores is None:
            raise ValueError("quality_scores required for policy='quality'")
        return quality_reward(client_ids, quality_scores, total_reward)

    # policy == "csra"
    missing = [
        name for name, val in [
            ("quality_scores", quality_scores),
            ("data_sizes", data_sizes),
            ("reputations", reputations),
        ] if val is None
    ]
    if missing:
        raise ValueError(f"csra reward needs {missing}, got None")
    return csra_reward(
        client_ids=client_ids,
        quality_scores=quality_scores,
        data_sizes=data_sizes,
        reputations=reputations,
        total_reward=total_reward,
        beta=beta,
        gamma=gamma,
        delta=delta,
        mean_data_size=mean_data_size,
        contrib_cfg=contrib_cfg,
    )
