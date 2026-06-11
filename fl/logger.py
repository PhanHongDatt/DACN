"""
logger.py — Ghi log mỗi vòng ra CSV để tính metric offline.
Một file CSV per experiment run.

Schema v2 (post-refactor):
  Tách config cũ A/B/C thành (aggregation_method, reward_policy) độc lập.
  Thêm cột beta, gamma, delta (CSRA weights), attack_type, seed.

Reference: docs/PLAN.md §8.
"""
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


FIELDS = [
    # Run metadata
    "run_id", "dataset", "scenario", "dirichlet_alpha",
    # Method specification (replaces old `config`)
    "aggregation_method", "reward_policy",
    "beta", "gamma", "delta",
    "mad_threshold", "cosine_threshold", "direction_min_norm_z",
    "min_honest_ratio", "fallback_hard_z",
    "suspicion_decay", "suspicion_threshold", "low_quality_z_threshold",
    "low_quality_suspicion", "zero_data_suspicion", "anomaly_suspicion",
    "authenticity_suspicion", "low_authenticity_threshold",
    "high_update_norm_z_threshold", "inefficient_update_suspicion",
    # FedLAW metadata
    "alpha_law", "beta_law", "sparsity_s", "capping_t",
    # Experiment condition
    "attack_type", "seed", "persistent_clients",
    "num_clients", "num_rounds", "local_epochs", "batch_size",
    "learning_rate", "client_fraction", "data_split", "data_imbalance",
    # Round-level
    "round", "client_id", "client_type",
    "ground_truth_honest",
    "quality_score", "data_size", "reported_data_size", "server_known_data_size",
    "w_new", "reputation",
    "reward_eth", "reward_blocked", "reward_eligible", "is_honest",
    # CSRA-DCD detection signals
    "anomaly_score", "robust_z", "is_anomaly", "detection_reason",
    "direction_anomaly", "cosine_to_reference", "risk_score", "anomaly_score_source",
    "raw_update_norm", "raw_update_norm_z", "normalized_update_score",
    "data_commitment_anomaly", "data_size_mismatch", "low_quality_outlier",
    "inefficient_update",
    "authenticity_score", "authenticity_anomaly",
    "suspicion_signal", "suspicion_score", "suspicion_quarantine", "suspicion_reason",
    "reward_component_quality", "reward_component_data", "reward_component_reputation",
    # FedLAW signals
    "alignment_score", "simplex_weight",
    # Per-round target
    "global_accuracy", "global_loss",
]


class ExperimentLogger:
    def __init__(self, run_id: str, log_dir: str = "./results/logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.filepath = os.path.join(log_dir, f"{run_id}.csv")
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log_round(
        self,
        # Run metadata
        dataset: str,
        scenario: str,
        aggregation_method: str,
        reward_policy: str,
        # Round-level
        round_num: int,
        client_id: int,
        client_type: str,
        quality: float,
        data_size: int,
        w_new: float,
        reputation: float,
        reward_eth: float,
        is_honest: bool,
        ground_truth_honest: Optional[bool] = None,
        reward_eligible: Optional[bool] = None,
        # Optional with defaults
        beta: float = 0.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        mad_threshold: Optional[float] = None,
        cosine_threshold: Optional[float] = None,
        direction_min_norm_z: Optional[float] = None,
        min_honest_ratio: Optional[float] = None,
        fallback_hard_z: Optional[float] = None,
        suspicion_decay: Optional[float] = None,
        suspicion_threshold: Optional[float] = None,
        low_quality_z_threshold: Optional[float] = None,
        low_quality_suspicion: Optional[float] = None,
        zero_data_suspicion: Optional[float] = None,
        anomaly_suspicion: Optional[float] = None,
        authenticity_suspicion: Optional[float] = None,
        low_authenticity_threshold: Optional[float] = None,
        high_update_norm_z_threshold: Optional[float] = None,
        inefficient_update_suspicion: Optional[float] = None,
        # FedLAW
        alpha_law: Optional[float] = None,
        beta_law: Optional[float] = None,
        sparsity_s: Optional[int] = None,
        capping_t: Optional[float] = None,
        attack_type: str = "none",
        seed: int = 0,
        persistent_clients: bool = False,
        num_clients: Optional[int] = None,
        num_rounds: Optional[int] = None,
        local_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        client_fraction: Optional[float] = None,
        data_split: str = "",
        data_imbalance: str = "",
        dirichlet_alpha: Optional[float] = None,
        global_accuracy: Optional[float] = None,
        global_loss: Optional[float] = None,
        reward_blocked: bool = False,
        anomaly_score: Optional[float] = None,
        robust_z: Optional[float] = None,
        is_anomaly: Optional[bool] = None,
        detection_reason: str = "",
        direction_anomaly: Optional[bool] = None,
        cosine_to_reference: Optional[float] = None,
        risk_score: Optional[float] = None,
        anomaly_score_source: str = "",
        raw_update_norm: Optional[float] = None,
        raw_update_norm_z: Optional[float] = None,
        normalized_update_score: Optional[float] = None,
        reported_data_size: Optional[int] = None,
        server_known_data_size: Optional[int] = None,
        data_commitment_anomaly: Optional[bool] = None,
        data_size_mismatch: Optional[bool] = None,
        low_quality_outlier: Optional[bool] = None,
        inefficient_update: Optional[bool] = None,
        authenticity_score: Optional[float] = None,
        authenticity_anomaly: Optional[bool] = None,
        suspicion_signal: Optional[float] = None,
        suspicion_score: Optional[float] = None,
        suspicion_quarantine: Optional[bool] = None,
        suspicion_reason: str = "",
        reward_component_quality: Optional[float] = None,
        reward_component_data: Optional[float] = None,
        reward_component_reputation: Optional[float] = None,
        # FedLAW signals
        alignment_score: Optional[float] = None,
        simplex_weight: Optional[float] = None,
    ):
        self._writer.writerow({
            "run_id": self.run_id,
            "dataset": dataset,
            "scenario": scenario,
            "dirichlet_alpha": dirichlet_alpha if dirichlet_alpha is not None else "",
            "aggregation_method": aggregation_method,
            "reward_policy": reward_policy,
            "beta": round(beta, 4),
            "gamma": round(gamma, 4),
            "delta": round(delta, 4),
            "mad_threshold": (
                round(mad_threshold, 4) if mad_threshold is not None else ""
            ),
            "cosine_threshold": (
                round(cosine_threshold, 4) if cosine_threshold is not None else ""
            ),
            "direction_min_norm_z": (
                round(direction_min_norm_z, 4)
                if direction_min_norm_z is not None else ""
            ),
            "min_honest_ratio": (
                round(min_honest_ratio, 4) if min_honest_ratio is not None else ""
            ),
            "fallback_hard_z": (
                round(fallback_hard_z, 4) if fallback_hard_z is not None else ""
            ),
            "suspicion_decay": (
                round(suspicion_decay, 4) if suspicion_decay is not None else ""
            ),
            "suspicion_threshold": (
                round(suspicion_threshold, 4)
                if suspicion_threshold is not None else ""
            ),
            "low_quality_z_threshold": (
                round(low_quality_z_threshold, 4)
                if low_quality_z_threshold is not None else ""
            ),
            "low_quality_suspicion": (
                round(low_quality_suspicion, 4)
                if low_quality_suspicion is not None else ""
            ),
            "zero_data_suspicion": (
                round(zero_data_suspicion, 4)
                if zero_data_suspicion is not None else ""
            ),
            "anomaly_suspicion": (
                round(anomaly_suspicion, 4)
                if anomaly_suspicion is not None else ""
            ),
            "authenticity_suspicion": (
                round(authenticity_suspicion, 4)
                if authenticity_suspicion is not None else ""
            ),
            "low_authenticity_threshold": (
                round(low_authenticity_threshold, 4)
                if low_authenticity_threshold is not None else ""
            ),
            "high_update_norm_z_threshold": (
                round(high_update_norm_z_threshold, 4)
                if high_update_norm_z_threshold is not None else ""
            ),
            "inefficient_update_suspicion": (
                round(inefficient_update_suspicion, 4)
                if inefficient_update_suspicion is not None else ""
            ),
            "alpha_law": round(alpha_law, 4) if alpha_law is not None else "",
            "beta_law": round(beta_law, 4) if beta_law is not None else "",
            "sparsity_s": sparsity_s if sparsity_s is not None else "",
            "capping_t": round(capping_t, 4) if capping_t is not None else "",
            "attack_type": attack_type,
            "seed": seed,
            "persistent_clients": int(bool(persistent_clients)),
            "num_clients": num_clients if num_clients is not None else "",
            "num_rounds": num_rounds if num_rounds is not None else "",
            "local_epochs": local_epochs if local_epochs is not None else "",
            "batch_size": batch_size if batch_size is not None else "",
            "learning_rate": (
                round(learning_rate, 8) if learning_rate is not None else ""
            ),
            "client_fraction": (
                round(client_fraction, 6) if client_fraction is not None else ""
            ),
            "data_split": data_split,
            "data_imbalance": data_imbalance,
            "round": round_num,
            "client_id": client_id,
            "client_type": client_type,
            "ground_truth_honest": int(
                ground_truth_honest
                if ground_truth_honest is not None
                else str(client_type) == "honest"
            ),
            "quality_score": round(quality, 6),
            "data_size": data_size,
            "reported_data_size": (
                reported_data_size if reported_data_size is not None else ""
            ),
            "server_known_data_size": (
                server_known_data_size if server_known_data_size is not None else ""
            ),
            "w_new": round(w_new, 6),
            "reputation": round(reputation, 6),
            "reward_eth": round(reward_eth, 8),
            "reward_blocked": int(reward_blocked),
            "reward_eligible": int(
                reward_eligible if reward_eligible is not None else is_honest
            ),
            "is_honest": int(is_honest),
            "anomaly_score": round(anomaly_score, 6) if anomaly_score is not None else "",
            "robust_z": round(robust_z, 6) if robust_z is not None else "",
            "is_anomaly": int(is_anomaly) if is_anomaly is not None else "",
            "detection_reason": detection_reason,
            "direction_anomaly": (
                int(direction_anomaly) if direction_anomaly is not None else ""
            ),
            "cosine_to_reference": (
                round(cosine_to_reference, 6)
                if cosine_to_reference is not None else ""
            ),
            "risk_score": round(risk_score, 6) if risk_score is not None else "",
            "anomaly_score_source": anomaly_score_source,
            "raw_update_norm": (
                round(raw_update_norm, 6) if raw_update_norm is not None else ""
            ),
            "raw_update_norm_z": (
                round(raw_update_norm_z, 6) if raw_update_norm_z is not None else ""
            ),
            "normalized_update_score": (
                round(normalized_update_score, 6)
                if normalized_update_score is not None else ""
            ),
            "data_commitment_anomaly": (
                int(data_commitment_anomaly)
                if data_commitment_anomaly is not None else ""
            ),
            "data_size_mismatch": (
                int(data_size_mismatch) if data_size_mismatch is not None else ""
            ),
            "low_quality_outlier": (
                int(low_quality_outlier) if low_quality_outlier is not None else ""
            ),
            "inefficient_update": (
                int(inefficient_update) if inefficient_update is not None else ""
            ),
            "authenticity_score": round(authenticity_score, 6) if authenticity_score is not None else "",
            "authenticity_anomaly": (
                int(authenticity_anomaly) if authenticity_anomaly is not None else ""
            ),
            "suspicion_signal": (
                round(suspicion_signal, 6) if suspicion_signal is not None else ""
            ),
            "suspicion_score": (
                round(suspicion_score, 6) if suspicion_score is not None else ""
            ),
            "suspicion_quarantine": (
                int(suspicion_quarantine)
                if suspicion_quarantine is not None else ""
            ),
            "suspicion_reason": suspicion_reason,
            "reward_component_quality": (
                round(reward_component_quality, 6)
                if reward_component_quality is not None else ""
            ),
            "reward_component_data": (
                round(reward_component_data, 6)
                if reward_component_data is not None else ""
            ),
            "reward_component_reputation": (
                round(reward_component_reputation, 6)
                if reward_component_reputation is not None else ""
            ),
            "alignment_score": round(alignment_score, 6) if alignment_score is not None else "",
            "simplex_weight": round(simplex_weight, 6) if simplex_weight is not None else "",
            "global_accuracy": round(global_accuracy, 4) if global_accuracy is not None else "",
            "global_loss": round(global_loss, 6) if global_loss is not None else "",
        })
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def make_run_id(
    dataset: str,
    scenario: str,
    aggregation_method: str,
    reward_policy: str,
    seed: int,
    beta: float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
    attack_type: str = "clean",
    dirichlet_alpha: Optional[float] = None,
) -> str:
    """
    Filename schema v2:
        <dataset>_<scenario>[_da<dirichlet>]_<agg>_<reward>_b<β>g<γ>d<δ>_s<seed>_<attack>_<timestamp>.csv

    Quy ước viết tắt (xem docs/PLAN.md §8.1):
      - Aggregation: fedavg / trimmed / csra_dcd
      - Reward    : equal / data / quality / csra
      - β,γ,δ    : 2 chữ số (b50 = β=0.50). Cho reward != csra thì b00g00d00.
      - attack   : clean / free_rider / stealth_free_rider / lazy / label_noise / sign_flip

    Ví dụ:
      mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_143022
      cifar10_K3_da010_csra_dcd_csra_b50g30d20_s2024_free_rider_20260521_091533
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirichlet_part = ""
    if scenario == "K3" and dirichlet_alpha is not None:
        dirichlet_part = f"_da{int(round(dirichlet_alpha * 100)):03d}"

    weights_part = (
        f"b{int(round(beta * 100)):02d}"
        f"g{int(round(gamma * 100)):02d}"
        f"d{int(round(delta * 100)):02d}"
    )
    return (
        f"{dataset}_{scenario}{dirichlet_part}"
        f"_{aggregation_method}_{reward_policy}"
        f"_{weights_part}_s{seed}_{attack_type}_{ts}"
    )
