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
    # Experiment condition
    "attack_type", "seed",
    # Round-level
    "round", "client_id", "client_type",
    "quality_score", "data_size",
    "w_new", "reputation",
    "reward_eth", "is_honest",
    # CSRA-DCD detection signals
    "anomaly_score", "robust_z", "is_anomaly", "detection_reason",
    # Per-round target
    "global_accuracy",
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
        # Optional with defaults
        beta: float = 0.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        attack_type: str = "none",
        seed: int = 0,
        dirichlet_alpha: Optional[float] = None,
        global_accuracy: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        robust_z: Optional[float] = None,
        is_anomaly: Optional[bool] = None,
        detection_reason: str = "",
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
            "attack_type": attack_type,
            "seed": seed,
            "round": round_num,
            "client_id": client_id,
            "client_type": client_type,
            "quality_score": round(quality, 6),
            "data_size": data_size,
            "w_new": round(w_new, 6),
            "reputation": round(reputation, 6),
            "reward_eth": round(reward_eth, 8),
            "is_honest": int(is_honest),
            "anomaly_score": round(anomaly_score, 6) if anomaly_score is not None else "",
            "robust_z": round(robust_z, 6) if robust_z is not None else "",
            "is_anomaly": int(is_anomaly) if is_anomaly is not None else "",
            "detection_reason": detection_reason,
            "global_accuracy": round(global_accuracy, 4) if global_accuracy is not None else "",
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
      - attack   : clean / free_rider / lazy / label_noise / sign_flip

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
