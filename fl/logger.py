"""
logger.py — Ghi log mỗi vòng ra CSV để tính metric offline.
Một file CSV per experiment run.
"""
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


FIELDS = [
    "run_id", "dataset", "scenario", "config", "alpha", "dirichlet_alpha",
    "alpha_runtime",
    "round", "client_id", "client_type",
    "quality_score", "data_size", "w_new", "reputation",
    "reward_eth", "is_honest",
    "anomaly_score", "robust_z", "is_anomaly", "detection_reason",
    "global_accuracy",
]


class ExperimentLogger:
    def __init__(self, run_id: str, log_dir: str = "./results/logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.run_id   = run_id
        self.filepath = os.path.join(log_dir, f"{run_id}.csv")
        self._file    = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer  = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log_round(
        self,
        dataset: str, scenario: str, config: str, alpha: float,
        round_num: int, client_id: int, client_type: str,
        quality: float, data_size: int, w_new: float,
        reputation: float, reward_eth: float, is_honest: bool,
        global_accuracy: Optional[float] = None,
        dirichlet_alpha: Optional[float] = None,
        alpha_runtime: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        robust_z: Optional[float] = None,
        is_anomaly: Optional[bool] = None,
        detection_reason: str = "",
    ):
        self._writer.writerow({
            "run_id":          self.run_id,
            "dataset":         dataset,
            "scenario":        scenario,
            "config":          config,
            "alpha":           alpha,
            "dirichlet_alpha":  dirichlet_alpha if dirichlet_alpha is not None else "",
            "alpha_runtime":    round(alpha_runtime, 6) if alpha_runtime is not None else "",
            "round":           round_num,
            "client_id":       client_id,
            "client_type":     client_type,
            "quality_score":   round(quality, 6),
            "data_size":       data_size,
            "w_new":           round(w_new, 6),
            "reputation":      round(reputation, 6),
            "reward_eth":      round(reward_eth, 8),
            "is_honest":       int(is_honest),
            "anomaly_score":    round(anomaly_score, 6) if anomaly_score is not None else "",
            "robust_z":         round(robust_z, 6) if robust_z is not None else "",
            "is_anomaly":       int(is_anomaly) if is_anomaly is not None else "",
            "detection_reason": detection_reason,
            "global_accuracy": round(global_accuracy, 4) if global_accuracy is not None else ""
        })
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def make_run_id(
    dataset: str,
    scenario: str,
    config: str,
    alpha: float,
    dirichlet_alpha: Optional[float] = None,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if scenario == "K3" and dirichlet_alpha is not None:
        suffix = f"_da{int(round(dirichlet_alpha * 100)):03d}"
    return f"{dataset}_{scenario}_{config}_a{int(alpha*10):02d}{suffix}_{ts}"
