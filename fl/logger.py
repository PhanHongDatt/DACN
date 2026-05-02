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
    "run_id", "dataset", "scenario", "config", "alpha",
    "round", "client_id", "client_type",
    "quality_score", "data_size", "w_new", "reputation",
    "reward_eth", "is_honest", "global_accuracy"
]


class ExperimentLogger:
    def __init__(self, run_id: str, log_dir: str = "./results/logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.run_id   = run_id
        self.filepath = os.path.join(log_dir, f"{run_id}.csv")
        self._file    = open(self.filepath, "w", newline="")
        self._writer  = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log_round(
        self,
        dataset: str, scenario: str, config: str, alpha: float,
        round_num: int, client_id: int, client_type: str,
        quality: float, data_size: int, w_new: float,
        reputation: float, reward_eth: float, is_honest: bool,
        global_accuracy: Optional[float] = None
    ):
        self._writer.writerow({
            "run_id":          self.run_id,
            "dataset":         dataset,
            "scenario":        scenario,
            "config":          config,
            "alpha":           alpha,
            "round":           round_num,
            "client_id":       client_id,
            "client_type":     client_type,
            "quality_score":   round(quality, 6),
            "data_size":       data_size,
            "w_new":           round(w_new, 6),
            "reputation":      round(reputation, 6),
            "reward_eth":      round(reward_eth, 8),
            "is_honest":       int(is_honest),
            "global_accuracy": round(global_accuracy, 4) if global_accuracy else ""
        })
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def make_run_id(dataset: str, scenario: str, config: str, alpha: float) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset}_{scenario}_{config}_a{int(alpha*10):02d}_{ts}"
