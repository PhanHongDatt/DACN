import logging
from typing import Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

SCALE = 1_000_000

class OffChainReputation:
    """
    Giả lập logic tính toán Reputation của Smart Contract (ContributionStore.sol)
    Dùng khi chạy thực nghiệm với cờ `--no-blockchain` để giữ tính toàn vẹn của thuật toán CSRA.
    Sử dụng EWMA (Exponential Weighted Moving Average) với decay = 0.9.
    """
    def __init__(
        self,
        min_warmup: int = 5,
        window_size: int = 10,
        data_weight: float = 0.5,
        quality_weight: float = 0.5,
        rep_decay: float = 0.9,
        rep_threshold: float = 0.3
    ):
        self.min_warmup = min_warmup
        self.window_size = window_size
        self.data_weight = data_weight
        self.quality_weight = quality_weight
        self.rep_decay = rep_decay
        self.rep_threshold = rep_threshold
        
        # client_id -> {"quality": [], "data_commit": [], "rounds_recorded": 0}
        self.clients: Dict[int, dict] = {}

    def _cap_to_scale(self, value: float) -> float:
        """Giới hạn giá trị ở mức 1.0 (tương đương SCALE trong Solidity)."""
        return min(1.0, value)

    def record_contribution(self, client_idx: int, quality: float, data_size: int, mean_data_size: float, round_num: int):
        """Mô phỏng function recordContribution của Contract."""
        if client_idx not in self.clients:
            self.clients[client_idx] = {"quality": [], "data_commit": [], "rounds_recorded": 0}
            
        c = self.clients[client_idx]
        
        # Tiền xử lý giống blockchain.py
        q_capped = np.clip(quality, 0.0, 5.0)  # Trong blockchain.py: np.clip(quality, 0.0, 5.0)
        dc = data_size / mean_data_size if mean_data_size > 0 else 1.0
        dc_capped = np.clip(dc, 0.0, 5.0)
        
        # Cập nhật mảng (giữ tối đa window_size phần tử)
        c["quality"].append(q_capped)
        c["data_commit"].append(dc_capped)
        if len(c["quality"]) > self.window_size:
            c["quality"].pop(0)
            c["data_commit"].pop(0)
            
        c["rounds_recorded"] += 1

    def get_reputation(self, client_idx: int) -> Tuple[float, bool]:
        """Mô phỏng function getReputation của Contract."""
        if client_idx not in self.clients:
            return 0.0, True
            
        c = self.clients[client_idx]
        if c["rounds_recorded"] < self.min_warmup:
            return 0.0, True
            
        weight = 1.0
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Lặp ngược từ lịch sử gần nhất (cuối mảng)
        for q, d in zip(reversed(c["quality"]), reversed(c["data_commit"])):
            q_cap = self._cap_to_scale(q)
            d_cap = self._cap_to_scale(d)
            
            round_score = (d_cap * self.data_weight) + (q_cap * self.quality_weight)
            weighted_sum += weight * round_score
            total_weight += weight
            weight *= self.rep_decay
            
        rep = weighted_sum / total_weight if total_weight > 0 else 0.0
        is_honest = rep >= self.rep_threshold
        return rep, is_honest
