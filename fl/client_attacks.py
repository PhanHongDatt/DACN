"""
client_attacks.py — Attack client classes cho ablation study.

Hierarchy:
    HonestClient (base, train bình thường)
      ├── FreeRiderClient   — không train, gửi global params (+ noise nhỏ)
      ├── LazyClient        — train với ít epoch/data hơn
      ├── LabelNoiseClient  — flip label trước khi train (đầu vào nhiễu)
      └── SignFlipClient    — train xong rồi đảo dấu delta (Byzantine)

Tất cả đều báo về anomaly_score = ||local_params - global_params||_2 (L2 norm)
để server CSRA-DCD có thể phát hiện.

Reference: docs/PLAN.md §9.4 (W3 — Attacks + Blockchain audit-only).
"""
from __future__ import annotations

import logging

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl.config import FLConfig
from fl.models import get_model, get_parameters, set_parameters

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base honest client
# ─────────────────────────────────────────────────────────────────────────────

class HonestClient(fl.client.NumPyClient):
    """
    Client trung thực: train đầy đủ local epochs, báo Δloss thực, gửi anomaly_score
    là L2 norm của delta. Đây là baseline cho mọi class attack kế thừa.
    """

    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        fl_cfg: FLConfig | None = None,
        client_type: str = "honest",
    ):
        self.client_id = client_id
        self.dataset = dataset_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = fl_cfg or FLConfig()
        self.client_type = client_type
        self.model = get_model(dataset_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.data_size = len(train_loader.dataset)

    # ── Flower API ────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        global_params = [np.array(p, copy=True) for p in parameters]
        set_parameters(self.model, parameters)

        loss_before = self._eval_loss()
        self._local_train()
        loss_after = self._eval_loss()

        quality = float(max(0.0, loss_before - loss_after))
        current_params = get_parameters(self.model)
        anomaly_score = self._delta_norm(current_params, global_params)

        return (
            current_params,
            self.data_size,
            self._fit_metrics(quality, self.data_size, anomaly_score),
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self._eval_accuracy()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    # ── Training & helpers (overridable) ─────────────────────────────────

    def _local_train(self) -> None:
        """Override để thay đổi training loop (lazy = ít epoch hơn)."""
        epochs = self.cfg.local_epochs
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for X, y in self.train_loader:
                if X.size(0) < 2:
                    continue
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()

    def _fit_metrics(self, quality: float, data_size: int, anomaly_score: float) -> dict:
        return {
            "quality_score": float(quality),
            "data_size": int(data_size),
            "anomaly_score": float(anomaly_score),
            "update_norm": float(anomaly_score),
            "variance": float(anomaly_score),  # backward compat
            "client_type": self.client_type,
        }

    # ── Numerical helpers ────────────────────────────────────────────────

    @staticmethod
    def _delta_norm(local_params, global_params) -> float:
        """L2 norm của model update delta. Dùng float64 để chống underflow."""
        deltas = [
            (np.asarray(local, dtype=np.float64).ravel()
             - np.asarray(base, dtype=np.float64).ravel())
            for local, base in zip(local_params, global_params)
        ]
        if not deltas:
            return 0.0
        return float(np.linalg.norm(np.concatenate(deltas)))

    def _eval_loss(self) -> float:
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                total_loss += criterion(self.model(X), y).item()
        return total_loss / max(len(self.train_loader), 1)

    def _eval_accuracy(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                total_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return total_loss / max(len(self.test_loader), 1), correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Attack 1: FreeRider — không train, gửi noise
# ─────────────────────────────────────────────────────────────────────────────

class FreeRiderClient(HonestClient):
    """
    Free-rider: không train. Gửi global params + nhiễu Gaussian rất nhỏ.
    Báo quality = 0, data_size = 0 (commitment thấp).

    Anomaly_score gần bằng 0 (do delta ≈ noise rất nhỏ) — đây là điểm KHÓ
    của free-rider: CSRA-DCD dùng MAD không thấy outlier rõ rệt, nhưng có thể
    phát hiện qua quality_score thấp + reputation suy giảm.
    """

    def __init__(self, *args, noise_std: float = 0.05, mode: str = "noise", **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = float(noise_std)
        self.mode = mode  # "noise" hoặc "copy"
        self.client_type = "free_rider"

    def fit(self, parameters, config):
        global_params = [np.array(p, copy=True) for p in parameters]

        if self.mode == "copy":
            updated = [np.array(p, copy=True) for p in global_params]
        else:
            updated = [
                p + np.random.normal(0, self.noise_std, p.shape).astype(p.dtype)
                for p in global_params
            ]

        anomaly_score = self._delta_norm(updated, global_params)
        return (
            updated,
            self.data_size,
            self._fit_metrics(
                quality=0.0,
                data_size=0,  # commitment = 0
                anomaly_score=anomaly_score,
            ),
        )


class StealthFreeRiderClient(FreeRiderClient):
    """
    Stealth free-rider: không train nhưng giả metadata giống client hợp lệ.

    Mục đích là stress-test threat model: nếu server chỉ tin data_size/quality
    do client báo về thì attacker dạng này có thể nhận thưởng dù không đóng góp.
    Server-side detection phải dựa trên update features và/hoặc server-known
    data commitment để xử lý.
    """

    def __init__(self, *args, fake_quality: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_quality = float(fake_quality)
        self.client_type = "stealth_free_rider"

    def fit(self, parameters, config):
        updated, num_examples, metrics = super().fit(parameters, config)
        metrics["quality_score"] = max(0.0, self.fake_quality)
        metrics["data_size"] = self.data_size
        metrics["client_type"] = self.client_type
        return updated, num_examples, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Attack 2: Lazy — train rất ít
# ─────────────────────────────────────────────────────────────────────────────

class LazyClient(HonestClient):
    """
    Lazy: train chỉ 1 epoch (thay vì cfg.local_epochs). Quality_score sẽ thấp
    so với honest, nhưng anomaly_score vẫn nằm trong range bình thường →
    khó phát hiện bằng MAD, dễ phát hiện bằng reward (quality thấp).
    """

    def __init__(self, *args, lazy_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_epochs = int(lazy_epochs)
        self.client_type = "lazy"

    def _local_train(self) -> None:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.lazy_epochs):
            for X, y in self.train_loader:
                if X.size(0) < 2:
                    continue
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Attack 3: LabelNoise — train với label sai
# ─────────────────────────────────────────────────────────────────────────────

class LabelNoiseClient(HonestClient):
    """
    Label noise: với xác suất `flip_ratio`, thay label gốc bằng label random khác.
    Train trên dữ liệu bị nhiễu → update có pattern bất thường, anomaly_score
    thường cao hơn honest → CSRA-DCD có khả năng phát hiện.

    Note: label flipping được thực hiện ONLINE trong batch loop, không sửa
    train_loader.dataset (giữ dataset gốc không thay đổi cho client khác).
    """

    def __init__(
        self,
        *args,
        flip_ratio: float = 0.3,
        n_classes: int = 10,
        rng_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= flip_ratio <= 1.0:
            raise ValueError(f"flip_ratio must be in [0, 1], got {flip_ratio}")
        self.flip_ratio = float(flip_ratio)
        self.n_classes = int(n_classes)
        self.client_type = "label_noise"
        self._rng = np.random.default_rng(rng_seed if rng_seed is not None else self.client_id)

    def _local_train(self) -> None:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.cfg.local_epochs):
            for X, y in self.train_loader:
                if X.size(0) < 2:
                    continue
                # Flip labels in-batch
                y_noisy = y.clone()
                n_batch = y.size(0)
                n_flip = int(n_batch * self.flip_ratio)
                if n_flip > 0:
                    flip_idx = self._rng.choice(n_batch, size=n_flip, replace=False)
                    for i in flip_idx:
                        orig = int(y[i].item())
                        candidates = [c for c in range(self.n_classes) if c != orig]
                        y_noisy[i] = int(self._rng.choice(candidates))

                X, y_noisy = X.to(self.device), y_noisy.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y_noisy)
                loss.backward()
                optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Attack 4: SignFlip — đảo dấu delta (Byzantine)
# ─────────────────────────────────────────────────────────────────────────────

class SignFlipClient(HonestClient):
    """
    Sign-flip: train bình thường, nhưng sau khi train xong, đảo dấu delta:
        local_params_attacked = global_params - (local_params - global_params)
                              = 2 * global_params - local_params

    Là Byzantine attack mạnh nhất ở đây. ||Δ||₂ của sign-flip ≥ ||Δ||₂ của
    honest (cùng magnitude nhưng ngược chiều) → KHÔNG outlier qua MAD trực tiếp,
    nhưng làm hỏng accuracy mạnh nếu không được lọc.
    """

    def __init__(self, *args, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # scale > 1 = boosted attack, scale < 0 = invert
        self.scale = float(scale)
        self.client_type = "sign_flip"

    def fit(self, parameters, config):
        global_params = [np.array(p, copy=True) for p in parameters]
        set_parameters(self.model, parameters)

        loss_before = self._eval_loss()
        self._local_train()
        loss_after = self._eval_loss()

        quality = float(max(0.0, loss_before - loss_after))
        current_params = get_parameters(self.model)

        # Apply sign-flip: new_params = global - scale * (local - global)
        attacked_params = [
            base + (-self.scale) * (local - base)
            for local, base in zip(current_params, global_params)
        ]
        anomaly_score = self._delta_norm(attacked_params, global_params)

        return (
            attacked_params,
            self.data_size,
            self._fit_metrics(quality, self.data_size, anomaly_score),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_NAMES = (
    "free_rider",
    "stealth_free_rider",
    "lazy",
    "label_noise",
    "sign_flip",
)

ATTACK_CLIENT_REGISTRY = {
    "honest": HonestClient,
    "free_rider": FreeRiderClient,
    "stealth_free_rider": StealthFreeRiderClient,
    "lazy": LazyClient,
    "label_noise": LabelNoiseClient,
    "sign_flip": SignFlipClient,
}


def make_client(
    client_type: str,
    client_id: int,
    dataset_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    fl_cfg: FLConfig | None = None,
    **extra_kwargs,
) -> HonestClient:
    """
    Factory cho client. client_type ∈ {"honest", "free_rider",
    "stealth_free_rider", "lazy", "label_noise", "sign_flip"}.

    extra_kwargs sẽ được forward tới constructor của attack class tương ứng:
      - FreeRider/StealthFreeRider: noise_std, mode, fake_quality
      - Lazy: lazy_epochs
      - LabelNoise: flip_ratio, n_classes, rng_seed
      - SignFlip: scale
    """
    if client_type not in ATTACK_CLIENT_REGISTRY:
        raise ValueError(
            f"unknown client_type '{client_type}', "
            f"expected one of {list(ATTACK_CLIENT_REGISTRY)}"
        )
    cls = ATTACK_CLIENT_REGISTRY[client_type]
    return cls(
        client_id=client_id,
        dataset_name=dataset_name,
        train_loader=train_loader,
        test_loader=test_loader,
        fl_cfg=fl_cfg,
        **extra_kwargs,
    )
