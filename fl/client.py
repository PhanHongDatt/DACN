"""
client.py — Flower client với 3 loại hành vi: honest, free_rider, lazy.
Tính quality_score = Δloss (mức độ cải thiện loss sau local training).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from fl.models import get_model, set_parameters, get_parameters
from fl.config import FLConfig


class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_type: str = "honest",   # "honest" | "free_rider" | "lazy"
        fl_cfg: FLConfig = None
    ):
        self.client_id   = client_id
        self.dataset     = dataset_name
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.client_type  = client_type
        self.cfg          = fl_cfg or FLConfig()
        self.model        = get_model(dataset_name)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.data_size    = len(train_loader.dataset)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Local training. Free-rider gửi noise, lazy chỉ dùng subset nhỏ."""
        set_parameters(self.model, parameters)
        round_num = config.get("round", 0)

        if self.client_type == "free_rider":
            # Gửi update ngẫu nhiên, không train thực sự.
            # BUG7 FIX: báo data_size=0 thay vì data_size thật.
            # Free-rider không cam kết dữ liệu → data commitment thấp
            # → reputation giảm dần → bị loại khỏi P_honest.
            noisy_params = [p + np.random.normal(0, 0.1, p.shape).astype(p.dtype)
                            for p in parameters]
            return (
                noisy_params,
                0,   # báo 0 mẫu — không cam kết dữ liệu
                {"quality_score": 0.0, "data_size": 0, "client_type": self.client_type}
            )

        # Tính loss trước khi train (để tính Δloss = quality score)
        loss_before = self._eval_loss()

        # Training
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.cfg.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(self.cfg.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()

        loss_after = self._eval_loss()
        quality = float(max(0.0, loss_before - loss_after))  # Δloss ≥ 0

        return (
            get_parameters(self.model),
            self.data_size,
            {"quality_score": quality, "data_size": self.data_size, "client_type": self.client_type}
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self._eval_accuracy()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def _eval_loss(self) -> float:
        """Tính loss trên local training data. Dùng eval mode để BatchNorm ổn định."""
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                total_loss += criterion(self.model(X), y).item()
        # Quan trọng: trả lại train mode sau khi eval
        self.model.train()
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
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        return total_loss / max(len(self.test_loader), 1), correct / max(total, 1)
