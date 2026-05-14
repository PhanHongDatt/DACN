"""
client_csra.py — Flower client implementation based on the CSRA framework.
This client reports update-delta statistics and a bidding signal to support
fair economic incentives and anomaly detection.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from fl.models import get_model, set_parameters, get_parameters
from fl.config import FLConfig

class FLClientCSRA(fl.client.NumPyClient):
    """
    Flower client with CSRA-inspired extensions.
    
    Attributes:
        client_id: Unique identifier for the client.
        dataset_name: Name of the dataset being used.
        train_loader: DataLoader for local training data.
        test_loader: DataLoader for local testing data.
        client_type: Behavior type ('honest', 'free_rider', or 'lazy').
        fl_cfg: Configuration object for FL parameters.
        bid_base: The base bid amount (ETH) this client requests for training.
    """
    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        client_type: str = "honest",
        fl_cfg: FLConfig = None,
        bid_base: float = 0.05,
        free_rider_mode: str = "noise",
        free_rider_noise_std: float = 0.05,
    ):
        self.client_id = client_id
        self.dataset = dataset_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_type = client_type
        self.cfg = fl_cfg or FLConfig()
        self.model = get_model(dataset_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.data_size = len(train_loader.dataset)
        self.free_rider_mode = free_rider_mode
        self.free_rider_noise_std = float(free_rider_noise_std)
        
        # CSRA Bidding: Honest clients bid their true cost, 
        # while malicious ones might underbid to stay in the selection pool.
        self.bid = bid_base if client_type == "honest" else bid_base * 0.5

    def get_parameters(self, config):
        """Returns the current local model parameters."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """
        Performs local training and returns model updates with CSRA metadata.
        
        Metadata includes:
            - quality_score: Improvement in local loss (Delta loss).
            - anomaly_score/update_norm: L2 norm of local update delta.
            - variance: Backward-compatible alias of anomaly_score.
            - bid: The requested reward for this round.
        """
        global_params = [np.array(p, copy=True) for p in parameters]
        set_parameters(self.model, parameters)
        
        if self.client_type == "free_rider":
            if self.free_rider_mode == "copy":
                updated_params = [np.array(p, copy=True) for p in global_params]
            else:
                updated_params = [
                    p + np.random.normal(0, self.free_rider_noise_std, p.shape).astype(p.dtype)
                    for p in global_params
                ]

            anomaly_score = self._delta_norm(updated_params, global_params)
            
            return (
                updated_params,
                self.data_size,
                {
                    "quality_score": 0.01, # Small fake quality score
                    "data_size": 0,
                    "anomaly_score": anomaly_score,
                    "update_norm": anomaly_score,
                    "variance": anomaly_score,
                    "bid": self.bid,
                    "client_type": self.client_type
                }
            )

        # Standard local training for honest and lazy clients.
        loss_before = self._eval_loss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(self.cfg.local_epochs):
            for X, y in self.train_loader:
                if X.size(0) < 2:
                    continue
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()

        loss_after = self._eval_loss()
        quality = float(max(0.0, loss_before - loss_after))
        
        current_params = get_parameters(self.model)
        anomaly_score = self._delta_norm(current_params, global_params)

        return (
            current_params,
            self.data_size,
            {
                "quality_score": quality,
                "data_size": self.data_size,
                "anomaly_score": anomaly_score,
                "update_norm": anomaly_score,
                "variance": anomaly_score,
                "bid": self.bid,
                "client_type": self.client_type
            }
        )

    @staticmethod
    def _delta_norm(local_params, global_params) -> float:
        """Return the L2 norm of the model update delta."""
        deltas = [
            (local.astype(np.float64, copy=False) - base.astype(np.float64, copy=False)).ravel()
            for local, base in zip(local_params, global_params)
        ]
        if not deltas:
            return 0.0
        return float(np.linalg.norm(np.concatenate(deltas)))

    def evaluate(self, parameters, config):
        """Evaluates the model on local test data."""
        set_parameters(self.model, parameters)
        loss, accuracy = self._eval_accuracy()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def _eval_loss(self) -> float:
        """Helper to calculate local loss."""
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                total_loss += criterion(self.model(X), y).item()
        return total_loss / max(len(self.train_loader), 1)

    def _eval_accuracy(self):
        """Helper to calculate local accuracy."""
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
