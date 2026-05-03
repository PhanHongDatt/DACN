"""
client_csra.py — Improved Flower client implementation based on the CSRA framework.
This client adds reporting for model update variance and a bidding mechanism 
to support fair economic incentives and anomaly detection.
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
        bid_base: float = 0.05
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
            - variance: Statistical variance of the model updates (used for DCD).
            - bid: The requested reward for this round.
        """
        set_parameters(self.model, parameters)
        
        if self.client_type == "free_rider":
            # Simulate a free-rider by adding high-variance noise to the global parameters.
            # This triggers the Variance-based Detection (DCD) at the server.
            noisy_params = [p + np.random.normal(0, 0.5, p.shape).astype(p.dtype)
                            for p in parameters]
            
            # Compute variance of the injected noise.
            flat_update = np.concatenate([p.flatten() for p in noisy_params])
            variance = float(np.var(flat_update))
            
            return (
                noisy_params,
                self.data_size,
                {
                    "quality_score": 0.01, # Small fake quality score
                    "data_size": self.data_size,
                    "variance": variance,
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
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()

        loss_after = self._eval_loss()
        quality = float(max(0.0, loss_before - loss_after))
        
        # Compute the statistical variance of the learned parameters.
        # This helps the server distinguish between actual learning and random noise.
        current_params = get_parameters(self.model)
        flat_params = np.concatenate([p.flatten() for p in current_params])
        variance = float(np.var(flat_params))

        return (
            current_params,
            self.data_size,
            {
                "quality_score": quality,
                "data_size": self.data_size,
                "variance": variance,
                "bid": self.bid,
                "client_type": self.client_type
            }
        )

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
