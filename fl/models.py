"""
models.py — Kiến trúc model cho 3 dataset.
  - MNIST / Fashion-MNIST: 2-layer MLP (đơn giản, huấn luyện nhanh)
  - CIFAR-10: Small CNN (2 conv + 2 fc)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    2-hidden-layer MLP cho MNIST và Fashion-MNIST.
    784 → 256 → 128 → 10, với BatchNorm và Dropout để ổn định trong Non-IID.
    Accuracy baseline: MNIST ~98%, Fashion-MNIST ~90-91% (IID).
    Trong Non-IID K3 có thể xuống 80-85% — đây là expected behavior.
    """
    def __init__(self, input_dim: int = 784, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class SmallCNN(nn.Module):
    """Small CNN cho CIFAR-10."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, n_classes)
        self.drop  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.drop(x)))
        return self.fc2(x)


def get_model(dataset_name: str) -> nn.Module:
    if dataset_name in ("mnist", "fashion_mnist"):
        return MLP()
    elif dataset_name == "cifar10":
        return SmallCNN()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_parameters(model: nn.Module):
    """
    Lấy toàn bộ state_dict bao gồm cả BatchNorm buffers (running_mean, running_var).
    Quan trọng: nếu chỉ dùng model.parameters() sẽ bỏ qua buffers → BatchNorm sai.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    """
    Set toàn bộ state_dict từ list numpy arrays.
    Thứ tự phải khớp với state_dict().keys() — Flower đảm bảo điều này.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = {k: torch.tensor(np.array(v)) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
