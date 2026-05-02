"""
data_utils.py — Partition dữ liệu cho 3 dataset × 3 kịch bản Non-IID.
Hỗ trợ: MNIST, Fashion-MNIST, CIFAR-10.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Optional, Tuple
from fl.config import ExperimentConfig

# ── Dataset registry ─────────────────────────────────────────
_DATASET_MAP = {
    "mnist": (
        datasets.MNIST,
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
    ),
    "fashion_mnist": (
        datasets.FashionMNIST,
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.2860,), (0.3530,))])
    ),
    "cifar10": (
        datasets.CIFAR10,
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2470, 0.2435, 0.2616))])
    ),
}

N_CLASSES = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10}


def load_dataset(name: str, train: bool = True):
    """Load dataset từ torchvision, download nếu chưa có."""
    DatasetClass, transform = _DATASET_MAP[name]
    return DatasetClass(root="./data", train=train, download=True, transform=transform)


def add_label_noise(indices: np.ndarray, labels: np.ndarray,
                    noise_ratio: float, n_classes: int, seed: int = 42) -> np.ndarray:
    """Flip label ngẫu nhiên cho một tỷ lệ mẫu trong tập indices."""
    rng = np.random.default_rng(seed)
    noisy = indices.copy()
    n_noisy = int(len(noisy) * noise_ratio)
    chosen = rng.choice(len(noisy), size=n_noisy, replace=False)
    for i in chosen:
        orig = labels[noisy[i]]
        candidates = [c for c in range(n_classes) if c != orig]
        labels[noisy[i]] = rng.choice(candidates)
    return noisy


def partition_iid(labels: np.ndarray, n_clients: int, seed: int = 42) -> List[np.ndarray]:
    """K1: IID — chia đều ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return [arr for arr in np.array_split(indices, n_clients)]


def partition_weak_noniid(labels: np.ndarray, n_clients: int,
                          classes_per_client: int = 5, seed: int = 42) -> List[np.ndarray]:
    """
    K2: Weak Non-IID — mỗi client chỉ có classes_per_client class.
    Thuật toán: chia từng class thành n_clients/classes_per_client phần,
    mỗi client nhận một phần từ mỗi class được chỉ định.
    Đảm bảo mọi client đều có đủ dữ liệu và không bị thiếu.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    # Shuffle và chia từng class thành n_clients phần bằng nhau
    class_shards = {}
    for c in range(n_classes):
        idx = rng.permutation(np.where(labels == c)[0])
        class_shards[c] = np.array_split(idx, n_clients)

    # Mỗi client nhận classes_per_client class ngẫu nhiên (không trùng hoàn toàn)
    client_splits = []
    for i in range(n_clients):
        # Chọn class cho client này bằng cách xoay vòng để đảm bảo coverage
        chosen = [(i * classes_per_client + j) % n_classes for j in range(classes_per_client)]
        # Shuffle nhẹ để tránh pattern quá đều
        chosen = [c ^ (i % 3) % n_classes if rng.random() > 0.7 else c for c in chosen]
        chosen = list(set(chosen))[:classes_per_client]
        while len(chosen) < classes_per_client:
            chosen.append(rng.integers(0, n_classes))
        chosen = list(set(chosen))[:classes_per_client]

        shards = [class_shards[c][i % len(class_shards[c])] for c in chosen]
        client_splits.append(np.concatenate(shards))

    return client_splits


def partition_dirichlet(labels: np.ndarray, n_clients: int,
                        n_classes: int, beta: float = 0.1,
                        seed: int = 42) -> List[np.ndarray]:
    """K3: Strong Non-IID — phân phối Dirichlet."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    client_splits = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx = rng.permutation(class_indices[c])
        proportions = rng.dirichlet(np.repeat(beta, n_clients))
        proportions = np.cumsum(proportions * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)
        for ci, s in enumerate(splits):
            client_splits[ci].extend(s.tolist())

    return [np.array(s) for s in client_splits]


def get_client_partitions(
    dataset_name: str,
    n_clients: int,
    cfg: ExperimentConfig
) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """
    Hàm tổng hợp: load dataset, partition theo scenario, thêm noise.

    Returns:
        splits:         List[np.ndarray] — index array cho từng client
        labels:         np.ndarray — labels của toàn bộ train set
        mean_data_size: float — |D̄| dùng cho Hybrid Normalization fallback
    """
    dataset = load_dataset(dataset_name, train=True)
    labels  = np.array(dataset.targets)
    n_cls   = N_CLASSES[dataset_name]

    if cfg.scenario == "K1":
        splits = partition_iid(labels, n_clients, cfg.seed)
    elif cfg.scenario == "K2":
        splits = partition_weak_noniid(labels, n_clients, classes_per_client=5, seed=cfg.seed)
    elif cfg.scenario == "K3":
        splits = partition_dirichlet(labels, n_clients, n_cls, cfg.dirichlet_beta, cfg.seed)
    else:
        raise ValueError(f"Unknown scenario: {cfg.scenario}")

    # Thêm label noise cho client chỉ định
    for ci in cfg.noise_clients:
        if ci < len(splits) and len(splits[ci]) > 0:
            splits[ci] = add_label_noise(splits[ci], labels, cfg.noise_ratio, n_cls, cfg.seed + ci)

    # Lazy clients: chỉ dùng subset nhỏ
    for ci in cfg.lazy_client_ids:
        if ci < len(splits):
            n_keep = max(10, int(len(splits[ci]) * cfg.lazy_data_ratio))
            splits[ci] = splits[ci][:n_keep]

    mean_data_size = float(np.mean([len(s) for s in splits]))
    return splits, labels, mean_data_size


def make_dataloader(dataset, indices: np.ndarray, batch_size: int,
                    shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
