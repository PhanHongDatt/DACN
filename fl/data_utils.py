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
    """Flip label ngẫu nhiên cho một tỷ lệ mẫu trong tập indices.

    Lưu ý: hàm này sửa mảng labels tại chỗ để runner có thể áp lại vào dataset.
    """
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


def apply_data_imbalance(
    splits: List[np.ndarray],
    pattern: str,
    seed: int,
    min_samples: int = 50,
) -> List[np.ndarray]:
    """
    Tạo data size heterogeneity giữa các client bằng cách downsample.

    Cần thiết cho K1/K2 vì partition mặc định cho ra sizes gần như đồng đều,
    khiến reward policies như DataSize/CSRA không có data variance để differentiate.

    Patterns:
      "uniform"   — không thay đổi (giữ partition gốc)
      "linear"    — weights = linspace(0.4, 1.6) of base, max/min ≈ 4×
      "lognormal" — μ=0 σ=0.7, clipped [0.2, 3.0], max/min ≈ 10-15× (realistic)
      "step"      — nửa client 0.5×, nửa 1.5× (max/min = 3×, hai nhóm rõ rệt)

    Args:
        splits: List of index arrays from partition function.
        pattern: One of {"uniform", "linear", "lognormal", "step"}.
        seed: For reproducible weight draw + sub-sampling.
        min_samples: Floor for client size (tránh client quá nhỏ không train được).

    Returns:
        New splits với sizes đã apply imbalance. Mỗi client giữ ngẫu nhiên một
        tập con index của partition ban đầu.

    Raises:
        ValueError nếu pattern không hợp lệ.
    """
    if pattern == "uniform" or len(splits) <= 1:
        return splits

    # Seed offset để weight draw không correlate với client_id partition
    rng = np.random.default_rng(int(seed) + 1000)
    n = len(splits)

    if pattern == "linear":
        weights = np.linspace(0.4, 1.6, n)
    elif pattern == "lognormal":
        weights = rng.lognormal(mean=0.0, sigma=0.7, size=n)
        weights = np.clip(weights, 0.2, 3.0)
    elif pattern == "step":
        half = n // 2
        weights = np.array([0.5] * half + [1.5] * (n - half), dtype=float)
    else:
        raise ValueError(
            f"Unknown data_imbalance pattern '{pattern}'. "
            f"Expected: uniform | linear | lognormal | step"
        )

    # Shuffle để client_id không correlate với size
    rng.shuffle(weights)

    # Normalize: max weight = 1.0 — chỉ downsample, không upsample (không có data thêm)
    weights = weights / weights.max()

    result: List[np.ndarray] = []
    for i, split in enumerate(splits):
        target_n = max(min_samples, int(len(split) * weights[i]))
        if target_n >= len(split):
            result.append(split)
        else:
            keep_idx = rng.choice(len(split), size=target_n, replace=False)
            # Sort indices để reproducible CSV row order
            result.append(np.sort(split[keep_idx]))
    return result


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

    # Mỗi client nhận classes_per_client class hợp lệ (không trùng trong client)
    client_splits = []
    target_classes = min(classes_per_client, n_classes)
    for i in range(n_clients):
        # Chọn class cho client này bằng cách xoay vòng để đảm bảo coverage
        chosen = [(i * classes_per_client + j) % n_classes for j in range(target_classes)]
        # Shuffle nhẹ để tránh pattern quá đều
        varied = []
        for c in chosen:
            if rng.random() > 0.7:
                c = (c + (i % 3)) % n_classes
            if c not in varied:
                varied.append(int(c))

        while len(varied) < target_classes:
            c = int(rng.integers(0, n_classes))
            if c not in varied:
                varied.append(c)

        shards = [class_shards[c][i % len(class_shards[c])] for c in varied]
        client_splits.append(np.concatenate(shards))

    return client_splits


def apply_labels(dataset, labels: np.ndarray) -> None:
    """Áp labels đã xử lý/noise vào torchvision dataset trước khi tạo Subset."""
    labels = np.asarray(labels, dtype=np.int64)
    if hasattr(dataset, "targets"):
        current = dataset.targets
        if torch.is_tensor(current):
            dataset.targets = torch.as_tensor(labels, dtype=current.dtype)
        else:
            dataset.targets = labels.tolist()
    elif hasattr(dataset, "labels"):
        current = dataset.labels
        if torch.is_tensor(current):
            dataset.labels = torch.as_tensor(labels, dtype=current.dtype)
        else:
            dataset.labels = labels.tolist()
    else:
        raise AttributeError("Dataset không có thuộc tính targets/labels để áp label noise")


def partition_dirichlet(labels: np.ndarray, n_clients: int,
                        n_classes: int, beta: float = 0.1,
                        seed: int = 42, min_size: int = 10,
                        max_retries: int = 100) -> List[np.ndarray]:
    """K3: Strong Non-IID — phân phối Dirichlet."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]

    def sample_once() -> List[np.ndarray]:
        client_splits = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = rng.permutation(class_indices[c])
            proportions = rng.dirichlet(np.repeat(beta, n_clients))
            proportions = np.cumsum(proportions * len(idx)).astype(int)[:-1]
            splits = np.split(idx, proportions)
            for ci, s in enumerate(splits):
                client_splits[ci].extend(s.tolist())
        return [np.array(s, dtype=int) for s in client_splits]

    for _ in range(max_retries):
        splits = sample_once()
        if min(len(s) for s in splits) >= min_size:
            return splits

    splits = sample_once()
    if len(labels) < n_clients * min_size:
        return splits

    # Fallback deterministic balancing: only used for extreme Dirichlet draws.
    while min(len(s) for s in splits) < min_size:
        dst = int(np.argmin([len(s) for s in splits]))
        src = int(np.argmax([len(s) for s in splits]))
        movable = len(splits[src]) - min_size
        if movable <= 0:
            break
        need = min_size - len(splits[dst])
        take = min(need, movable)
        moved = splits[src][-take:]
        splits[src] = splits[src][:-take]
        splits[dst] = np.concatenate([splits[dst], moved])
    return splits


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

    # Thêm label noise cho client chỉ định (legacy, schema v2 dùng LabelNoiseClient)
    for ci in cfg.noise_clients:
        if ci < len(splits) and len(splits[ci]) > 0:
            splits[ci] = add_label_noise(splits[ci], labels, cfg.noise_ratio, n_cls, cfg.seed + ci)

    # Lazy clients: chỉ dùng subset nhỏ (legacy, schema v2 dùng LazyClient)
    for ci in cfg.lazy_client_ids:
        if ci < len(splits):
            n_keep = max(10, int(len(splits[ci]) * cfg.lazy_data_ratio))
            splits[ci] = splits[ci][:n_keep]

    # Schema v2: Apply data imbalance để có variance cho DataSize/CSRA reward.
    # Mặc định "lognormal" tạo max/min ≈ 10-15× realistic cho FL.
    pattern = getattr(cfg, "data_imbalance", "uniform")
    if pattern != "uniform":
        splits = apply_data_imbalance(splits, pattern, cfg.seed)

    mean_data_size = float(np.mean([len(s) for s in splits]))
    return splits, labels, mean_data_size


def make_dataloader(dataset, indices: np.ndarray, batch_size: int,
                    shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
