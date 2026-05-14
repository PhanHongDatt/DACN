import numpy as np
import torch

from fl.data_utils import (
    add_label_noise,
    apply_labels,
    partition_dirichlet,
    partition_weak_noniid,
)


def test_weak_noniid_uses_valid_classes_for_many_seeds():
    labels = np.repeat(np.arange(10), 100)

    for seed in range(100):
        splits = partition_weak_noniid(labels, n_clients=10, classes_per_client=5, seed=seed)
        assert len(splits) == 10
        assert all(len(split) > 0 for split in splits)
        assert all(np.all((split >= 0) & (split < len(labels))) for split in splits)


def test_dirichlet_keeps_minimum_client_size_for_many_seeds():
    labels = np.repeat(np.arange(10), 600)

    for seed in range(20):
        splits = partition_dirichlet(labels, n_clients=10, n_classes=10, beta=0.1, seed=seed)
        assert min(len(split) for split in splits) >= 10


def test_add_label_noise_mutates_labels_in_place():
    labels = np.arange(20) % 10
    before = labels.copy()

    add_label_noise(np.arange(20), labels, noise_ratio=1.0, n_classes=10, seed=42)

    assert not np.array_equal(labels, before)
    assert np.all((labels >= 0) & (labels < 10))


def test_apply_labels_updates_list_targets():
    class Dataset:
        targets = [0, 1, 2]

    dataset = Dataset()
    apply_labels(dataset, np.array([2, 1, 0]))

    assert dataset.targets == [2, 1, 0]


def test_apply_labels_updates_tensor_targets():
    class Dataset:
        targets = torch.tensor([0, 1, 2], dtype=torch.long)

    dataset = Dataset()
    apply_labels(dataset, np.array([2, 1, 0]))

    assert torch.equal(dataset.targets, torch.tensor([2, 1, 0], dtype=torch.long))
