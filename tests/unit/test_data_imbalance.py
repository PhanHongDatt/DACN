"""Tests for fl.data_utils.apply_data_imbalance — Schema v2 data heterogeneity."""
import numpy as np
import pytest

from fl.data_utils import apply_data_imbalance


def _make_splits(n_clients: int, samples_per_client: int = 1000):
    """Tạo n_clients splits, mỗi client có samples_per_client samples."""
    rng = np.random.default_rng(0)
    total = n_clients * samples_per_client
    indices = rng.permutation(total)
    return np.array_split(indices, n_clients)


class TestApplyDataImbalance:
    def test_uniform_returns_unchanged(self):
        splits = _make_splits(10)
        result = apply_data_imbalance(splits, pattern="uniform", seed=42)
        # Sizes should be identical
        assert [len(s) for s in result] == [len(s) for s in splits]

    def test_linear_creates_3_to_4x_variance(self):
        splits = _make_splits(10, samples_per_client=1000)
        result = apply_data_imbalance(splits, pattern="linear", seed=42)
        sizes = sorted([len(s) for s in result])
        ratio = sizes[-1] / max(sizes[0], 1)
        assert 3.0 <= ratio <= 5.0, f"linear ratio out of range: {ratio:.2f}"

    def test_lognormal_creates_high_variance(self):
        splits = _make_splits(10, samples_per_client=1000)
        result = apply_data_imbalance(splits, pattern="lognormal", seed=42)
        sizes = sorted([len(s) for s in result])
        ratio = sizes[-1] / max(sizes[0], 1)
        # lognormal có thể từ ~5x đến ~15x
        assert ratio >= 3.0, f"lognormal ratio too low: {ratio:.2f}"

    def test_step_creates_two_groups(self):
        splits = _make_splits(10, samples_per_client=1000)
        result = apply_data_imbalance(splits, pattern="step", seed=42)
        sizes = sorted([len(s) for s in result])
        # 5 ở mức thấp, 5 ở mức cao → max/min = 3x
        ratio = sizes[-1] / max(sizes[0], 1)
        assert 2.5 <= ratio <= 3.5, f"step ratio expected ~3x, got {ratio:.2f}"

    def test_only_downsamples_never_upsamples(self):
        """Total samples sau không bao giờ vượt total trước (chỉ drop)."""
        splits = _make_splits(10, samples_per_client=1000)
        before = sum(len(s) for s in splits)
        for pattern in ["linear", "lognormal", "step"]:
            result = apply_data_imbalance(splits, pattern=pattern, seed=42)
            after = sum(len(s) for s in result)
            assert after <= before, f"{pattern} upsamples: {before} -> {after}"

    def test_indices_preserve_partition_origin(self):
        """Indices của mỗi client sau phải là subset của partition gốc."""
        splits = _make_splits(10, samples_per_client=1000)
        result = apply_data_imbalance(splits, pattern="lognormal", seed=42)
        for orig, new in zip(splits, result):
            assert set(new.tolist()).issubset(set(orig.tolist())), \
                "Client indices không phải subset partition gốc"

    def test_reproducibility(self):
        """Cùng seed → cùng result."""
        splits = _make_splits(10)
        r1 = apply_data_imbalance(splits, pattern="lognormal", seed=42)
        r2 = apply_data_imbalance(splits, pattern="lognormal", seed=42)
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b)

    def test_different_seeds_give_different_results(self):
        splits = _make_splits(10)
        r1 = apply_data_imbalance(splits, pattern="lognormal", seed=42)
        r2 = apply_data_imbalance(splits, pattern="lognormal", seed=123)
        sizes_1 = tuple(len(s) for s in r1)
        sizes_2 = tuple(len(s) for s in r2)
        # Lognormal random → sizes thường khác nhau
        assert sizes_1 != sizes_2

    def test_unknown_pattern_rejected(self):
        splits = _make_splits(5)
        with pytest.raises(ValueError, match="Unknown data_imbalance"):
            apply_data_imbalance(splits, pattern="garbage", seed=42)

    def test_min_samples_floor(self):
        """Client nhỏ vẫn giữ tối thiểu min_samples."""
        splits = _make_splits(10, samples_per_client=1000)
        result = apply_data_imbalance(splits, pattern="lognormal", seed=42, min_samples=100)
        sizes = [len(s) for s in result]
        assert min(sizes) >= 100, f"min_samples violated: {min(sizes)}"

    def test_empty_or_single_split_passthrough(self):
        assert apply_data_imbalance([], pattern="lognormal", seed=42) == []
        single = [np.array([1, 2, 3])]
        result = apply_data_imbalance(single, pattern="lognormal", seed=42)
        assert len(result) == 1
