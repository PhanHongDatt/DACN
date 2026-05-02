"""
test_stability.py — Kiểm tra các edge case có thể gây crash khi chạy thực nghiệm.
Đây là bộ test quan trọng nhất để đảm bảo 70 runs chạy không gián đoạn.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fl.normalization import hybrid_normalize, compute_w_new
from fl.metrics import fairness_gap, jain_index, gini_coefficient
from fl.config import ContributionConfig

cfg = ContributionConfig()


# ── Edge cases của Hybrid Normalization ─────────────────────

class TestNormalizationStability:

    def test_all_zeros_no_crash(self):
        """Tất cả client có quality=0 (free-rider round đầu)."""
        v = np.zeros(10)
        result = hybrid_normalize(v, mean_val=1.0, cfg=cfg)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_single_client_no_nan(self):
        """Chỉ 1 client trong P_honest."""
        v = np.array([42.0])
        result = hybrid_normalize(v, mean_val=42.0, cfg=cfg)
        assert not np.isnan(result[0])
        assert result[0] > 0

    def test_iid_identical_values(self):
        """IID: mọi client cùng data size — LH1."""
        v = np.full(10, 6000.0)
        result = hybrid_normalize(v, mean_val=6000.0, cfg=cfg)
        assert not np.any(np.isnan(result))
        assert np.allclose(result, 1.0)

    def test_extreme_noniid_large_gap(self):
        """Non-IID cực đoan: gap 1000x giữa client lớn nhất và nhỏ nhất."""
        v = np.array([50000.0, 50.0, 100.0, 200.0, 150.0,
                      60000.0, 80.0, 90.0, 110.0, 130.0])
        result = hybrid_normalize(v, cfg=cfg)
        assert not np.any(np.isnan(result))
        assert result.min() >= 0
        assert result.max() <= 1.0 + 1e-6

    def test_very_small_values_near_zero(self):
        """Quality score gần 0 sau nhiều vòng IID."""
        v = np.array([1e-8, 2e-8, 1.5e-8, 3e-8, 1e-8,
                      2.5e-8, 1e-8, 2e-8, 1e-8, 1e-8])
        result = hybrid_normalize(v, cfg=cfg)
        assert not np.any(np.isnan(result))

    def test_w_new_always_sums_to_one(self):
        """W_new phải luôn sum = 1 bất kể input."""
        test_cases = [
            (np.zeros(5), np.ones(5) * 1000),
            (np.ones(5), np.zeros(5)),
            (np.array([0.9, 0.0, 0.0, 0.0, 0.1]), np.array([100, 100, 100, 100, 100])),
            (np.full(10, 1e-10), np.full(10, 1e-10)),
        ]
        for q, d in test_cases:
            w = compute_w_new(q, d, alpha=0.5, cfg=cfg)
            assert abs(w.sum() - 1.0) < 1e-5, f"sum={w.sum()} for q={q}, d={d}"
            assert all(wi >= 0 for wi in w)

    def test_p_honest_with_2_clients(self):
        """P_honest chỉ còn 2 client sau khi lọc free-rider."""
        q = np.array([0.8, 0.7])
        d = np.array([5000, 6000])
        w = compute_w_new(q, d, alpha=0.5, mean_data_size=5500.0, cfg=cfg)
        assert abs(w.sum() - 1.0) < 1e-5
        assert w[1] > w[0]  # client 1 có data lớn hơn → w lớn hơn ở α=0.5


# ── Edge cases của Metrics ───────────────────────────────────

class TestMetricsStability:

    def test_fairness_gap_zero_reward(self):
        """FG không crash khi một client reward=0."""
        r = np.array([0.0, 1.0, 2.0])
        c = np.array([0.1, 0.4, 0.5])
        fg = fairness_gap(r, c)
        assert not np.isnan(fg)

    def test_jain_index_one_client(self):
        """Jain index với 1 client = 1.0."""
        assert abs(jain_index(np.array([5.0])) - 1.0) < 1e-6

    def test_gini_all_zeros(self):
        """Gini với tất cả reward = 0."""
        g = gini_coefficient(np.zeros(5))
        assert not np.isnan(g)
        assert g == 0.0

    def test_fairness_gap_all_equal(self):
        """FG = 0 khi tất cả client có cùng tỷ lệ reward/contribution."""
        r = np.array([2.0, 4.0, 6.0])
        c = np.array([1.0, 2.0, 3.0])
        assert fairness_gap(r, c) < 1e-6


# ── Kiểm tra data partition không tạo empty split ────────────

class TestDataPartition:

    def test_no_empty_splits_k1(self):
        """IID: không client nào có 0 mẫu."""
        pytest.importorskip("torch", reason="torch not installed in test env")
        import numpy as np
        from fl.data_utils import partition_iid
        labels = np.random.randint(0, 10, size=60000)
        splits = partition_iid(labels, n_clients=10, seed=42)
        for i, s in enumerate(splits):
            assert len(s) > 0, f"Client {i} có 0 mẫu trong K1"

    def test_no_empty_splits_k3(self):
        """Dirichlet K3: tất cả client đều có ít nhất vài mẫu."""
        pytest.importorskip("torch", reason="torch not installed in test env")
        import numpy as np
        from fl.data_utils import partition_dirichlet
        labels = np.random.randint(0, 10, size=60000)
        splits = partition_dirichlet(labels, n_clients=10, n_classes=10,
                                     beta=0.1, seed=42)
        for i, s in enumerate(splits):
            assert len(s) > 0, f"Client {i} có 0 mẫu trong K3 (Dirichlet)"

    def test_lazy_client_has_minimum_samples(self):
        """Lazy client phải có ít nhất 10 mẫu (tránh DataLoader crash)."""
        pytest.importorskip("torch", reason="torch not installed in test env")
        from fl.config import ExperimentConfig
        from fl.data_utils import get_client_partitions
        import unittest.mock as mock

        mock_ds = mock.MagicMock()
        mock_ds.targets = list(range(10)) * 6000

        with mock.patch("fl.data_utils.load_dataset", return_value=mock_ds):
            mock_ds.__len__ = lambda self: 60000
            exp_cfg = ExperimentConfig(
                dataset="mnist", scenario="K1",
                lazy_client_ids=[9], lazy_data_ratio=0.1, seed=42
            )
            splits, _, _ = get_client_partitions("mnist", 10, exp_cfg)
            assert len(splits[9]) >= 10, f"Lazy client chỉ có {len(splits[9])} mẫu"
