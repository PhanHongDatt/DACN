"""Test Hybrid Normalization — đặc biệt kiểm tra LH1 (IID suy biến)."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from fl.normalization import hybrid_normalize, compute_w_new
from fl.config import ContributionConfig

cfg = ContributionConfig()

def test_minmax_normal():
    v = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    result = hybrid_normalize(v, cfg=cfg)
    assert result.min() >= 0 and result.max() <= 1.0
    assert abs(result.max() - 1.0) < 1e-6

def test_iid_no_nan():
    """LH1: tất cả client cùng data size → không NaN, trả về 1.0"""
    v = np.array([6000.0] * 10)
    result = hybrid_normalize(v, mean_val=6000.0, cfg=cfg)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert np.allclose(result, 1.0)

def test_single_client():
    """Trường hợp chỉ 1 client trung thực — không crash."""
    v = np.array([5000.0])
    result = hybrid_normalize(v, mean_val=5000.0, cfg=cfg)
    assert not np.isnan(result[0])

def test_w_new_sums_to_one():
    q = np.array([0.8, 0.6, 0.4, 0.3, 0.7])
    d = np.array([5000, 3000, 8000, 2000, 6000])
    w = compute_w_new(q, d, alpha=0.5, mean_data_size=4800.0, cfg=cfg)
    assert abs(w.sum() - 1.0) < 1e-6
    assert all(wi >= 0 for wi in w)

def test_alpha_zero_only_data():
    """α=0: W_new chỉ phụ thuộc data size."""
    q = np.array([0.9, 0.1])  # quality rất khác nhau
    d = np.array([1000, 1000])  # data size bằng nhau
    w = compute_w_new(q, d, alpha=0.0, cfg=cfg)
    assert abs(w[0] - w[1]) < 1e-6  # phải bằng nhau vì chỉ xét data size

def test_alpha_one_only_quality():
    """α=1: W_new chỉ phụ thuộc quality score."""
    q = np.array([0.8, 0.2])
    d = np.array([1000, 9000])  # data size rất khác nhau
    w = compute_w_new(q, d, alpha=1.0, cfg=cfg)
    assert w[0] > w[1]  # client 0 có quality cao hơn → w lớn hơn
