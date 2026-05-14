"""Test tất cả metric functions."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from fl.metrics import (
    abc_metric,
    accuracy_reversal,
    contribution_reward_correlation,
    convergence_round,
    economic_incentive_index,
    fairness_gap,
    false_positive_rate,
    free_rider_detection_rate,
    gini_coefficient,
    jain_index,
    reward_leakage,
    reward_ratio,
)


def test_jain_equal():
    r = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(jain_index(r) - 1.0) < 1e-6


def test_jain_unequal():
    r = np.array([10.0, 0.0, 0.0, 0.0])
    assert jain_index(r) < 0.4  # bất bình đẳng cao


def test_gini_equal():
    r = np.array([5.0, 5.0, 5.0])
    assert gini_coefficient(r) < 1e-6


def test_fairness_gap_perfect():
    """FG=0 khi reward tỷ lệ hoàn hảo với contribution."""
    c = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 2.0, 3.0])
    assert fairness_gap(r, c) < 1e-6


def test_fairness_gap_imperfect():
    c = np.array([1.0, 1.0, 1.0])
    r = np.array([3.0, 0.0, 0.0])
    fg = fairness_gap(r, c)
    assert fg > 0.3


def test_abc_positive():
    acc_low = [0.90, 0.88, 0.85]
    acc_high = [0.80, 0.75, 0.70]
    assert abc_metric(acc_low, acc_high) > 0


def test_accuracy_reversal_none():
    acc_low = [0.90, 0.88, 0.85]
    acc_high = [0.80, 0.75, 0.70]
    assert accuracy_reversal(acc_low, acc_high) == 0


def test_accuracy_reversal_detected():
    acc_low = [0.70, 0.88]
    acc_high = [0.80, 0.75]
    assert accuracy_reversal(acc_low, acc_high) == 1


def test_fdr_perfect():
    assert free_rider_detection_rate([7, 8], [7, 8]) == 1.0


def test_fdr_miss():
    assert free_rider_detection_rate([7], [7, 8]) == 0.5


def test_eii_above_one():
    # EII = (r_honest - r_lazy) / (d_honest / d_lazy)
    # = (10 - 2) / (1000/100) = 8/10 = 0.8 — đi 0.8 < 1 là đúng về kinh tế học
    # Test thực tế: EII > 1 khi delta_reward lớn hơn delta_cost nhiều
    eii = economic_incentive_index(r_honest=50, r_lazy=2, d_honest=1000, d_lazy=100)
    assert eii > 1.0  # (50-2)/(1000/100) = 48/10 = 4.8 > 1


def test_false_positive_rate_counts_honest_only():
    assert false_positive_rate(
        detected_clients=[1, 3],
        malicious_clients=[3],
        all_clients=[0, 1, 2, 3],
    ) == 1 / 3


def test_reward_leakage_uses_malicious_reward_share():
    rewards = np.array([0.6, 0.3, 0.1])
    malicious = np.array([False, False, True])
    assert reward_leakage(rewards, malicious) == pytest.approx(0.1)


def test_convergence_round_requires_stable_peak_ratio_window():
    accuracies = [0.1, 0.5, 0.91, 0.7, 0.93, 0.94, 0.95, 0.96]
    rounds = list(range(1, 9))
    assert convergence_round(accuracies, rounds=rounds, peak_ratio=0.95, patience=3) == 5
