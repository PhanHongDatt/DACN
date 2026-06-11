"""Unit tests cho fl/aggregation_methods.py."""
import math

import numpy as np
import pytest

from fl.aggregation_methods import (
    AGGREGATION_NAMES,
    apply_aggregation,
    compute_update_features,
    csra_dcd_aggregation,
    detect_anomalies,
    fedavg_aggregation,
    mad_robust_zscore,
    trimmed_mean_aggregation,
)


def _make_client_params(values_per_client: list[float], shape=(2, 3)):
    """Tạo client_params đơn giản: mỗi client có 1 layer toàn giá trị constant."""
    return [[np.full(shape, v, dtype=np.float32)] for v in values_per_client]


# ── FedAvg ────────────────────────────────────────────────────────────────────

class TestFedAvg:
    def test_uniform_weights_equals_mean(self):
        # 3 client, mỗi client có layer toàn 1, 2, 3
        params = _make_client_params([1.0, 2.0, 3.0])
        agg = fedavg_aggregation(params, weights=[1.0, 1.0, 1.0])
        # Trung bình = 2.0
        assert np.allclose(agg[0], 2.0)

    def test_weighted_average(self):
        # Client 0 weight 1, client 1 weight 3 → trung bình = 0.25*1 + 0.75*3 = 2.5
        params = _make_client_params([1.0, 3.0])
        agg = fedavg_aggregation(params, weights=[1.0, 3.0])
        assert np.allclose(agg[0], 2.5)

    def test_all_zero_weights_fallback_uniform(self):
        params = _make_client_params([1.0, 2.0])
        agg = fedavg_aggregation(params, weights=[0.0, 0.0])
        # Fallback uniform → mean = 1.5
        assert np.allclose(agg[0], 1.5)

    def test_multi_layer(self):
        # 2 client, 2 layer khác nhau
        params = [
            [np.array([1.0, 2.0]), np.array([[10.0]])],
            [np.array([3.0, 4.0]), np.array([[20.0]])],
        ]
        agg = fedavg_aggregation(params, weights=[1.0, 1.0])
        assert np.allclose(agg[0], [2.0, 3.0])
        assert np.allclose(agg[1], [[15.0]])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fedavg_aggregation([], weights=[])

    def test_rejects_mismatched_layers(self):
        params = [
            [np.array([1.0])],
            [np.array([1.0]), np.array([2.0])],  # 2 layer
        ]
        with pytest.raises(ValueError, match="layers"):
            fedavg_aggregation(params, weights=[1.0, 1.0])

    def test_rejects_negative_weights(self):
        params = _make_client_params([1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            fedavg_aggregation(params, weights=[1.0, -1.0])


# ── TrimmedMean ──────────────────────────────────────────────────────────────

class TestTrimmedMean:
    def test_trim_outliers(self):
        # 10 client, 1 outlier ở giá trị 100, còn lại 1.0. trim_ratio=0.1 → trim 1 mỗi đầu
        params = _make_client_params([1.0] * 9 + [100.0])
        agg = trimmed_mean_aggregation(params, trim_ratio=0.1)
        # Sau khi trim đỉnh (100) và đáy (1.0), còn 8 client toàn 1.0 → mean = 1.0
        assert np.allclose(agg[0], 1.0)

    def test_zero_trim_equals_mean(self):
        params = _make_client_params([1.0, 2.0, 3.0])
        agg = trimmed_mean_aggregation(params, trim_ratio=0.0)
        assert np.allclose(agg[0], 2.0)

    def test_high_trim_fallback_to_mean(self):
        # trim_ratio quá cao → fallback plain mean
        params = _make_client_params([1.0, 2.0])
        agg = trimmed_mean_aggregation(params, trim_ratio=0.4)
        # n=2, trim_k = int(2 * 0.4) = 0 → mean = 1.5
        # Nhưng nếu trim_k * 2 >= n, fallback. Ở đây trim_k=0 nên không fallback.
        assert np.allclose(agg[0], 1.5)

    def test_rejects_invalid_trim_ratio(self):
        params = _make_client_params([1.0, 2.0])
        with pytest.raises(ValueError, match="trim_ratio"):
            trimmed_mean_aggregation(params, trim_ratio=0.5)
        with pytest.raises(ValueError, match="trim_ratio"):
            trimmed_mean_aggregation(params, trim_ratio=-0.1)


# ── MAD robust z-score ───────────────────────────────────────────────────────

class TestMadRobustZScore:
    def test_normal_case_mad(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z, method = mad_robust_zscore(scores)
        assert method == "mad"
        # median = 3, MAD = median(|x-3|) = median([2,1,0,1,2]) = 1
        # z = |x-3| / (1.4826 * 1)
        expected_abs_dev = np.array([2, 1, 0, 1, 2]) / 1.4826
        assert np.allclose(z, expected_abs_dev)

    def test_outlier_high_z(self):
        # 4 client gần như bằng nhau, 1 outlier
        # MAD = 0 ở đây (do 4 client đều bằng median) → method dùng fallback
        scores = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        z, method = mad_robust_zscore(scores)
        assert method in {"mad", "mean_abs_dev_fallback"}
        # Client cuối phải có z rất cao
        assert z[-1] > 3
        # 4 client đầu z = 0 (đều == median)
        for zi in z[:-1]:
            assert zi < 1

    def test_outlier_with_real_variance(self):
        # MAD path: 5 client phân bố, 1 outlier
        scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        z, method = mad_robust_zscore(scores)
        assert method == "mad"
        assert z[-1] > 10  # outlier có z rất cao

    def test_all_equal_returns_zeros(self):
        scores = np.array([5.0, 5.0, 5.0])
        z, method = mad_robust_zscore(scores)
        assert method == "mad_zero_all_equal"
        assert np.allclose(z, 0)

    def test_empty_returns_empty(self):
        z, method = mad_robust_zscore(np.array([]))
        assert method == "empty"
        assert z.size == 0


class TestDetectAnomalies:
    def test_threshold_works(self):
        scores = [1.0, 1.0, 1.0, 1.0, 100.0]
        mask, z, method = detect_anomalies(scores, mad_threshold=3.0)
        assert mask == [False, False, False, False, True]
        # MAD = 0 → fallback method; chỉ cần đảm bảo flagged đúng
        assert method in {"mad", "mean_abs_dev_fallback"}

    def test_no_anomaly_when_all_equal(self):
        scores = [5.0, 5.0, 5.0]
        mask, z, method = detect_anomalies(scores)
        assert mask == [False, False, False]

    def test_empty(self):
        mask, z, method = detect_anomalies([])
        assert mask == []
        assert z == []


# ── CSRA-DCD aggregation ─────────────────────────────────────────────────────

class TestCSRADCDAggregation:
    def test_filters_outlier(self):
        # 5 client: 4 honest (giá trị 1.0), 1 byzantine (giá trị 100.0)
        params = _make_client_params([1.0, 1.0, 1.0, 1.0, 100.0])
        anomaly_scores = [0.1, 0.1, 0.1, 0.1, 50.0]

        agg, mask, z, method, meta = csra_dcd_aggregation(
            params, weights=[1.0] * 5, anomaly_scores=anomaly_scores,
            mad_threshold=3.0,
        )
        # Client 4 phải bị flag
        assert mask == [False, False, False, False, True]
        # Aggregation chỉ trên 4 honest → mean ≈ 1.0
        assert np.allclose(agg[0], 1.0)
        # method có thể là "mad" hoặc fallback tuỳ vào MAD value
        assert method in {"mad", "mean_abs_dev_fallback"}
        assert meta["detection_reasons"][-1] == "norm_mad"
        assert meta["reward_block_mask"] == [False, False, False, False, True]
        assert meta["fallback_accept_all"] is False

    def test_no_filter_when_uniform(self):
        params = _make_client_params([1.0, 2.0, 3.0])
        scores = [1.0, 1.0, 1.0]  # đều nhau
        agg, mask, z, method, meta = csra_dcd_aggregation(
            params, weights=[1.0] * 3, anomaly_scores=scores,
        )
        assert mask == [False, False, False]
        assert np.allclose(agg[0], 2.0)

    def test_failsafe_too_many_anomalies(self):
        # Cấu trúc: 2 client honest (giá trị 0), 3 client byzantine (giá trị 100)
        # MAD path: median=100, abs_dev=[100,100,0,0,0], MAD=0 → fallback
        # mean abs_dev = 40, scale = 1.4826*40 = 59.3
        # z = [100/59.3, 100/59.3, 0, 0, 0] ≈ [1.69, 1.69, 0, 0, 0]
        # Với threshold=1.0, 2 client bị flag. n_honest=3, min_honest=⌈0.8*5⌉=4. 3<4 → failsafe!
        params = _make_client_params([0.0, 0.0, 100.0, 100.0, 100.0])
        scores = [0.0, 0.0, 100.0, 100.0, 100.0]
        agg, mask, z, method, meta = csra_dcd_aggregation(
            params, weights=[1.0] * 5, anomaly_scores=scores,
            mad_threshold=1.0,
            min_honest_ratio=0.8,
        )
        # Failsafe kick in → tất cả mask = False
        assert mask == [False, False, False, False, False]
        assert "fallback_accept_all" in method
        assert meta["detection_reasons"] == ["fallback_accept_all"] * 5
        assert meta["reward_block_mask"] == [False] * 5
        assert meta["pre_fallback_anomaly_mask"] == [True, True, False, False, False]
        assert meta["fallback_triggered"] is True
        assert meta["fallback_soft_filter"] is False

    def test_failsafe_soft_filters_extreme_norm_outlier(self):
        # min_honest_ratio=1.0 forces failsafe even when only one client is
        # filtered. The outlier is still a high-confidence norm anomaly, so the
        # soft hard-filter excludes it from aggregation/reward.
        params = _make_client_params([1.0] * 9 + [100.0])
        scores = [1.0] * 9 + [100.0]

        agg, mask, z, method, meta = csra_dcd_aggregation(
            params,
            weights=[1.0] * 10,
            anomaly_scores=scores,
            mad_threshold=3.0,
            min_honest_ratio=1.0,
            fallback_hard_z=6.0,
        )

        assert "fallback_soft_filter" in method
        assert mask == [False] * 9 + [True]
        assert meta["fallback_triggered"] is True
        assert meta["fallback_accept_all"] is False
        assert meta["fallback_soft_filter"] is True
        assert meta["fallback_hard_norm_mask"] == [False] * 9 + [True]
        assert meta["reward_block_mask"] == [False] * 9 + [True]
        assert meta["detection_reasons"][-1] == "norm_mad+fallback_hard_block"
        assert z[-1] >= 6.0
        assert np.allclose(agg[0], 1.0)

    def test_rejects_mismatched_scores(self):
        params = _make_client_params([1.0, 2.0])
        with pytest.raises(ValueError, match="anomaly_scores length"):
            csra_dcd_aggregation(
                params, weights=[1.0, 1.0], anomaly_scores=[1.0],
            )

    def test_direction_filter_catches_equal_norm_sign_flip(self):
        # Base params = [0]. Two honest updates point +1, one sign-flip points -1.
        # Norms are identical, so Norm-MAD alone cannot separate the attacker.
        base = [np.array([0.0], dtype=np.float32)]
        params = [
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([-1.0], dtype=np.float32)],
        ]
        features = compute_update_features(params, base)
        assert np.allclose(features["update_norms"], [1.0, 1.0, 1.0])
        assert features["cosine_to_reference"] == [1.0, 1.0, -1.0]

        agg, mask, z, method, meta = csra_dcd_aggregation(
            params,
            weights=[1.0, 1.0, 1.0],
            anomaly_scores=features["update_norms"],
            update_cosines=features["cosine_to_reference"],
            mad_threshold=3.0,
            cosine_threshold=-0.5,
        )

        assert mask == [False, False, True]
        assert meta["direction_anomaly_mask"] == [False, False, True]
        assert meta["reward_block_mask"] == [False, False, True]
        assert meta["detection_reasons"][2] == "direction_cosine"
        assert np.allclose(agg[0], 1.0)

    def test_failsafe_still_reward_blocks_direction_anomaly(self):
        base = [np.array([0.0], dtype=np.float32)]
        params = [
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([-1.0], dtype=np.float32)],
        ]
        features = compute_update_features(params, base)

        agg, mask, z, method, meta = csra_dcd_aggregation(
            params,
            weights=[1.0, 1.0, 1.0],
            anomaly_scores=features["update_norms"],
            update_cosines=features["cosine_to_reference"],
            mad_threshold=3.0,
            cosine_threshold=-0.5,
            min_honest_ratio=0.9,
        )

        assert mask == [False, False, False]
        assert "fallback_accept_all" in method
        assert meta["direction_anomaly_mask"] == [False, False, True]
        assert meta["pre_fallback_anomaly_mask"] == [False, False, True]
        assert meta["reward_block_mask"] == [False, False, True]
        assert np.allclose(agg[0], 1.0 / 3.0)


# ── Dispatcher ───────────────────────────────────────────────────────────────

class TestApplyAggregation:
    def test_fedavg_returns_no_anomaly(self):
        params = _make_client_params([1.0, 2.0])
        agg, meta = apply_aggregation("fedavg", params, weights=[1.0, 1.0])
        assert meta["anomaly_mask"] == [False, False]
        assert meta["robust_z"] is None
        assert meta["reward_block_mask"] == [False, False]

    def test_trimmed_returns_no_anomaly(self):
        params = _make_client_params([1.0, 2.0, 3.0, 4.0, 5.0])
        agg, meta = apply_aggregation(
            "trimmed", params, weights=[1.0] * 5, trim_ratio=0.1,
        )
        assert meta["anomaly_mask"] == [False] * 5

    def test_csra_dcd_returns_metadata(self):
        params = _make_client_params([1.0, 1.0, 1.0, 1.0, 100.0])
        scores = [0.1, 0.1, 0.1, 0.1, 50.0]
        agg, meta = apply_aggregation(
            "csra_dcd", params, weights=[1.0] * 5,
            anomaly_scores=scores, mad_threshold=3.0,
        )
        assert meta["anomaly_mask"] == [False, False, False, False, True]
        assert meta["robust_z"] is not None
        assert len(meta["robust_z"]) == 5
        assert meta["detection_method"] is not None

    def test_unknown_method_rejected(self):
        params = _make_client_params([1.0, 2.0])
        with pytest.raises(ValueError, match="unknown aggregation"):
            apply_aggregation("foo", params, weights=[1.0, 1.0])

    def test_csra_requires_scores(self):
        params = _make_client_params([1.0, 2.0])
        with pytest.raises(ValueError, match="anomaly_scores required"):
            apply_aggregation("csra_dcd", params, weights=[1.0, 1.0])

    def test_all_methods_dispatchable(self):
        params = _make_client_params([1.0, 2.0, 3.0, 4.0, 5.0])
        for name in AGGREGATION_NAMES:
            kwargs = {"weights": [1.0] * 5}
            if name == "csra_dcd":
                kwargs["anomaly_scores"] = [0.5] * 5
            if name == "fedlaw":
                kwargs.update({
                    "trial_params": params,
                    "base_params": _make_client_params([0.0])[0],
                    "w_k": np.ones(5) / 5,
                    "local_losses": np.ones(5),
                })
            agg, meta = apply_aggregation(name, params, **kwargs)
            assert agg is not None
            assert len(meta["anomaly_mask"]) == 5
