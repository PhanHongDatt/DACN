"""Unit tests cho fl/reward_policies.py.

Coverage:
  - 4 policy functions với happy path
  - Invariants: sum == total_reward, non-negative
  - Edge cases: n=1, all-equal, all-zero, mismatched length
  - apply_reward_policy dispatch
  - CSRA weight validation (β+γ+δ=1)
"""
import math

import numpy as np
import pytest

from fl.reward_policies import (
    apply_reward_policy,
    csra_reward,
    data_size_reward,
    equal_split,
    quality_reward,
    POLICY_NAMES,
)


# ── EqualSplit ───────────────────────────────────────────────────────────────

class TestEqualSplit:
    def test_basic_split(self):
        rewards = equal_split([0, 1, 2, 3], total_reward=10.0)
        assert set(rewards) == {0, 1, 2, 3}
        assert all(r == 2.5 for r in rewards.values())
        assert math.isclose(sum(rewards.values()), 10.0)

    def test_single_client(self):
        rewards = equal_split([5], total_reward=1.0)
        assert rewards == {5: 1.0}

    def test_empty_clients(self):
        rewards = equal_split([], total_reward=10.0)
        assert rewards == {}

    def test_zero_reward(self):
        rewards = equal_split([0, 1], total_reward=0.0)
        assert rewards == {0: 0.0, 1: 0.0}

    def test_rejects_negative_reward(self):
        with pytest.raises(ValueError, match="non-negative"):
            equal_split([0, 1], total_reward=-1.0)

    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match="unique"):
            equal_split([0, 0, 1], total_reward=10.0)


# ── DataSize ─────────────────────────────────────────────────────────────────

class TestDataSizeReward:
    def test_proportional(self):
        rewards = data_size_reward([0, 1, 2], data_sizes=[100, 200, 300], total_reward=6.0)
        assert math.isclose(rewards[0], 1.0)
        assert math.isclose(rewards[1], 2.0)
        assert math.isclose(rewards[2], 3.0)
        assert math.isclose(sum(rewards.values()), 6.0)

    def test_all_equal_sizes(self):
        rewards = data_size_reward([0, 1, 2], data_sizes=[100, 100, 100], total_reward=3.0)
        for r in rewards.values():
            assert math.isclose(r, 1.0)

    def test_all_zero_falls_back_to_equal(self):
        rewards = data_size_reward([0, 1, 2], data_sizes=[0, 0, 0], total_reward=3.0)
        for r in rewards.values():
            assert math.isclose(r, 1.0)

    def test_rejects_negative_sizes(self):
        with pytest.raises(ValueError, match="non-negative"):
            data_size_reward([0, 1], data_sizes=[100, -1], total_reward=1.0)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            data_size_reward([0, 1, 2], data_sizes=[100, 200], total_reward=1.0)


# ── QualityOnly ──────────────────────────────────────────────────────────────

class TestQualityReward:
    def test_proportional(self):
        rewards = quality_reward([0, 1], quality_scores=[0.3, 0.7], total_reward=1.0)
        assert math.isclose(rewards[0], 0.3)
        assert math.isclose(rewards[1], 0.7)

    def test_negative_quality_clipped(self):
        # -0.1 → 0, 0.5 → 0.5. Total = 0.5 → client 1 nhận hết
        rewards = quality_reward([0, 1], quality_scores=[-0.1, 0.5], total_reward=1.0)
        assert math.isclose(rewards[0], 0.0)
        assert math.isclose(rewards[1], 1.0)

    def test_all_zero_falls_back_to_equal(self):
        rewards = quality_reward([0, 1, 2], quality_scores=[0.0, 0.0, 0.0], total_reward=3.0)
        for r in rewards.values():
            assert math.isclose(r, 1.0)


# ── CSRAReward ────────────────────────────────────────────────────────────────

class TestCSRAReward:
    def test_default_weights_sum_to_one(self):
        # β=0.5, γ=0.3, δ=0.2 → tổng = 1.0
        rewards = csra_reward(
            client_ids=[0, 1, 2],
            quality_scores=[0.1, 0.5, 0.9],
            data_sizes=[100, 500, 900],
            reputations=[0.1, 0.5, 0.9],
            total_reward=1.0,
        )
        assert math.isclose(sum(rewards.values()), 1.0, abs_tol=1e-6)
        # Client 2 có chất lượng/data/rep cao nhất → nhận nhiều nhất
        assert rewards[2] > rewards[1] > rewards[0]

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            csra_reward(
                client_ids=[0, 1],
                quality_scores=[0.5, 0.5],
                data_sizes=[100, 100],
                reputations=[0.5, 0.5],
                total_reward=1.0,
                beta=0.5, gamma=0.5, delta=0.5,  # sum = 1.5
            )

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            csra_reward(
                client_ids=[0, 1],
                quality_scores=[0.5, 0.5],
                data_sizes=[100, 100],
                reputations=[0.5, 0.5],
                total_reward=1.0,
                beta=-0.1, gamma=0.6, delta=0.5,
            )

    def test_all_equal_inputs_fallback_equal(self):
        # Khi mọi client có Q=D=R bằng nhau, hybrid_normalize → composite có thể ~0
        # → fallback equal_split
        rewards = csra_reward(
            client_ids=[0, 1, 2],
            quality_scores=[0.5, 0.5, 0.5],
            data_sizes=[100, 100, 100],
            reputations=[0.5, 0.5, 0.5],
            total_reward=3.0,
            mean_data_size=100.0,
        )
        # Khi mọi input bằng nhau, mỗi client phải nhận ~1.0 (R/n)
        for r in rewards.values():
            assert math.isclose(r, 1.0, abs_tol=1e-4)

    def test_pure_quality_weight(self):
        # β=1, γ=δ=0 → tương đương quality_reward (sau hybrid_normalize)
        rewards = csra_reward(
            client_ids=[0, 1, 2],
            quality_scores=[0.1, 0.5, 0.9],
            data_sizes=[1, 1, 1],
            reputations=[0.5, 0.5, 0.5],
            total_reward=1.0,
            beta=1.0, gamma=0.0, delta=0.0,
        )
        assert math.isclose(sum(rewards.values()), 1.0, abs_tol=1e-6)
        # Client 2 quality cao nhất → reward cao nhất
        assert rewards[2] > rewards[1] > rewards[0]

    def test_sweep_beta_changes_distribution(self):
        # Cùng input, β khác nhau → distribution thay đổi
        ids = [0, 1, 2]
        q = [0.9, 0.5, 0.1]  # client 0 quality cao
        d = [100, 500, 900]  # client 2 data nhiều
        r = [0.3, 0.5, 0.7]  # client 2 reputation cao

        rewards_high_beta = csra_reward(
            client_ids=ids, quality_scores=q, data_sizes=d, reputations=r,
            total_reward=1.0,
            beta=0.7, gamma=0.18, delta=0.12,
        )
        rewards_low_beta = csra_reward(
            client_ids=ids, quality_scores=q, data_sizes=d, reputations=r,
            total_reward=1.0,
            beta=0.3, gamma=0.42, delta=0.28,
        )
        # β cao → client 0 (quality cao) được lợi nhiều hơn
        assert rewards_high_beta[0] > rewards_low_beta[0]

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            csra_reward(
                client_ids=[0, 1, 2],
                quality_scores=[0.5, 0.5],  # thiếu 1
                data_sizes=[100, 200, 300],
                reputations=[0.5, 0.5, 0.5],
                total_reward=1.0,
            )

    def test_empty_clients(self):
        assert csra_reward(
            client_ids=[],
            quality_scores=[],
            data_sizes=[],
            reputations=[],
            total_reward=1.0,
        ) == {}


# ── Dispatcher ───────────────────────────────────────────────────────────────

class TestApplyRewardPolicy:
    def test_dispatch_equal(self):
        r = apply_reward_policy("equal", [0, 1], total_reward=2.0)
        assert r == {0: 1.0, 1: 1.0}

    def test_dispatch_data(self):
        r = apply_reward_policy(
            "data", [0, 1], total_reward=3.0, data_sizes=[100, 200]
        )
        assert math.isclose(r[1], 2.0)

    def test_dispatch_quality(self):
        r = apply_reward_policy(
            "quality", [0, 1], total_reward=1.0, quality_scores=[0.4, 0.6]
        )
        assert math.isclose(r[0], 0.4)
        assert math.isclose(r[1], 0.6)

    def test_dispatch_csra(self):
        r = apply_reward_policy(
            "csra", [0, 1, 2], total_reward=1.0,
            quality_scores=[0.1, 0.5, 0.9],
            data_sizes=[100, 500, 900],
            reputations=[0.1, 0.5, 0.9],
        )
        assert math.isclose(sum(r.values()), 1.0, abs_tol=1e-6)

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="unknown reward policy"):
            apply_reward_policy("foo", [0, 1], total_reward=1.0)

    def test_missing_input_for_data(self):
        with pytest.raises(ValueError, match="data_sizes required"):
            apply_reward_policy("data", [0, 1], total_reward=1.0)

    def test_missing_input_for_quality(self):
        with pytest.raises(ValueError, match="quality_scores required"):
            apply_reward_policy("quality", [0, 1], total_reward=1.0)

    def test_missing_input_for_csra(self):
        with pytest.raises(ValueError, match="csra reward needs"):
            apply_reward_policy(
                "csra", [0, 1], total_reward=1.0,
                quality_scores=[0.5, 0.5],
                # thiếu data_sizes, reputations
            )

    def test_all_policy_names_dispatchable(self):
        # Smoke test: tất cả tên trong POLICY_NAMES đều hợp lệ
        for name in POLICY_NAMES:
            try:
                apply_reward_policy(
                    name, [0, 1], total_reward=1.0,
                    quality_scores=[0.5, 0.5],
                    data_sizes=[100, 100],
                    reputations=[0.5, 0.5],
                )
            except ValueError as e:
                pytest.fail(f"Policy '{name}' không nên fail: {e}")


# ── Invariants ────────────────────────────────────────────────────────────────

class TestInvariants:
    @pytest.mark.parametrize("policy,kwargs", [
        ("equal", {}),
        ("data", {"data_sizes": [10, 20, 30, 40]}),
        ("quality", {"quality_scores": [0.1, 0.3, 0.5, 0.7]}),
        ("csra", {
            "quality_scores": [0.1, 0.3, 0.5, 0.7],
            "data_sizes": [10, 20, 30, 40],
            "reputations": [0.2, 0.4, 0.6, 0.8],
        }),
    ])
    def test_sum_equals_total_reward(self, policy, kwargs):
        rewards = apply_reward_policy(
            policy, [0, 1, 2, 3], total_reward=10.0, **kwargs
        )
        assert math.isclose(sum(rewards.values()), 10.0, abs_tol=1e-6), \
            f"Policy '{policy}' không bảo toàn tổng reward"

    @pytest.mark.parametrize("policy,kwargs", [
        ("equal", {}),
        ("data", {"data_sizes": [10, 20, 30, 40]}),
        ("quality", {"quality_scores": [0.1, 0.3, 0.5, 0.7]}),
        ("csra", {
            "quality_scores": [0.1, 0.3, 0.5, 0.7],
            "data_sizes": [10, 20, 30, 40],
            "reputations": [0.2, 0.4, 0.6, 0.8],
        }),
    ])
    def test_all_rewards_non_negative(self, policy, kwargs):
        rewards = apply_reward_policy(
            policy, [0, 1, 2, 3], total_reward=10.0, **kwargs
        )
        for cid, r in rewards.items():
            assert r >= 0, f"Policy '{policy}' tạo reward âm cho client {cid}: {r}"
