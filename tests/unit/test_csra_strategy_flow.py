from types import SimpleNamespace

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl.config import ExperimentConfig, FLConfig, ProjectConfig
from fl.logger import ExperimentLogger
from fl.server_csra import FLCSRAStrategy
from fl.server_base import FLUnifiedStrategy


class DummyBridge:
    def __init__(self):
        self.submissions = []
        self.excluded = None
        self.audit_rewards = []

    def get_reputation(self, client_idx):
        return 0.5, True

    def submit_contribution(self, client_idx, quality, data_size, mean_data_size, round_num):
        self.submissions.append((client_idx, quality, data_size, round_num))

    def distribute_audit(self, rewards_eth, round_num):
        self.audit_rewards.append((dict(rewards_eth), round_num))
        return True

    def filter_and_distribute(
        self,
        n_clients,
        quality_scores,
        data_sizes,
        alpha,
        mean_data_size,
        round_num,
        pool_eth,
        exclude_client_indices=None,
    ):
        self.excluded = list(exclude_client_indices or [])
        return {
            cid: 1.0 / (n_clients - len(self.excluded))
            for cid in range(n_clients)
            if cid not in self.excluded
        }


class ReputationBridge(DummyBridge):
    def __init__(self, dishonest_cids):
        super().__init__()
        self.dishonest_cids = set(dishonest_cids)

    def get_reputation(self, client_idx):
        if client_idx in self.dishonest_cids:
            return 0.05, False
        return 0.8, True


def _fit_res(
    value: float,
    anomaly_score: float,
    quality: float = 0.2,
    data_size: int = 10,
    num_examples: int | None = None,
):
    params = ndarrays_to_parameters([np.array([value], dtype=np.float32)])
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=data_size if num_examples is None else num_examples,
        metrics={
            "quality_score": quality,
            "data_size": data_size,
            "anomaly_score": anomaly_score,
            "client_type": "honest",
        },
    )


def _fit_res_params(
    values,
    quality: float = 0.2,
    data_size: int = 10,
    num_examples: int | None = None,
):
    params = ndarrays_to_parameters([np.asarray(values, dtype=np.float32)])
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=data_size if num_examples is None else num_examples,
        metrics={
            "quality_score": quality,
            "data_size": data_size,
            "anomaly_score": 0.0,
            "client_type": "honest",
        },
    )


def test_csra_dcd_excludes_detected_client_from_reward(tmp_path):
    bridge = DummyBridge()
    logger = ExperimentLogger("csra-flow", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(config_type="C"),
    )
    strategy = FLCSRAStrategy(
        bridge=bridge,
        exp_logger=logger,
        cfg=cfg,
        alpha=0.5,
        mean_data_size=10,
        client_types={0: "honest", 1: "honest", 2: "free_rider"},
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(0.0, 0.10)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.11)),
            (SimpleNamespace(cid="2"), _fit_res(100.0, 10.00, quality=0.4, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert bridge.excluded == [2]
    assert bridge.submissions[-1] == (2, 0.0, 0, 1)
    assert rows[2]["is_anomaly"] is True
    assert rows[2]["is_honest"] is False
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_excludes_anomaly_from_reward_audit(tmp_path):
    bridge = DummyBridge()
    logger = ExperimentLogger("unified-audit", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "sign_flip"},
        mean_data_size=10,
        seed=42,
        bridge=bridge,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(-1.0, 0.0, quality=0.4, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert (2, 0.0, 0, 1) in bridge.submissions
    assert bridge.audit_rewards
    rewards_eth, round_num = bridge.audit_rewards[-1]
    assert round_num == 1
    assert set(rewards_eth) == {0, 1}
    assert 2 not in rewards_eth
    assert rows[2]["is_anomaly"] is True
    assert rows[2]["is_honest"] is False
    assert rows[2]["reward_eth"] == 0.0


def test_unified_strategy_blocks_low_reputation_from_reward_only(tmp_path):
    bridge = ReputationBridge(dishonest_cids={2})
    logger = ExperimentLogger("unified-reputation-gate", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "previously_flagged"},
        mean_data_size=10,
        seed=42,
        bridge=bridge,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(1.0, 0.0, quality=0.4, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert bridge.audit_rewards
    rewards_eth, round_num = bridge.audit_rewards[-1]
    assert round_num == 1
    assert set(rewards_eth) == {0, 1}
    assert 2 not in rewards_eth
    assert rows[2]["is_anomaly"] is False
    assert rows[2]["detection_reason"] == "low_reputation"
    assert rows[2]["is_honest"] is False
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_reward_quarantines_zero_data_commitment(tmp_path):
    logger = ExperimentLogger("unified-zero-data-quarantine", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="fedavg",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "free_rider"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(
                1.0, 0.0, quality=0.0, data_size=0, num_examples=10,
            )),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["is_anomaly"] is False
    assert rows[2]["data_commitment_anomaly"] is True
    assert rows[2]["suspicion_quarantine"] is True
    assert rows[2]["reward_blocked"] is True
    assert "data_commitment_zero" in rows[2]["detection_reason"]
    assert "suspicion_quarantine" in rows[2]["detection_reason"]
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_uses_server_known_size_and_logs_mismatch(tmp_path):
    logger = ExperimentLogger("unified-server-known-size", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="fedavg",
        reward_policy="data",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "free_rider"},
        mean_data_size=10,
        server_data_sizes={0: 10, 1: 10, 2: 10},
        seed=42,
        bridge=None,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0, data_size=10)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0, data_size=10)),
            (SimpleNamespace(cid="2"), _fit_res(
                1.0, 0.0, quality=0.0, data_size=0, num_examples=10,
            )),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["reported_data_size"] == 0
    assert rows[2]["server_known_data_size"] == 10
    assert rows[2]["data_size"] == 10
    assert rows[2]["data_commitment_anomaly"] is True
    assert rows[2]["data_size_mismatch"] is True
    assert rows[2]["reward_blocked"] is True
    assert rows[2]["ground_truth_honest"] is False
    assert rows[2]["reward_eligible"] is False
    assert rows[2]["is_honest"] is False
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["ground_truth_honest"] is True
    assert rows[0]["reward_eligible"] is True
    assert rows[0]["is_honest"] is True
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0
    assert rows[0]["reward_component_data"] == 0.5
    assert rows[1]["reward_component_data"] == 0.5
    assert rows[2]["reward_component_data"] == 0.0
    assert rows[0]["reward_component_quality"] == 0.0
    assert rows[0]["reward_component_reputation"] == 0.0


def test_unified_strategy_logs_quality_policy_reward_components(tmp_path):
    logger = ExperimentLogger("unified-quality-components", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=2),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="fedavg",
        reward_policy="quality",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest"},
        mean_data_size=10,
        server_data_sizes={0: 10, 1: 10},
        seed=42,
        bridge=None,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0, data_size=10)),
            (SimpleNamespace(cid="1"), _fit_res(2.0, 0.0, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert np.isclose(rows[0]["reward_component_quality"], 0.5)
    assert np.isclose(rows[1]["reward_component_quality"], 0.5)
    assert rows[0]["reward_component_data"] == 0.0
    assert rows[0]["reward_component_reputation"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_quarantines_stealth_free_rider_low_authenticity(tmp_path):
    logger = ExperimentLogger("unified-stealth-freerider", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=5),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="csra",
        cfg=cfg,
        exp_logger=logger,
        client_types={**{i: "honest" for i in range(3)}, 3: "stealth_free_rider", 4: "stealth_free_rider"},
        mean_data_size=10,
        server_data_sizes={i: 10 for i in range(5)},
        seed=42,
        bridge=None,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        initial_parameters=ndarrays_to_parameters([np.array([0.0, 0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res_params([1.0, -1.0], quality=0.4, data_size=10)),
            (SimpleNamespace(cid="1"), _fit_res_params([1.0, -1.0], quality=0.4, data_size=10)),
            (SimpleNamespace(cid="2"), _fit_res_params([1.0, -1.0], quality=0.4, data_size=10)),
            (SimpleNamespace(cid="3"), _fit_res_params([0.0, 0.0], quality=0.25, data_size=10)),
            (SimpleNamespace(cid="4"), _fit_res_params([0.0, 0.0], quality=0.25, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[3]["reported_data_size"] == 10
    assert rows[3]["server_known_data_size"] == 10
    assert rows[3]["data_commitment_anomaly"] is False
    assert rows[3]["authenticity_anomaly"] is True
    assert rows[3]["suspicion_quarantine"] is True
    assert rows[3]["reward_blocked"] is True
    assert rows[3]["reward_eth"] == 0.0
    assert rows[4]["reward_blocked"] is True
    assert rows[4]["reward_eth"] == 0.0
    assert all(rows[cid]["reward_eth"] > 0.0 for cid in range(3))
    assert rows[0]["raw_update_norm"] > 0.0
    assert rows[0]["normalized_update_score"] > 0.0
    assert rows[0]["reward_component_quality"] >= 0.0


def test_unified_strategy_quarantines_inefficient_large_update_reward_only(tmp_path):
    logger = ExperimentLogger("unified-inefficient-update", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="csra",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "label_noise"},
        mean_data_size=(10 + 20 + 100) / 3,
        server_data_sizes={0: 10, 1: 20, 2: 100},
        seed=42,
        bridge=None,
        high_update_norm_z_threshold=4.0,
        inefficient_update_suspicion=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res_params([1.0], quality=1.0, data_size=10)),
            (SimpleNamespace(cid="1"), _fit_res_params([2.0], quality=1.2, data_size=20)),
            (SimpleNamespace(cid="2"), _fit_res_params([10.0], quality=0.9, data_size=100)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["raw_update_norm_z"] >= 4.0
    assert np.isclose(
        rows[2]["normalized_update_score"],
        rows[0]["normalized_update_score"],
    )
    assert rows[2]["is_anomaly"] is False
    assert rows[2]["inefficient_update"] is True
    assert rows[2]["suspicion_quarantine"] is True
    assert rows[2]["reward_blocked"] is True
    assert rows[2]["reward_eth"] == 0.0
    assert "inefficient_update" in rows[2]["suspicion_reason"]
    assert rows[0]["reward_blocked"] is False
    assert rows[1]["reward_blocked"] is False


def test_unified_strategy_filters_zero_data_commitment_from_csra_aggregation(tmp_path):
    logger = ExperimentLogger("unified-zero-data-agg-filter", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "free_rider"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        mad_threshold=999.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(2.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(
                100.0, 0.0, quality=0.0, data_size=0, num_examples=10,
            )),
        ]
        aggregated_params, _ = strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["is_anomaly"] is True
    assert rows[2]["data_commitment_anomaly"] is True
    assert rows[2]["reward_eth"] == 0.0
    assert np.allclose(parameters_to_ndarrays(aggregated_params)[0], 1.5)


def test_unified_strategy_rolling_suspicion_blocks_repeated_low_quality(tmp_path):
    logger = ExperimentLogger("unified-rolling-suspicion", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="fedavg",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "lazy"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        suspicion_decay=0.5,
        suspicion_threshold=1.0,
        low_quality_suspicion=0.7,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        for round_num in [1, 2]:
            results = [
                (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0, quality=0.4, data_size=10)),
                (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0, quality=0.4, data_size=10)),
                (SimpleNamespace(cid="2"), _fit_res(1.0, 0.0, quality=0.0, data_size=10)),
            ]
            strategy.aggregate_fit(round_num, results, [])
        rows_r1 = {row["client_id"]: row for row in strategy._pending_logs[1]}
        rows_r2 = {row["client_id"]: row for row in strategy._pending_logs[2]}
    finally:
        logger.close()

    assert rows_r1[2]["low_quality_outlier"] is True
    assert rows_r1[2]["suspicion_quarantine"] is False
    assert rows_r1[2]["reward_eth"] > 0.0
    assert rows_r2[2]["suspicion_quarantine"] is True
    assert rows_r2[2]["reward_blocked"] is True
    assert rows_r2[2]["reward_eth"] == 0.0
    assert "suspicion_quarantine" in rows_r2[2]["detection_reason"]


def test_unified_strategy_reward_quarantines_direction_after_failsafe(tmp_path):
    logger = ExperimentLogger("unified-failsafe-quarantine", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "sign_flip"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        min_honest_ratio=0.9,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(-1.0, 0.0, quality=0.4, data_size=10)),
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["is_anomaly"] is False
    assert rows[2]["direction_anomaly"] is True
    assert rows[2]["reward_blocked"] is True
    assert rows[2]["detection_reason"] == (
        "fallback_accept_all+reward_quarantine_direction"
    )
    assert rows[2]["is_honest"] is False
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_soft_filters_extreme_norm_after_failsafe(tmp_path):
    logger = ExperimentLogger("unified-failsafe-soft-filter", str(tmp_path))
    n_clients = 10
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=n_clients),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={**{i: "honest" for i in range(9)}, 9: "sign_flip"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        min_honest_ratio=1.0,
        fallback_hard_z=6.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (SimpleNamespace(cid=str(cid)), _fit_res(1.0, 0.0))
            for cid in range(9)
        ]
        results.append(
            (SimpleNamespace(cid="9"), _fit_res(100.0, 0.0, quality=0.4, data_size=10))
        )
        aggregated_params, _ = strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[9]["is_anomaly"] is True
    assert rows[9]["reward_blocked"] is True
    assert rows[9]["detection_reason"] == "norm_mad+fallback_hard_block"
    assert rows[9]["is_honest"] is False
    assert rows[9]["reward_eth"] == 0.0
    assert all(rows[cid]["reward_eth"] > 0.0 for cid in range(9))

    assert np.allclose(parameters_to_ndarrays(aggregated_params)[0], 1.0)


def test_unified_strategy_recomputes_server_update_features(tmp_path):
    logger = ExperimentLogger("unified-flow", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(dataset="mnist", scenario="K1"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={0: "honest", 1: "honest", 2: "sign_flip"},
        mean_data_size=10,
        seed=42,
        bridge=None,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        # All clients lie/report anomaly_score=0. Norm-MAD alone cannot catch
        # client 2 because all update norms are equal, but direction cosine can.
        results = [
            (SimpleNamespace(cid="0"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0, 0.0)),
            (SimpleNamespace(cid="2"), _fit_res(-1.0, 0.0, quality=0.4, data_size=10)),
        ]
        strategy.set_current_global_parameters([np.array([0.0], dtype=np.float32)])
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert rows[2]["anomaly_score"] == 1.0
    assert rows[2]["anomaly_score_source"] == "server_data_normalized"
    assert rows[2]["direction_anomaly"] is True
    assert rows[2]["is_anomaly"] is True
    assert rows[2]["detection_reason"] == "direction_cosine"
    assert rows[2]["reward_eth"] == 0.0
    assert rows[0]["reward_eth"] > 0.0
    assert rows[1]["reward_eth"] > 0.0


def test_unified_strategy_does_not_flag_data_volume_norm_in_noniid(tmp_path):
    logger = ExperimentLogger("unified-data-normalized-dcd", str(tmp_path))
    data_sizes = [8505, 1420, 888, 16379, 1737]
    mean_data_size = float(np.mean(data_sizes))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=5),
        experiment=ExperimentConfig(dataset="mnist", scenario="K3"),
    )
    strategy = FLUnifiedStrategy(
        aggregation_method="csra_dcd",
        reward_policy="equal",
        cfg=cfg,
        exp_logger=logger,
        client_types={cid: "honest" for cid in range(5)},
        mean_data_size=mean_data_size,
        seed=42,
        bridge=None,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    try:
        results = [
            (
                SimpleNamespace(cid=str(cid)),
                _fit_res(
                    value=float(size) * 0.044,
                    anomaly_score=0.0,
                    quality=0.4,
                    data_size=size,
                ),
            )
            for cid, size in enumerate(data_sizes)
        ]
        strategy.aggregate_fit(1, results, [])
        rows = {row["client_id"]: row for row in strategy._pending_logs[1]}
    finally:
        logger.close()

    assert all(rows[cid]["anomaly_score_source"] == "server_data_normalized" for cid in rows)
    assert all(rows[cid]["is_anomaly"] is False for cid in rows)
    assert all(rows[cid]["reward_blocked"] is False for cid in rows)
    assert all(rows[cid]["reward_eth"] > 0.0 for cid in rows)
