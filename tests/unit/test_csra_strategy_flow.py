from types import SimpleNamespace

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from fl.config import ExperimentConfig, FLConfig, ProjectConfig
from fl.logger import ExperimentLogger
from fl.server_csra import FLCSRAStrategy


class DummyBridge:
    def __init__(self):
        self.submissions = []
        self.excluded = None

    def get_reputation(self, client_idx):
        return 0.5, True

    def submit_contribution(self, client_idx, quality, data_size, mean_data_size, round_num):
        self.submissions.append((client_idx, quality, data_size, round_num))

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


def _fit_res(value: float, anomaly_score: float, quality: float = 0.2, data_size: int = 10):
    params = ndarrays_to_parameters([np.array([value], dtype=np.float32)])
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=data_size,
        metrics={
            "quality_score": quality,
            "data_size": data_size,
            "anomaly_score": anomaly_score,
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
