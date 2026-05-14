from types import SimpleNamespace

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl.config import ExperimentConfig, FLConfig, ProjectConfig
from fl.logger import ExperimentLogger
from fl.server_trimmed import TrimmedMeanStrategy


def _fit_res(value: float):
    params = ndarrays_to_parameters([np.array([value], dtype=np.float32)])
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=1,
        metrics={"quality_score": 0.1, "data_size": 1, "client_type": "honest"},
    )


def test_trimmed_mean_removes_coordinate_outlier(tmp_path):
    logger = ExperimentLogger("trimmed", str(tmp_path))
    cfg = ProjectConfig(
        fl=FLConfig(n_clients=3),
        experiment=ExperimentConfig(config_type="TrimmedMean"),
    )
    strategy = TrimmedMeanStrategy(
        exp_logger=logger,
        cfg=cfg,
        client_types={0: "honest", 1: "honest", 2: "free_rider"},
        trim_ratio=0.34,
    )

    try:
        results = [
            (SimpleNamespace(cid="0"), _fit_res(0.0)),
            (SimpleNamespace(cid="1"), _fit_res(1.0)),
            (SimpleNamespace(cid="2"), _fit_res(100.0)),
        ]
        aggregated, _ = strategy.aggregate_fit(1, results, [])
        value = parameters_to_ndarrays(aggregated)[0][0]
    finally:
        logger.close()

    assert value == np.float32(1.0)
