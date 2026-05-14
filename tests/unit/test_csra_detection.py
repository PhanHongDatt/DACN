import numpy as np

from fl.config import ProjectConfig
from fl.logger import ExperimentLogger
from fl.server_csra import FLCSRAStrategy


def _strategy(tmp_path):
    logger = ExperimentLogger("csra", str(tmp_path))
    strategy = FLCSRAStrategy(
        bridge=None,
        exp_logger=logger,
        cfg=ProjectConfig(),
        alpha=0.5,
        mean_data_size=1.0,
        client_types={},
    )
    return strategy, logger


def test_mad_zero_fallback_does_not_flag_small_normal_spread(tmp_path):
    strategy, logger = _strategy(tmp_path)
    try:
        scores = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.13])
        z_scores, method = strategy._robust_z_scores(scores)
    finally:
        logger.close()

    assert method == "mean_abs_dev_fallback"
    assert float(np.max(z_scores)) < strategy.mad_threshold


def test_mad_zero_fallback_still_detects_clear_outlier(tmp_path):
    strategy, logger = _strategy(tmp_path)
    try:
        scores = np.array([0.1] * 9 + [10.0])
        z_scores, method = strategy._robust_z_scores(scores)
    finally:
        logger.close()

    assert method == "mean_abs_dev_fallback"
    assert z_scores[-1] > strategy.mad_threshold
