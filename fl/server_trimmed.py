"""
server_trimmed.py — TrimmedMean baseline strategy for robust FL aggregation.
This baseline intentionally does not use blockchain rewards.
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fl.config import ProjectConfig
from fl.logger import ExperimentLogger

logger = logging.getLogger(__name__)
warnings.warn(
    "fl.server_trimmed is legacy. Use fl.server_base.FLUnifiedStrategy with "
    "aggregation_method='trimmed' for schema-v2 experiments.",
    RuntimeWarning,
    stacklevel=2,
)


class TrimmedMeanStrategy(fl.server.strategy.FedAvg):
    """Coordinate-wise TrimmedMean aggregation with standard experiment logging."""

    def __init__(
        self,
        exp_logger: ExperimentLogger,
        cfg: ProjectConfig,
        client_types: Dict[int, str],
        trim_ratio: float = 0.1,
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)
        self.exp_logger = exp_logger
        self.cfg = cfg
        self.client_types = client_types
        self.trim_ratio = float(trim_ratio)
        self._last_accuracy: Dict[int, float] = {}
        self._pending_logs: Dict[int, List[dict]] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number into each client fit config."""
        instructions = super().configure_fit(server_round, parameters, client_manager)
        return [
            (proxy, FitIns(
                parameters=fi.parameters,
                config={**fi.config, "round": server_round}
            ))
            for proxy, fi in instructions
        ]

    def _aggregate_trimmed_mean(self, results: List[Tuple[ClientProxy, FitRes]]):
        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        n_clients = len(all_params)
        trim_k = int(n_clients * self.trim_ratio)
        if trim_k * 2 >= n_clients:
            logger.warning(
                "Trim ratio %.3f is too high for %s clients; falling back to mean",
                self.trim_ratio, n_clients,
            )
            trim_k = 0

        aggregated = []
        for layer_idx in range(len(all_params[0])):
            stacked = np.stack([params[layer_idx] for params in all_params], axis=0)
            if trim_k > 0:
                sorted_layer = np.sort(stacked, axis=0)
                selected = sorted_layer[trim_k:n_clients - trim_k]
            else:
                selected = stacked
            aggregated.append(np.mean(selected, axis=0))
        return aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            logger.warning("Round %s: no results, skipping", server_round)
            return None, {}

        aggregated_params = ndarrays_to_parameters(self._aggregate_trimmed_mean(results))

        n = self.cfg.fl.n_clients
        client_metrics: Dict[int, dict] = {}
        for proxy, fit_res in results:
            cid = int(proxy.cid)
            quality = max(0.0, float(fit_res.metrics.get("quality_score", 0.0)))
            data_size = max(0, int(fit_res.metrics.get("data_size", 0)))
            client_metrics[cid] = {"quality": quality, "data_size": data_size}

        pending_rows = []
        for cid in range(n):
            client_type = self.client_types.get(cid, "honest")
            metrics = client_metrics.get(cid, {"quality": 0.0, "data_size": 0})
            pending_rows.append({
                "dataset": self.cfg.experiment.dataset,
                "scenario": self.cfg.experiment.scenario,
                "config": self.cfg.experiment.config_type,
                "alpha": 0.0,
                "dirichlet_alpha": self.cfg.experiment.dirichlet_beta if self.cfg.experiment.scenario == "K3" else None,
                "round_num": server_round,
                "client_id": cid,
                "client_type": client_type,
                "quality": metrics["quality"],
                "data_size": metrics["data_size"],
                "w_new": 0.0,
                "reputation": 0.0,
                "reward_eth": 0.0,
                "is_honest": client_type == "honest",
            })
        self._pending_logs[server_round] = pending_rows

        logger.info(
            "Round %s — TrimmedMean aggregated %s clients with trim_ratio=%.3f",
            server_round, len(results), self.trim_ratio,
        )
        return aggregated_params, {}

    def _flush_round_logs(self, server_round: int, global_accuracy: Optional[float]):
        for row in self._pending_logs.pop(server_round, []):
            self.exp_logger.log_round(**row, global_accuracy=global_accuracy)

    def flush_pending_logs(self):
        for round_num in sorted(list(self._pending_logs.keys())):
            self._flush_round_logs(round_num, self._last_accuracy.get(round_num))

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ):
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        acc = None
        weighted_results = [
            (res.num_examples, float(res.metrics["accuracy"]))
            for _, res in results
            if "accuracy" in res.metrics
        ]
        weighted_total = sum(num_examples for num_examples, _ in weighted_results)
        if weighted_total > 0:
            acc = float(sum(num_examples * accuracy for num_examples, accuracy in weighted_results) / weighted_total)
            self._last_accuracy[server_round] = acc
            logger.info("Round %s — global accuracy: %.4f", server_round, acc)
        self._flush_round_logs(server_round, acc)
        return agg_loss, metrics
