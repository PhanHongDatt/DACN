"""
server.py — Flower server với custom FedAvg strategy tích hợp blockchain.
Sau mỗi vòng: thu quality_score từ clients → ghi lên contract → phân phối reward.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
)
from flwr.server.client_proxy import ClientProxy

from fl.blockchain import BlockchainBridge
from fl.config import ProjectConfig
from fl.logger import ExperimentLogger

logger = logging.getLogger(__name__)


class FLBlockchainStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg mở rộng: sau mỗi vòng aggregate, gọi blockchain để ghi
    đóng góp và phân phối reward cho P_honest.
    """

    def __init__(
        self,
        bridge: Optional[BlockchainBridge],
        exp_logger: ExperimentLogger,
        cfg: ProjectConfig,
        alpha: float,
        mean_data_size: float,
        client_types: Dict[int, str],
        **fedavg_kwargs
    ):
        super().__init__(**fedavg_kwargs)
        self.bridge         = bridge
        self.exp_logger     = exp_logger
        self.cfg            = cfg
        self.alpha          = alpha
        self.mean_data_size = mean_data_size
        self.client_types   = client_types
        # Cache global_accuracy từ aggregate_evaluate để dùng trong log
        self._last_accuracy: Dict[int, float] = {}
        self._pending_logs: Dict[int, List[dict]] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number vào config của mỗi client instruction."""
        instructions = super().configure_fit(server_round, parameters, client_manager)
        # FitIns(parameters, config) — dùng constructor trực tiếp, không dùng __class__
        return [
            (proxy, FitIns(
                parameters=fi.parameters,
                config={**fi.config, "round": server_round}
            ))
            for proxy, fi in instructions
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            logger.warning(f"Round {server_round}: no results, skipping")
            return None, {}

        # FedAvg aggregation chuẩn
        aggregated_params, agg_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # ── Thu thập metrics từ clients ───────────────────────
        n = self.cfg.fl.n_clients
        quality_scores = np.zeros(n, dtype=float)
        data_sizes     = np.zeros(n, dtype=int)
        client_metrics: Dict[int, dict] = {}

        for proxy, fit_res in results:
            cid = int(proxy.cid)
            qs  = float(fit_res.metrics.get("quality_score", 0.0))
            ds  = int(fit_res.metrics.get("data_size", 0))
            ct  = str(fit_res.metrics.get("client_type", "honest"))
            # Clamp quality score về [0, ∞) — không được âm
            quality_scores[cid]  = max(0.0, qs)
            data_sizes[cid]      = max(0, ds)
            client_metrics[cid]  = {"quality": max(0.0, qs), "data_size": ds, "type": ct}

        # ── Blockchain: ghi contribution + phân phối reward ───
        w_new_map:      Dict[int, float] = {}
        reputation_map: Dict[int, float] = {}

        if self.bridge:
            try:
                for cid, m in client_metrics.items():
                    self.bridge.submit_contribution(
                        cid, m["quality"], m["data_size"],
                        self.mean_data_size, server_round
                    )

                reputation_map = {}
                for cid in range(n):
                    rep, _ = self.bridge.get_reputation(cid)
                    reputation_map[cid] = rep

                w_new_map = self.bridge.filter_and_distribute(
                    n, quality_scores, data_sizes,
                    self.alpha, self.mean_data_size,
                    server_round, self.cfg.blockchain.pool_eth_per_round
                )
            except Exception as e:
                logger.error(f"Round {server_round} blockchain error: {e}")
                # Không crash FL — tiếp tục không có reward round này

        # ── Queue log mỗi client; flush sau aggregate_evaluate để đúng accuracy round ──
        pending_rows = []
        for cid in range(n):
            m      = client_metrics.get(cid, {"quality": 0.0, "data_size": 0, "type": "unknown"})
            w      = w_new_map.get(cid, 0.0)
            rep    = reputation_map.get(cid, 0.0)
            reward = w * self.cfg.blockchain.pool_eth_per_round if cid in w_new_map else 0.0

            pending_rows.append({
                "dataset": self.cfg.experiment.dataset,
                "scenario": self.cfg.experiment.scenario,
                "config": self.cfg.experiment.config_type,
                "alpha": self.alpha,
                "dirichlet_alpha": self.cfg.experiment.dirichlet_beta if self.cfg.experiment.scenario == "K3" else None,
                "round_num": server_round,
                "client_id": cid,
                "client_type": self.client_types.get(cid, "honest"),
                "quality": m["quality"],
                "data_size": m["data_size"],
                "w_new": w,
                "reputation": rep,
                "reward_eth": reward,
                "is_honest": cid in w_new_map,
            })
        self._pending_logs[server_round] = pending_rows

        n_honest = len(w_new_map)
        logger.info(
            f"Round {server_round} — honest={n_honest}/{n}  "
            f"α={self.alpha}  pool={self.cfg.blockchain.pool_eth_per_round} ETH"
        )
        return aggregated_params, agg_metrics

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
        failures
    ):
        """Cache global accuracy để log_round có thể dùng."""
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
            logger.info(f"Round {server_round} — global accuracy: {acc:.4f}")
        self._flush_round_logs(server_round, acc)
        return agg_loss, metrics
