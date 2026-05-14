"""
server_csra.py — CSRA-inspired FL Strategy with advanced anomaly detection.
Implements CSRA-DCD (Detection) and CSRA-QMS (Quality Management) to ensure 
system robustness against free-riders and malicious updates.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, FitIns
from flwr.server.client_proxy import ClientProxy
from fl.blockchain import BlockchainBridge
from fl.config import ProjectConfig
from fl.logger import ExperimentLogger

logger = logging.getLogger(__name__)

class FLCSRAStrategy(fl.server.strategy.FedAvg):
    """
    Advanced Federated Learning Strategy using CSRA principles.
    
    Features:
        - Variance-based Anomaly Detection (DCD): Detects noise-based updates.
        - Bid-aware Reward Distribution: Integrates economic incentives.
        - Reputation-gated Selection: Uses on-chain history for participation logic.
    """
    def __init__(
        self,
        bridge: Optional[BlockchainBridge],
        exp_logger: ExperimentLogger,
        cfg: ProjectConfig,
        alpha: float,
        mean_data_size: float,
        client_types: Dict[int, str],
        dynamic_alpha: bool = False,
        mad_threshold: float = 3.0,
        min_honest_ratio: float = 0.5,
        mad_epsilon: float = 1e-8,
        **fedavg_kwargs
    ):
        super().__init__(**fedavg_kwargs)
        self.bridge = bridge
        self.exp_logger = exp_logger
        self.cfg = cfg
        self.alpha = alpha
        self.base_alpha = alpha
        self.mean_data_size = mean_data_size
        self.client_types = client_types
        self.dynamic_alpha = dynamic_alpha
        self.mad_threshold = float(mad_threshold)
        self.min_honest_ratio = float(min_honest_ratio)
        self.mad_epsilon = float(mad_epsilon)
        self._last_accuracy: Dict[int, float] = {}
        self._pending_logs: Dict[int, List[dict]] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number into client fit config."""
        instructions = super().configure_fit(server_round, parameters, client_manager)
        return [
            (proxy, FitIns(
                parameters=fi.parameters,
                config={**fi.config, "round": server_round}
            ))
            for proxy, fi in instructions
        ]

    def _robust_z_scores(self, scores: np.ndarray) -> Tuple[np.ndarray, str]:
        """Return two-sided robust z-scores using MAD with a stable fallback."""
        if scores.size == 0:
            return np.array([], dtype=float), "empty"

        median = float(np.median(scores))
        abs_dev = np.abs(scores - median)
        mad = float(np.median(abs_dev))

        if mad >= self.mad_epsilon:
            return abs_dev / (1.4826 * mad), "mad"

        max_dev = float(np.max(abs_dev)) if abs_dev.size else 0.0
        if max_dev < self.mad_epsilon:
            return np.zeros_like(scores, dtype=float), "mad_zero_all_equal"

        fallback_scale = max(float(np.mean(abs_dev)), self.mad_epsilon)
        return abs_dev / (1.4826 * fallback_scale), "mean_abs_dev_fallback"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates model updates using CSRA-DCD and Reputation-Weighted Aggregation (RWA).
        """
        if not results:
            return None, {}

        if self.dynamic_alpha:
            total_rounds = max(self.cfg.fl.n_rounds, 1)
            progress = server_round / total_rounds
            current_alpha = 0.3 + (0.5 * progress)
            self.alpha = float(np.clip(current_alpha, 0.0, 1.0))
            logger.info(f"Round {server_round}: Dynamic Alpha set to {self.alpha:.2f}")
        else:
            self.alpha = self.base_alpha

        # --- PHASE 1: CSRA-DCD (STATISTICAL ANOMALY DETECTION) ---
        scores = []
        cids = []
        for proxy, fit_res in results:
            score = float(fit_res.metrics.get("anomaly_score", fit_res.metrics.get("variance", 0.0)))
            scores.append(score)
            cids.append(int(proxy.cid))
        
        score_arr = np.asarray(scores, dtype=float)
        robust_z, detection_method = self._robust_z_scores(score_arr)
        dcd_anomaly_cids = set()
        reputation_filtered_cids = set()
        detection_info: Dict[int, dict] = {}

        # --- OPTIMIZATION 2: REPUTATION-WEIGHTED FILTERING ---
        # We fetch reputation before aggregation to influence the selection.
        reputation_map: Dict[int, float] = {}
        honest_results: List[Tuple[ClientProxy, FitRes]] = []

        for i, (proxy, fit_res) in enumerate(results):
            cid = cids[i]
            z = float(robust_z[i]) if i < len(robust_z) else 0.0

            # Fetch on-chain reputation for gating
            rep = 0.0
            is_honest_on_chain = True
            if self.bridge:
                rep, is_honest_on_chain = self.bridge.get_reputation(cid)
            reputation_map[cid] = rep

            is_anomaly = z > self.mad_threshold
            reason = "accepted"
            if is_anomaly:
                dcd_anomaly_cids.add(cid)
                reason = detection_method
            if not is_honest_on_chain:
                reputation_filtered_cids.add(cid)
                reason = "low_reputation" if reason == "accepted" else f"{reason}+low_reputation"

            detection_info[cid] = {
                "anomaly_score": scores[i],
                "robust_z": z,
                "is_anomaly": is_anomaly,
                "detection_reason": reason,
            }

            if is_anomaly or not is_honest_on_chain:
                logger.warning(f"Round {server_round}: Client {cid} filtered. Reason: {reason}")
            else:
                honest_results.append((proxy, fit_res))

        min_honest_clients = max(1, math.ceil(len(results) * self.min_honest_ratio))
        if len(honest_results) < min_honest_clients and dcd_anomaly_cids:
            logger.warning(
                "Round %s: DCD flagged too many clients (%s/%s honest left), "
                "falling back to accept DCD-filtered updates for this round",
                server_round, len(honest_results), len(results),
            )
            for cid in sorted(dcd_anomaly_cids):
                info = detection_info[cid]
                info["is_anomaly"] = False
                info["detection_reason"] = "fallback_accept_all"
            dcd_anomaly_cids.clear()
            honest_results = [
                (proxy, fit_res)
                for proxy, fit_res in results
                if int(proxy.cid) not in reputation_filtered_cids
            ]

        # --- PHASE 2: REPUTATION-WEIGHTED AGGREGATION (RWA) ---
        # Instead of standard FedAvg, we weight updates by (data_size * reputation).
        if not honest_results:
            logger.warning(f"Round {server_round}: No honest clients after filtering.")
            return None, {}

        # Custom weighting for aggregation
        weighted_results = []
        for proxy, fit_res in honest_results:
            cid = int(proxy.cid)
            rep = reputation_map.get(cid, 0.1) # Minimum base weight
            # Effective data size = actual size * reputation
            # This prioritizes updates from consistently high-quality nodes.
            fit_res.num_examples = int(fit_res.num_examples * (1.0 + rep))
            weighted_results.append((proxy, fit_res))

        aggregated_params, agg_metrics = super().aggregate_fit(
            server_round, weighted_results, failures
        )

        # --- PHASE 3: CSRA-QMS (REWARD CALCULATION) ---
        n = self.cfg.fl.n_clients
        quality_scores = np.zeros(n, dtype=float)
        data_sizes = np.zeros(n, dtype=int)
        excluded_reward_cids = dcd_anomaly_cids | reputation_filtered_cids
        
        client_metrics: Dict[int, dict] = {}
        for proxy, fit_res in results:
            cid = int(proxy.cid)
            qs = float(fit_res.metrics.get("quality_score", 0.0))
            ds = int(fit_res.metrics.get("data_size", 0))
            
            if cid in excluded_reward_cids:
                qs = 0.0
                ds = 0
                
            quality_scores[cid] = qs
            data_sizes[cid] = ds
            client_metrics[cid] = {"quality": qs, "data_size": ds}

        # Blockchain Reward Distribution
        w_new_map: Dict[int, float] = {}
        if self.bridge:
            try:
                for cid, m in client_metrics.items():
                    self.bridge.submit_contribution(
                        cid, m["quality"], m["data_size"],
                        self.mean_data_size, server_round
                    )

                w_new_map = self.bridge.filter_and_distribute(
                    n, quality_scores, data_sizes,
                    self.alpha, self.mean_data_size,
                    server_round, self.cfg.blockchain.pool_eth_per_round,
                    exclude_client_indices=sorted(excluded_reward_cids),
                )
            except Exception as e:
                logger.error(f"Round {server_round} blockchain error: {e}")

        # Queue logging until aggregate_evaluate provides accuracy for this round.
        pending_rows = []
        for cid in range(n):
            m = client_metrics.get(cid, {"quality": 0.0, "data_size": 0})
            w = w_new_map.get(cid, 0.0)
            rep = reputation_map.get(cid, 0.0)
            reward = w * self.cfg.blockchain.pool_eth_per_round if cid in w_new_map else 0.0
            detection = detection_info.get(cid, {
                "anomaly_score": None,
                "robust_z": None,
                "is_anomaly": False,
                "detection_reason": "missing_result",
            })

            pending_rows.append({
                "dataset": self.cfg.experiment.dataset,
                "scenario": self.cfg.experiment.scenario,
                "config": self.cfg.experiment.config_type + "-CSRA-Opt",
                "alpha": self.alpha,
                "alpha_runtime": self.alpha,
                "dirichlet_alpha": self.cfg.experiment.dirichlet_beta if self.cfg.experiment.scenario == "K3" else None,
                "round_num": server_round,
                "client_id": cid,
                "client_type": self.client_types.get(cid, "honest"),
                "quality": m["quality"],
                "data_size": m["data_size"],
                "w_new": w,
                "reputation": rep,
                "reward_eth": reward,
                "is_honest": cid not in excluded_reward_cids,
                "anomaly_score": detection["anomaly_score"],
                "robust_z": detection["robust_z"],
                "is_anomaly": detection["is_anomaly"],
                "detection_reason": detection["detection_reason"],
            })
        self._pending_logs[server_round] = pending_rows

        return aggregated_params, agg_metrics

    def _flush_round_logs(self, server_round: int, global_accuracy: Optional[float]):
        for row in self._pending_logs.pop(server_round, []):
            self.exp_logger.log_round(**row, global_accuracy=global_accuracy)

    def flush_pending_logs(self):
        for round_num in sorted(list(self._pending_logs.keys())):
            self._flush_round_logs(round_num, self._last_accuracy.get(round_num))

    def aggregate_evaluate(self, server_round, results, failures):
        """Captures global accuracy for logging purposes."""
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
        self._flush_round_logs(server_round, acc)
        return agg_loss, metrics
