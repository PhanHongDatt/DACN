"""
server_csra.py — CSRA-inspired FL Strategy with advanced anomaly detection.
Implements CSRA-DCD (Detection) and CSRA-QMS (Quality Management) to ensure 
system robustness against free-riders and malicious updates.
"""
import logging
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
        **fedavg_kwargs
    ):
        super().__init__(**fedavg_kwargs)
        self.bridge = bridge
        self.exp_logger = exp_logger
        self.cfg = cfg
        self.alpha = alpha
        self.mean_data_size = mean_data_size
        self.client_types = client_types
        self._last_accuracy: Dict[int, float] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates model updates after performing CSRA-DCD detection.
        """
        if not results:
            return None, {}

        # --- PHASE 1: CSRA-DCD (COARSE-GRAINED ANOMALY DETECTION) ---
        # We analyze the variance of model updates to catch free-riders 
        # who send random noise instead of real training gradients.
        variances = []
        cids = []
        for proxy, fit_res in results:
            v = float(fit_res.metrics.get("variance", 0.0))
            variances.append(v)
            cids.append(int(proxy.cid))
        
        # Calculate statistical thresholds using Z-score logic.
        v_mean = np.mean(variances)
        v_std = np.std(variances)
        honest_results = []
        malicious_cids = []

        for i, (proxy, fit_res) in enumerate(results):
            cid = cids[i]
            v = variances[i]
            # Threshold: Updates with variance > 2 standard deviations from mean 
            # are flagged as anomalies (noise injection).
            if v > v_mean + 2 * v_std and v_std > 1e-6:
                logger.warning(f"Round {server_round}: Client {cid} flagged as MALICIOUS (Statistical Variance Outlier)")
                malicious_cids.append(cid)
            else:
                honest_results.append((proxy, fit_res))

        # --- PHASE 2: MODEL AGGREGATION (Filtered) ---
        # Only honest (non-flagged) updates are aggregated into the global model.
        aggregated_params, agg_metrics = super().aggregate_fit(
            server_round, honest_results, failures
        )

        # --- PHASE 3: CSRA-QMS (REWARD CALCULATION) ---
        n = self.cfg.fl.n_clients
        quality_scores = np.zeros(n, dtype=float)
        data_sizes = np.zeros(n, dtype=int)
        
        client_metrics: Dict[int, dict] = {}
        for proxy, fit_res in results:
            cid = int(proxy.cid)
            qs = float(fit_res.metrics.get("quality_score", 0.0))
            ds = int(fit_res.metrics.get("data_size", 0))
            
            # If flagged by DCD, zero out the quality score to deny reward.
            if cid in malicious_cids:
                qs = 0.0
                
            quality_scores[cid] = qs
            data_sizes[cid] = ds
            client_metrics[cid] = {"quality": qs, "data_size": ds}

        # Submit metrics to the Blockchain for reputation and reward logic.
        w_new_map: Dict[int, float] = {}
        reputation_map: Dict[int, float] = {}

        if self.bridge:
            try:
                for cid, m in client_metrics.items():
                    # Record the contribution on-chain.
                    self.bridge.submit_contribution(
                        cid, m["quality"], m["data_size"],
                        self.mean_data_size, server_round
                    )

                for cid in range(n):
                    rep, _ = self.bridge.get_reputation(cid)
                    reputation_map[cid] = rep

                # Distribute rewards based on the filtered honest pool.
                w_new_map = self.bridge.filter_and_distribute(
                    n, quality_scores, data_sizes,
                    self.alpha, self.mean_data_size,
                    server_round, self.cfg.blockchain.pool_eth_per_round
                )
            except Exception as e:
                logger.error(f"Round {server_round} blockchain integration error: {e}")

        # Logging round details for comparative analysis.
        global_acc = self._last_accuracy.get(server_round - 1)
        for cid in range(n):
            m = client_metrics.get(cid, {"quality": 0.0, "data_size": 0})
            w = w_new_map.get(cid, 0.0)
            rep = reputation_map.get(cid, 0.0)
            reward = w * self.cfg.blockchain.pool_eth_per_round if cid in w_new_map else 0.0

            self.exp_logger.log_round(
                dataset=self.cfg.experiment.dataset,
                scenario=self.cfg.experiment.scenario,
                config=self.cfg.experiment.config_type + "-CSRA",
                alpha=self.alpha,
                round_num=server_round,
                client_id=cid,
                client_type=self.client_types.get(cid, "honest"),
                quality=m["quality"],
                data_size=m["data_size"],
                w_new=w,
                reputation=rep,
                reward_eth=reward,
                is_honest=(cid not in malicious_cids and cid in w_new_map),
                global_accuracy=global_acc
            )

        return aggregated_params, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Captures global accuracy for logging purposes."""
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics and "accuracy" in metrics:
            self._last_accuracy[server_round] = float(metrics["accuracy"])
        return agg_loss, metrics
