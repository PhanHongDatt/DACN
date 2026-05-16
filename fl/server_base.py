"""
server_base.py — Unified Flower strategy cho hệ thống refactor.

Thay thế hoàn toàn 3 file legacy:
  - fl/server.py (FLBlockchainStrategy)
  - fl/server_csra.py (FLCSRAStrategy)
  - fl/server_trimmed.py (TrimmedMeanStrategy)

Thiết kế tách rõ:
  - aggregation_method (fedavg / trimmed / csra_dcd) — quyết định CÁCH HỢP NHẤT params
  - reward_policy (equal / data / quality / csra) — quyết định CÁCH PHÂN PHỐI reward
  - blockchain (optional) — chỉ làm AUDIT LAYER, không có logic thuật toán

Reference: docs/PLAN.md §7.5.
"""
from __future__ import annotations

import logging
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

from fl.aggregation_methods import apply_aggregation
from fl.blockchain import BlockchainBridge
from fl.config import ProjectConfig
from fl.logger import ExperimentLogger
from fl.reward_policies import apply_reward_policy

logger = logging.getLogger(__name__)


class FLUnifiedStrategy(fl.server.strategy.FedAvg):
    """
    Unified strategy: nhận aggregation_method + reward_policy, dispatch tới
    các hàm trong fl/aggregation_methods.py và fl/reward_policies.py.

    Semantic xử lý mỗi round (xem docs/PLAN.md §3.2):
      1. Thu thập params + metadata từ tất cả client tham gia.
      2. Apply aggregation_method → (aggregated_params, anomaly_mask).
      3. valid_clients = client_ids[not anomaly_mask].
      4. Submit contribution lên blockchain cho TẤT CẢ client (clients bị flag → quality=0).
      5. Đọc reputation từ blockchain (sau bước 4 để có giá trị cập nhật).
      6. Apply reward_policy chỉ trên valid_clients → reward dict.
      7. Audit reward lên blockchain (distribute_audit).
      8. Queue log mỗi client; flush sau aggregate_evaluate (để có global_accuracy).
    """

    def __init__(
        self,
        # Method selection
        aggregation_method: str,
        reward_policy: str,
        # Shared resources
        cfg: ProjectConfig,
        exp_logger: ExperimentLogger,
        client_types: Dict[int, str],
        mean_data_size: float,
        seed: int,
        attack_type: str = "clean",
        # Aggregation params
        trim_ratio: float = 0.1,
        mad_threshold: float = 3.0,
        min_honest_ratio: float = 0.5,
        # Reward policy params (CSRA)
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.2,
        # Blockchain (optional)
        bridge: Optional[BlockchainBridge] = None,
        # FedAvg base kwargs (min_fit_clients, ...)
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)

        # Method config
        self.aggregation_method = aggregation_method
        self.reward_policy = reward_policy
        self.trim_ratio = float(trim_ratio)
        self.mad_threshold = float(mad_threshold)
        self.min_honest_ratio = float(min_honest_ratio)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        # Resources
        self.cfg = cfg
        self.exp_logger = exp_logger
        self.client_types = client_types
        self.mean_data_size = float(mean_data_size)
        self.bridge = bridge
        self.seed = int(seed)
        self.attack_type = attack_type

        # Pending logs (flushed when aggregate_evaluate finalizes accuracy)
        self._last_accuracy: Dict[int, float] = {}
        self._pending_logs: Dict[int, List[dict]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Flower lifecycle hooks
    # ─────────────────────────────────────────────────────────────────────────

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number into client config (clients need it for attack logic)."""
        instructions = super().configure_fit(server_round, parameters, client_manager)
        return [
            (proxy, FitIns(
                parameters=fi.parameters,
                config={**fi.config, "round": server_round},
            ))
            for proxy, fi in instructions
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            logger.warning(f"Round {server_round}: no results, skipping")
            return None, {}

        # ── 1. Extract metadata ──────────────────────────────────────────────
        client_data = []
        for proxy, fit_res in results:
            cid = int(proxy.cid)
            client_data.append({
                "cid": cid,
                "params": parameters_to_ndarrays(fit_res.parameters),
                "num_examples": int(fit_res.num_examples),
                "quality_score": max(0.0, float(fit_res.metrics.get("quality_score", 0.0))),
                "data_size": max(0, int(fit_res.metrics.get("data_size", 0))),
                "anomaly_score": float(fit_res.metrics.get(
                    "anomaly_score", fit_res.metrics.get("variance", 0.0)
                )),
                "client_type_runtime": str(fit_res.metrics.get("client_type", "honest")),
            })
        client_data.sort(key=lambda d: d["cid"])
        cids = [d["cid"] for d in client_data]

        # ── 2. Aggregation ───────────────────────────────────────────────────
        agg_kwargs = {
            "weights": [d["num_examples"] for d in client_data],
        }
        if self.aggregation_method == "csra_dcd":
            agg_kwargs["anomaly_scores"] = [d["anomaly_score"] for d in client_data]
            agg_kwargs["mad_threshold"] = self.mad_threshold
            agg_kwargs["min_honest_ratio"] = self.min_honest_ratio
        elif self.aggregation_method == "trimmed":
            agg_kwargs["trim_ratio"] = self.trim_ratio

        aggregated_ndarrays, agg_meta = apply_aggregation(
            self.aggregation_method,
            [d["params"] for d in client_data],
            **agg_kwargs,
        )
        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        anomaly_mask: List[bool] = agg_meta["anomaly_mask"]
        robust_z_list: Optional[List[float]] = agg_meta["robust_z"]
        detection_method: Optional[str] = agg_meta["detection_method"]

        # Per-client detection info (chỉ valid với csra_dcd)
        detection_info: Dict[int, dict] = {}
        for i, d in enumerate(client_data):
            cid = d["cid"]
            detection_info[cid] = {
                "anomaly_score": d["anomaly_score"],
                "robust_z": (
                    float(robust_z_list[i])
                    if robust_z_list is not None and i < len(robust_z_list)
                    else None
                ),
                "is_anomaly": bool(anomaly_mask[i]),
                "detection_reason": (
                    detection_method if anomaly_mask[i] and detection_method
                    else "accepted"
                ),
            }

        # ── 3. Valid clients (post-filter) ───────────────────────────────────
        valid_data = [d for d, anom in zip(client_data, anomaly_mask) if not anom]
        valid_cids = [d["cid"] for d in valid_data]

        # ── 4. Submit contribution lên blockchain (audit cho TẤT CẢ) ────────
        reputations_map: Dict[int, float] = {}
        if self.bridge is not None:
            for i, d in enumerate(client_data):
                # Client bị flag → contribution = 0 (audit-friendly)
                qs = 0.0 if anomaly_mask[i] else d["quality_score"]
                ds = 0 if anomaly_mask[i] else d["data_size"]
                try:
                    self.bridge.submit_contribution(
                        d["cid"], qs, ds, self.mean_data_size, server_round,
                    )
                except Exception as e:
                    logger.warning(
                        f"Round {server_round}: submit_contribution failed for "
                        f"client {d['cid']}: {e}"
                    )

            # ── 5. Đọc reputation sau khi submit ──────────────────────────────
            for cid in range(self.cfg.fl.n_clients):
                try:
                    rep, _ = self.bridge.get_reputation(cid)
                    reputations_map[cid] = float(rep)
                except Exception:
                    reputations_map[cid] = 0.0
        else:
            # Không có blockchain → reputation default = 0.5 cho mọi participant
            # (đủ để CSRA reward formula chạy, tránh bias)
            for cid in range(self.cfg.fl.n_clients):
                reputations_map[cid] = 0.5

        # ── 6. Reward policy trên valid clients ──────────────────────────────
        reward_kwargs: Dict = {}
        if self.reward_policy == "data":
            reward_kwargs["data_sizes"] = [d["data_size"] for d in valid_data]
        elif self.reward_policy == "quality":
            reward_kwargs["quality_scores"] = [d["quality_score"] for d in valid_data]
        elif self.reward_policy == "csra":
            reward_kwargs["quality_scores"] = [d["quality_score"] for d in valid_data]
            reward_kwargs["data_sizes"] = [d["data_size"] for d in valid_data]
            reward_kwargs["reputations"] = [
                reputations_map.get(d["cid"], 0.0) for d in valid_data
            ]
            reward_kwargs["beta"] = self.beta
            reward_kwargs["gamma"] = self.gamma
            reward_kwargs["delta"] = self.delta
            reward_kwargs["mean_data_size"] = self.mean_data_size
            reward_kwargs["contrib_cfg"] = self.cfg.contrib

        pool_eth = self.cfg.blockchain.pool_eth_per_round
        rewards_map = apply_reward_policy(
            self.reward_policy,
            client_ids=valid_cids,
            total_reward=pool_eth,
            **reward_kwargs,
        )

        # w_new = fraction (rewards_map / pool_eth) — để tương thích với cột legacy
        w_new_map: Dict[int, float] = {
            cid: (r / pool_eth if pool_eth > 0 else 0.0)
            for cid, r in rewards_map.items()
        }

        # ── 7. Blockchain audit distribution ─────────────────────────────────
        if self.bridge is not None and rewards_map:
            try:
                self.bridge.distribute_audit(rewards_map, server_round)
            except Exception as e:
                logger.warning(
                    f"Round {server_round}: distribute_audit failed: {e}. "
                    f"Logging tiếp tục với reward đã tính."
                )

        # ── 8. Queue logs (flush sau aggregate_evaluate) ────────────────────
        pending_rows = []
        n_total = self.cfg.fl.n_clients
        participating = {d["cid"]: d for d in client_data}

        for cid in range(n_total):
            d = participating.get(cid)
            det = detection_info.get(cid, {
                "anomaly_score": None, "robust_z": None,
                "is_anomaly": False, "detection_reason": "not_participating",
            })

            if d is None:
                # Client không tham gia round này (lỗi hoặc dropout)
                row = {
                    "round_num": server_round,
                    "client_id": cid,
                    "client_type": self.client_types.get(cid, "honest"),
                    "quality": 0.0,
                    "data_size": 0,
                    "w_new": 0.0,
                    "reputation": reputations_map.get(cid, 0.0),
                    "reward_eth": 0.0,
                    "is_honest": False,
                    "anomaly_score": None,
                    "robust_z": None,
                    "is_anomaly": False,
                    "detection_reason": "not_participating",
                }
            else:
                row = {
                    "round_num": server_round,
                    "client_id": cid,
                    "client_type": self.client_types.get(cid, "honest"),
                    "quality": d["quality_score"],
                    "data_size": d["data_size"],
                    "w_new": w_new_map.get(cid, 0.0),
                    "reputation": reputations_map.get(cid, 0.0),
                    "reward_eth": rewards_map.get(cid, 0.0),
                    "is_honest": cid in valid_cids,
                    "anomaly_score": det["anomaly_score"],
                    "robust_z": det["robust_z"],
                    "is_anomaly": det["is_anomaly"],
                    "detection_reason": det["detection_reason"],
                }
            pending_rows.append(row)

        self._pending_logs[server_round] = pending_rows

        n_valid = len(valid_cids)
        n_filtered = len(client_data) - n_valid
        logger.info(
            f"Round {server_round} — agg={self.aggregation_method} "
            f"reward={self.reward_policy} valid={n_valid}/{len(client_data)} "
            f"filtered={n_filtered} pool={pool_eth} ETH"
        )
        return aggregated_params, {}

    # ─────────────────────────────────────────────────────────────────────────
    # Logging plumbing
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_round_logs(self, server_round: int, global_accuracy: Optional[float]):
        for row in self._pending_logs.pop(server_round, []):
            self.exp_logger.log_round(
                dataset=self.cfg.experiment.dataset,
                scenario=self.cfg.experiment.scenario,
                aggregation_method=self.aggregation_method,
                reward_policy=self.reward_policy,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                attack_type=self.attack_type,
                seed=self.seed,
                dirichlet_alpha=(
                    self.cfg.experiment.dirichlet_beta
                    if self.cfg.experiment.scenario == "K3"
                    else None
                ),
                global_accuracy=global_accuracy,
                **row,
            )

    def flush_pending_logs(self):
        """Force-flush bất kỳ pending log nào (gọi khi shutdown)."""
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
        weighted = [
            (res.num_examples, float(res.metrics["accuracy"]))
            for _, res in results
            if "accuracy" in res.metrics
        ]
        total = sum(n for n, _ in weighted)
        if total > 0:
            acc = float(sum(n * a for n, a in weighted) / total)
            self._last_accuracy[server_round] = acc
            logger.info(f"Round {server_round} — global accuracy: {acc:.4f}")
        self._flush_round_logs(server_round, acc)
        return agg_loss, metrics
