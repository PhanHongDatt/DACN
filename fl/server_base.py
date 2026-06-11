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

from fl.aggregation_methods import apply_aggregation, compute_update_features
from fl.blockchain import BlockchainBridge
from fl.config import ProjectConfig
from fl.logger import ExperimentLogger
from fl.normalization import hybrid_normalize
from fl.reward_policies import apply_reward_policy

logger = logging.getLogger(__name__)


class FLUnifiedStrategy(fl.server.strategy.FedAvg):
    """
    Unified strategy: nhận aggregation_method + reward_policy, dispatch tới
    các hàm trong fl/aggregation_methods.py và fl/reward_policies.py.

    Semantic xử lý mỗi round (xem docs/PLAN.md §3.2):
      1. Thu thập params + metadata từ tất cả client tham gia.
      2. Apply aggregation_method → aggregated_params + detection metadata.
      3. valid_clients = client_ids[not anomaly_mask] cho aggregation.
      4. reward_eligible = valid_clients trừ reward_blocked và low_reputation.
      5. Submit contribution lên blockchain cho TẤT CẢ client
         (reward-blocked clients → quality=0, data_size=0).
      6. Đọc reputation từ blockchain (sau bước 5 để có giá trị cập nhật).
      7. Apply reward_policy chỉ trên reward_eligible → reward dict.
      8. Audit reward lên blockchain (distribute_audit).
      9. Queue log mỗi client; flush sau aggregate_evaluate (để có global_accuracy/global_loss).
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
        server_data_sizes: Optional[Dict[int, int]] = None,
        # Aggregation params
        trim_ratio: float = 0.1,
        mad_threshold: float = 3.0,
        cosine_threshold: float = -0.8,
        direction_min_norm_z: float = 0.0,
        min_honest_ratio: float = 0.5,
        fallback_hard_z: float = 6.0,
        suspicion_decay: float = 0.6,
        suspicion_threshold: float = 1.0,
        low_quality_z_threshold: float = 2.0,
        low_quality_suspicion: float = 0.5,
        zero_data_suspicion: float = 1.0,
        anomaly_suspicion: float = 0.8,
        authenticity_suspicion: float = 1.0,
        low_authenticity_threshold: float = 1.5,
        high_update_norm_z_threshold: float = 4.0,
        inefficient_update_suspicion: float = 1.0,
        # FedLAW params
        alpha_law: float = 0.1,
        beta_law: float = 0.01,
        sparsity_s: int = 10,
        capping_t: float = 0.2,
        # Reward policy params (CSRA)
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.2,
        # Blockchain (optional)
        bridge: Optional[BlockchainBridge] = None,
        # FedAvg base kwargs (min_fit_clients, ...)
        **fedavg_kwargs,
    ):
        initial_parameters = fedavg_kwargs.get("initial_parameters")
        super().__init__(**fedavg_kwargs)

        # Method config
        self.aggregation_method = aggregation_method
        self.reward_policy = reward_policy
        self.trim_ratio = float(trim_ratio)
        self.mad_threshold = float(mad_threshold)
        self.cosine_threshold = float(cosine_threshold)
        self.direction_min_norm_z = float(direction_min_norm_z)
        self.min_honest_ratio = float(min_honest_ratio)
        self.fallback_hard_z = float(fallback_hard_z)
        self.suspicion_decay = float(suspicion_decay)
        self.suspicion_threshold = float(suspicion_threshold)
        self.low_quality_z_threshold = float(low_quality_z_threshold)
        self.low_quality_suspicion = float(low_quality_suspicion)
        self.zero_data_suspicion = float(zero_data_suspicion)
        self.anomaly_suspicion = float(anomaly_suspicion)
        self.authenticity_suspicion = float(authenticity_suspicion)
        self.low_authenticity_threshold = float(low_authenticity_threshold)
        self.high_update_norm_z_threshold = float(high_update_norm_z_threshold)
        self.inefficient_update_suspicion = float(inefficient_update_suspicion)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        # FedLAW params
        self.alpha_law = float(alpha_law)
        self.beta_law = float(beta_law)
        self.sparsity_s = int(sparsity_s)
        self.capping_t = float(capping_t)

        # Resources
        self.cfg = cfg
        self.exp_logger = exp_logger
        self.client_types = client_types
        self.mean_data_size = float(mean_data_size)
        self.server_data_sizes = {
            int(cid): max(0, int(size))
            for cid, size in (server_data_sizes or {}).items()
        }
        self.bridge = bridge
        self.seed = int(seed)
        self.attack_type = attack_type

        # FedLAW state
        self._fedlaw_w_k: Optional[np.ndarray] = None
        self._fedlaw_G_k: Optional[List[dict]] = None  # Store phase 1 results
        self._fedlaw_base_params: Optional[List[np.ndarray]] = None

        # Pending logs (flushed when aggregate_evaluate finalizes accuracy)
        self._last_accuracy: Dict[int, float] = {}
        self._last_loss: Dict[int, float] = {}
        self._pending_logs: Dict[int, List[dict]] = {}
        self._suspicion_scores: Dict[int, float] = {}
        self._current_global_ndarrays: Optional[List[np.ndarray]] = None
        if initial_parameters is not None:
            self.set_current_global_parameters(initial_parameters)

    # ─────────────────────────────────────────────────────────────────────────
    # Flower lifecycle hooks
    # ─────────────────────────────────────────────────────────────────────────

    def set_current_global_parameters(self, parameters) -> None:
        """Store the global parameters used for the next fit round."""
        if isinstance(parameters, Parameters):
            ndarrays = parameters_to_ndarrays(parameters)
        else:
            ndarrays = parameters
        self._current_global_ndarrays = [
            np.array(layer, copy=True) for layer in ndarrays
        ]

    def _low_quality_mask(self, client_data: List[dict]) -> List[bool]:
        """Flag lower-tail quality outliers without using labels/client type."""
        if not client_data:
            return []

        quality = np.asarray(
            [float(d.get("quality_score", 0.0)) for d in client_data],
            dtype=float,
        )
        if float(np.nanmax(quality)) <= 1e-12:
            return [False] * len(client_data)

        median = float(np.median(quality))
        abs_dev = np.abs(quality - median)
        mad = float(np.median(abs_dev))
        if mad <= 1e-12:
            fallback_scale = max(float(np.mean(abs_dev)), 1e-12)
        else:
            fallback_scale = 1.4826 * mad

        lower_z = np.maximum(0.0, median - quality) / fallback_scale
        return [
            bool(z >= self.low_quality_z_threshold and q < median)
            for z, q in zip(lower_z, quality)
        ]

    @staticmethod
    def _upper_tail_robust_z(values: np.ndarray) -> np.ndarray:
        """Robust upper-tail z-score based on MAD/mean absolute deviation."""
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array([], dtype=float)

        median = float(np.nanmedian(values))
        abs_dev = np.abs(values - median)
        mad = float(np.nanmedian(abs_dev))
        if mad <= 1e-12:
            scale = max(float(np.nanmean(abs_dev)), 1e-12)
        else:
            scale = 1.4826 * mad
        return np.maximum(0.0, values - median) / scale

    def _inefficient_update_mask(
        self,
        client_data: List[dict],
    ) -> tuple[List[bool], List[float]]:
        """
        Flag reward-risk updates with extreme raw norm and mediocre quality.

        This is a soft reward-quarantine signal only. It does not change the
        aggregation filter because large raw updates can be honest in Non-IID.
        """
        if not client_data or self.high_update_norm_z_threshold <= 0.0:
            return [False] * len(client_data), [0.0] * len(client_data)

        norms = np.asarray(
            [float(d.get("raw_update_norm") or 0.0) for d in client_data],
            dtype=float,
        )
        qualities = np.asarray(
            [float(d.get("quality_score", 0.0)) for d in client_data],
            dtype=float,
        )
        if float(np.nanmax(norms)) <= 1e-12:
            return [False] * len(client_data), [0.0] * len(client_data)

        z_scores = self._upper_tail_robust_z(norms)
        median_quality = float(np.nanmedian(qualities))
        mask = [
            bool(
                z >= self.high_update_norm_z_threshold
                and q < median_quality
            )
            for z, q in zip(z_scores, qualities)
        ]
        return mask, [float(z) for z in z_scores]

    def _authenticity_mask(self, client_data: List[dict]) -> List[bool]:
        """Flag lower-tail authenticity (STD) outliers (Free-riders)."""
        if not client_data:
            return []

        auths = np.asarray(
            [float(d.get("authenticity_score", 0.0)) for d in client_data],
            dtype=float,
        )
        if float(np.nanmax(auths)) <= 1e-12:
            return [False] * len(client_data)

        median = float(np.median(auths))
        abs_dev = np.abs(auths - median)
        mad = float(np.median(abs_dev))
        if mad <= 1e-12:
            fallback_scale = max(float(np.mean(abs_dev)), 1e-12)
        else:
            fallback_scale = 1.4826 * mad

        # Free riders typically have very low variance (just the artificial noise)
        lower_z = np.maximum(0.0, median - auths) / fallback_scale
        return [
            bool(z >= self.low_authenticity_threshold and a < median)
            for z, a in zip(lower_z, auths)
        ]

    def _data_normalized_update_score(
        self,
        update_norm: float,
        data_size: int,
    ) -> float:
        """
        Normalize update norm by relative data commitment for DCD scoring.

        In strong Non-IID with full local epochs, clients with more samples do
        more optimizer steps and naturally produce larger raw update norms.
        Using the raw norm alone can therefore flag high-volume honest clients.
        The normalized score keeps the detector focused on update magnitude per
        committed data volume while zero-data participants remain handled by the
        explicit data-commitment gate.
        """
        norm = max(0.0, float(update_norm))
        if data_size <= 0 or self.mean_data_size <= 0:
            return norm

        relative_size = max(float(data_size) / self.mean_data_size, 1e-12)
        return norm / relative_size

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number into client config (clients need it for attack logic)."""
        self.set_current_global_parameters(parameters)
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
            reported_data_size = max(0, int(fit_res.metrics.get("data_size", 0)))
            server_known_size = self.server_data_sizes.get(cid)
            effective_data_size = (
                server_known_size if server_known_size is not None else reported_data_size
            )
            client_data.append({
                "cid": cid,
                "params": parameters_to_ndarrays(fit_res.parameters),
                "num_examples": int(fit_res.num_examples),
                "quality_score": max(0.0, float(fit_res.metrics.get("quality_score", 0.0))),
                "data_size": max(0, int(effective_data_size)),
                "reported_data_size": reported_data_size,
                "server_known_data_size": server_known_size,
                "anomaly_score": float(fit_res.metrics.get(
                    "anomaly_score", fit_res.metrics.get("variance", 0.0)
                )),
                "client_type_runtime": str(fit_res.metrics.get("client_type", "honest")),
            })
        client_data.sort(key=lambda d: d["cid"])
        cids = [d["cid"] for d in client_data]
        commitment_mask: List[bool] = [
            bool(
                d["num_examples"] > 0
                and (
                    int(d.get("reported_data_size", 0)) <= 0
                    or int(d["data_size"]) <= 0
                )
            )
            for d in client_data
        ]
        data_size_mismatch_mask: List[bool] = [
            bool(
                d.get("server_known_data_size") is not None
                and int(d.get("reported_data_size", 0)) != int(d["data_size"])
            )
            for d in client_data
        ]

        # Server-side update features. This avoids trusting client-reported
        # anomaly_score and enables direction-based detection for sign-flip.
        if self._current_global_ndarrays is not None:
            try:
                update_features = compute_update_features(
                    [d["params"] for d in client_data],
                    self._current_global_ndarrays,
                )
                for d, norm, std, cosine in zip(
                    client_data,
                    update_features["update_norms"],
                    update_features["update_stds"],
                    update_features["cosine_to_reference"],
                ):
                    d["raw_update_norm"] = float(norm)
                    d["anomaly_score"] = self._data_normalized_update_score(
                        update_norm=float(norm),
                        data_size=int(d.get("data_size", 0)),
                    )
                    d["normalized_update_score"] = float(d["anomaly_score"])
                    d["authenticity_score"] = float(std)
                    d["anomaly_score_source"] = "server_data_normalized"
                    d["cosine_to_reference"] = float(cosine)
            except Exception as exc:
                logger.warning(
                    "Round %s: server-side update feature computation failed: %s. "
                    "Falling back to client-reported anomaly_score.",
                    server_round,
                    exc,
                )
                for d in client_data:
                    d["raw_update_norm"] = None
                    d["normalized_update_score"] = d["anomaly_score"]
                    d["anomaly_score_source"] = "client_reported"
                    d["cosine_to_reference"] = None
        else:
            logger.warning(
                "Round %s: current global params unavailable; using client-reported "
                "anomaly_score.",
                server_round,
            )
            for d in client_data:
                d["raw_update_norm"] = None
                d["normalized_update_score"] = d["anomaly_score"]
                d["anomaly_score_source"] = "client_reported"
                d["cosine_to_reference"] = None

        # ── 2. Aggregation ───────────────────────────────────────────────────
        agg_kwargs = {
            "weights": [d["num_examples"] for d in client_data],
        }

        # FEDLAW logic: 2-phase protocol
        if self.aggregation_method == "fedlaw":
            is_trial_phase = (server_round % 2 == 1)
            n_clients = self.cfg.fl.n_clients

            # Initialize weights if first round
            if self._fedlaw_w_k is None:
                self._fedlaw_w_k = np.ones(n_clients) / n_clients

            if is_trial_phase:
                # Giai đoạn 1 (Trial): Thu thập G_k, phát mô hình thử nghiệm theta_tilde
                logger.info(f"Round {server_round}: FedLAW Phase 1 (Trial)")
                self._fedlaw_G_k = client_data
                self._fedlaw_base_params = [np.array(p, copy=True) for p in self._current_global_ndarrays]

                # theta_tilde = theta_k - alpha * sum(w_k * g_i)
                # Tương đương FedAvg với weights = w_k
                aggregated_ndarrays, _ = apply_aggregation(
                    "fedavg",
                    [d["params"] for d in client_data],
                    weights=self._fedlaw_w_k,
                )
                agg_meta = {
                    "anomaly_mask": [False] * len(client_data),
                    "robust_z": None,
                    "detection_method": "fedlaw_trial",
                    "reward_block_mask": [True] * len(client_data),  # Không thưởng ở trial round
                }
            else:
                # Giai đoạn 2 (Final): Thu thập G_tilde, f_tilde; tối ưu w_k+1
                logger.info(f"Round {server_round}: FedLAW Phase 2 (Final)")
                if self._fedlaw_G_k is None or self._fedlaw_base_params is None:
                    logger.warning("FedLAW: missing Phase 1 data, falling back to FedAvg")
                    aggregated_ndarrays, agg_meta = apply_aggregation(
                        "fedavg", [d["params"] for d in client_data], **agg_kwargs
                    )
                else:
                    agg_kwargs.update({
                        "trial_params": [d["params"] for d in client_data],
                        "base_params": self._fedlaw_base_params,
                        "w_k": self._fedlaw_w_k,
                        "local_losses": np.array([d["quality_score"] for d in client_data]), # f_tilde
                        "alpha": self.cfg.fedlaw.alpha,
                        "beta_law": self.cfg.fedlaw.beta,
                        "sparsity_s": self.cfg.fedlaw.sparsity_s,
                        "capping_t": self.cfg.fedlaw.capping_t,
                    })
                    # aggregation_fedlaw dùng G_k (lưu từ Phase 1)
                    aggregated_ndarrays, agg_meta = apply_aggregation(
                        "fedlaw",
                        [d["params"] for d in self._fedlaw_G_k],
                        **agg_kwargs
                    )
                    # Cập nhật w_k cho epoch tiếp theo
                    self._fedlaw_w_k = agg_meta["w_next"]

                # Reset state
                self._fedlaw_G_k = None
                self._fedlaw_base_params = None

        elif self.aggregation_method == "csra_dcd":
            agg_kwargs["anomaly_scores"] = [d["anomaly_score"] for d in client_data]
            if all(d["cosine_to_reference"] is not None for d in client_data):
                agg_kwargs["update_cosines"] = [
                    d["cosine_to_reference"] for d in client_data
                ]
            agg_kwargs["mad_threshold"] = self.mad_threshold
            agg_kwargs["cosine_threshold"] = self.cosine_threshold
            agg_kwargs["direction_min_norm_z"] = self.direction_min_norm_z
            agg_kwargs["min_honest_ratio"] = self.min_honest_ratio
            agg_kwargs["fallback_hard_z"] = self.fallback_hard_z
            aggregated_ndarrays, agg_meta = apply_aggregation(
                self.aggregation_method,
                [d["params"] for d in client_data],
                **agg_kwargs,
            )
        elif self.aggregation_method == "trimmed":
            agg_kwargs["trim_ratio"] = self.trim_ratio
            aggregated_ndarrays, agg_meta = apply_aggregation(
                self.aggregation_method,
                [d["params"] for d in client_data],
                **agg_kwargs,
            )
        else:
            # fedavg
            aggregated_ndarrays, agg_meta = apply_aggregation(
                self.aggregation_method,
                [d["params"] for d in client_data],
                **agg_kwargs,
            )

        anomaly_mask: List[bool] = list(agg_meta["anomaly_mask"])
        robust_z_list: Optional[List[float]] = agg_meta["robust_z"]
        detection_method: Optional[str] = agg_meta["detection_method"]
        direction_mask: Optional[List[bool]] = agg_meta.get("direction_anomaly_mask")
        reward_block_mask_raw: Optional[List[bool]] = agg_meta.get("reward_block_mask")
        reward_block_mask: List[bool] = (
            list(reward_block_mask_raw)
            if reward_block_mask_raw is not None
            else [False] * len(client_data)
        )
        risk_scores: Optional[List[float]] = agg_meta.get("risk_score")
        detection_reasons_raw: Optional[List[str]] = agg_meta.get("detection_reasons")
        detection_reasons: List[str] = (
            list(detection_reasons_raw)
            if detection_reasons_raw is not None
            else ["accepted"] * len(client_data)
        )
        if len(detection_reasons) < len(client_data):
            detection_reasons.extend(["accepted"] * (len(client_data) - len(detection_reasons)))

        commitment_blocked_cids = {
            d["cid"] for d, blocked in zip(client_data, commitment_mask) if blocked
        }
        if commitment_blocked_cids:
            reward_block_mask = [
                bool(rb or cm) for rb, cm in zip(reward_block_mask, commitment_mask)
            ]
            for i, blocked in enumerate(commitment_mask):
                if not blocked:
                    continue
                reason = detection_reasons[i] or "accepted"
                suffix = "data_commitment_zero"
                detection_reasons[i] = (
                    suffix if reason == "accepted" else f"{reason}+{suffix}"
                )

        if self.aggregation_method == "csra_dcd" and any(commitment_mask):
            anomaly_mask = [
                bool(anom or commitment)
                for anom, commitment in zip(anomaly_mask, commitment_mask)
            ]
            agg_valid = [
                d for d, anom in zip(client_data, anomaly_mask) if not anom
            ]
            if agg_valid:
                aggregated_ndarrays, _ = apply_aggregation(
                    "fedavg",
                    [d["params"] for d in agg_valid],
                    weights=[d["num_examples"] for d in agg_valid],
                )
            else:
                logger.warning(
                    "Round %s: all CSRA-DCD clients blocked by data commitment; "
                    "keeping original aggregate",
                    server_round,
                )

        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        low_quality_mask = self._low_quality_mask(client_data)
        auth_mask = self._authenticity_mask(client_data)
        inefficient_update_mask, raw_update_norm_z = self._inefficient_update_mask(
            client_data
        )

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
                    detection_reasons[i]
                    if detection_reasons is not None and i < len(detection_reasons)
                    else (
                        detection_method if anomaly_mask[i] and detection_method
                        else "accepted"
                    )
                ),
                "direction_anomaly": (
                    bool(direction_mask[i])
                    if direction_mask is not None and i < len(direction_mask)
                    else False
                ),
                "reward_blocked": (
                    bool(reward_block_mask[i])
                    if reward_block_mask is not None and i < len(reward_block_mask)
                    else bool(anomaly_mask[i])
                ),
                "cosine_to_reference": d.get("cosine_to_reference"),
                "risk_score": (
                    float(risk_scores[i])
                    if risk_scores is not None and i < len(risk_scores)
                    else None
                ),
                "anomaly_score_source": d.get("anomaly_score_source", ""),
                "raw_update_norm": d.get("raw_update_norm"),
                "raw_update_norm_z": (
                    float(raw_update_norm_z[i])
                    if i < len(raw_update_norm_z) else None
                ),
                "normalized_update_score": d.get("normalized_update_score"),
                "reported_data_size": d.get("reported_data_size"),
                "server_known_data_size": d.get("server_known_data_size"),
                "data_commitment_anomaly": bool(commitment_mask[i]),
                "data_size_mismatch": bool(data_size_mismatch_mask[i]),
                "low_quality_outlier": bool(low_quality_mask[i]),
                "inefficient_update": bool(inefficient_update_mask[i]),
                "authenticity_score": d.get("authenticity_score"),
                "authenticity_anomaly": bool(auth_mask[i]),
                "suspicion_signal": 0.0,
                "suspicion_score": self._suspicion_scores.get(cid, 0.0),
                "suspicion_quarantine": False,
                "suspicion_reason": "",
                "alignment_score": (
                    float(agg_meta["alignment_scores"][i])
                    if "alignment_scores" in agg_meta and agg_meta["alignment_scores"] is not None
                    else None
                ),
                "simplex_weight": (
                    float(agg_meta["w_next"][i])
                    if "w_next" in agg_meta and agg_meta["w_next"] is not None
                    else None
                ),
            }

        # ── 3. Valid clients for aggregation (post update-filter) ────────────
        valid_data = [d for d, anom in zip(client_data, anomaly_mask) if not anom]
        valid_cids = [d["cid"] for d in valid_data]
        reward_blocked_cids = {
            d["cid"] for i, d in enumerate(client_data)
            if (
                bool(reward_block_mask[i])
                if reward_block_mask is not None and i < len(reward_block_mask)
                else bool(anomaly_mask[i])
            )
        }
        for cid in reward_blocked_cids:
            det = detection_info.get(cid)
            if det is None or det.get("is_anomaly"):
                continue
            reason = det.get("detection_reason") or "accepted"
            suffix = (
                "reward_quarantine_direction"
                if det.get("direction_anomaly")
                else "reward_quarantine"
            )
            det["detection_reason"] = (
                suffix if reason == "accepted" else f"{reason}+{suffix}"
            )

        suspicion_blocked_cids = set()
        for i, d in enumerate(client_data):
            cid = d["cid"]
            det = detection_info[cid]
            signal = 0.0
            reasons = []

            if commitment_mask[i]:
                signal += self.zero_data_suspicion
                reasons.append("zero_data")
            if data_size_mismatch_mask[i]:
                signal += self.authenticity_suspicion
                reasons.append("data_size_mismatch")
            if det.get("is_anomaly") or det.get("direction_anomaly"):
                signal += self.anomaly_suspicion
                reasons.append("update_anomaly")
            if low_quality_mask[i] and not commitment_mask[i]:
                signal += self.low_quality_suspicion
                reasons.append("low_quality")
            if inefficient_update_mask[i]:
                signal += self.inefficient_update_suspicion
                reasons.append("inefficient_update")
            if auth_mask[i]:
                signal += self.authenticity_suspicion
                reasons.append("low_authenticity")

            previous = self._suspicion_scores.get(cid, 0.0)
            score = max(0.0, previous * self.suspicion_decay + signal)
            self._suspicion_scores[cid] = score

            det["suspicion_signal"] = float(signal)
            det["suspicion_score"] = float(score)
            det["suspicion_reason"] = "+".join(reasons)
            det["suspicion_quarantine"] = bool(
                self.suspicion_threshold > 0.0
                and score >= self.suspicion_threshold
            )

            if det["suspicion_quarantine"]:
                suspicion_blocked_cids.add(cid)

        for cid in suspicion_blocked_cids:
            det = detection_info.get(cid)
            if det is None:
                continue
            det["reward_blocked"] = True
            reason = det.get("detection_reason") or "accepted"
            det["detection_reason"] = (
                "suspicion_quarantine"
                if reason == "accepted"
                else f"{reason}+suspicion_quarantine"
            )
        reward_blocked_cids.update(suspicion_blocked_cids)

        # ── 4. Submit contribution lên blockchain (audit cho TẤT CẢ) ────────
        reputations_map: Dict[int, float] = {}
        onchain_honesty_map: Dict[int, bool] = {}
        if self.bridge is not None:
            for i, d in enumerate(client_data):
                # Client bị reward-block → contribution = 0 (audit-friendly)
                is_reward_blocked = d["cid"] in reward_blocked_cids
                alignment_qs = max(0.0, detection_info[d["cid"]].get("cosine_to_reference", 0.0) or 0.0)
                qs = 0.0 if is_reward_blocked else alignment_qs
                ds = 0 if is_reward_blocked else d["data_size"]
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
                    rep, is_honest_on_chain = self.bridge.get_reputation(cid)
                    reputations_map[cid] = float(rep)
                    onchain_honesty_map[cid] = bool(is_honest_on_chain)
                except Exception:
                    reputations_map[cid] = 0.0
                    onchain_honesty_map[cid] = True
        else:
            # Không có blockchain → reputation default = 0.5 cho mọi participant
            # (đủ để CSRA reward formula chạy, tránh bias)
            for cid in range(self.cfg.fl.n_clients):
                reputations_map[cid] = 0.5
                onchain_honesty_map[cid] = True

        reputation_blocked_cids = {
            d["cid"] for d in valid_data
            if not onchain_honesty_map.get(d["cid"], True)
        }
        for cid in reputation_blocked_cids:
            det = detection_info.get(cid)
            if det is None:
                continue
            reason = det.get("detection_reason") or "accepted"
            det["detection_reason"] = (
                "low_reputation"
                if reason == "accepted"
                else f"{reason}+low_reputation"
            )

        # Reward eligibility is stricter than aggregation validity: a client
        # can contribute an update but still be blocked from reward by
        # historical on-chain reputation.
        reward_eligible_data = [
            d for d in valid_data
            if (
                d["cid"] not in reward_blocked_cids
                and d["cid"] not in reputation_blocked_cids
            )
        ]
        reward_eligible_cids = [d["cid"] for d in reward_eligible_data]
        reward_components: Dict[int, dict] = {
            d["cid"]: {
                "quality": 0.0,
                "data": 0.0,
                "reputation": 0.0,
            }
            for d in client_data
        }

        # ── 6. Reward policy trên clients đủ điều kiện nhận thưởng ───────────
        reward_kwargs: Dict = {}
        if self.reward_policy == "data":
            data_policy_sizes = [
                d["data_size"] for d in reward_eligible_data
            ]
            reward_kwargs["data_sizes"] = data_policy_sizes
            if reward_eligible_data:
                sizes = np.asarray(data_policy_sizes, dtype=float)
                total_size = float(np.sum(np.clip(sizes, 0.0, None)))
                if total_size <= 1e-12:
                    data_scores = np.ones(len(sizes), dtype=float) / len(sizes)
                else:
                    data_scores = sizes / total_size
                for d, score in zip(reward_eligible_data, data_scores):
                    reward_components[d["cid"]]["data"] = float(score)
        elif self.reward_policy == "quality":
            quality_policy_scores = [
                max(0.0, detection_info[d["cid"]].get("cosine_to_reference", 0.0) or 0.0)
                for d in reward_eligible_data
            ]
            reward_kwargs["quality_scores"] = quality_policy_scores
            if reward_eligible_data:
                quality = np.asarray(quality_policy_scores, dtype=float)
                quality = np.clip(quality, 0.0, None)
                total_quality = float(quality.sum())
                if total_quality <= 1e-12:
                    quality_scores = np.ones(len(quality), dtype=float) / len(quality)
                else:
                    quality_scores = quality / total_quality
                for d, score in zip(reward_eligible_data, quality_scores):
                    reward_components[d["cid"]]["quality"] = float(score)
        elif self.reward_policy == "csra":
            csra_quality_scores = [
                max(0.0, detection_info[d["cid"]].get("cosine_to_reference", 0.0) or 0.0)
                for d in reward_eligible_data
            ]
            csra_data_sizes = [
                d["data_size"] for d in reward_eligible_data
            ]
            csra_reputations = [
                reputations_map.get(d["cid"], 0.0) for d in reward_eligible_data
            ]
            reward_kwargs["quality_scores"] = csra_quality_scores
            reward_kwargs["data_sizes"] = csra_data_sizes
            reward_kwargs["reputations"] = csra_reputations
            reward_kwargs["beta"] = self.beta
            reward_kwargs["gamma"] = self.gamma
            reward_kwargs["delta"] = self.delta
            reward_kwargs["mean_data_size"] = self.mean_data_size
            reward_kwargs["contrib_cfg"] = self.cfg.contrib

            if reward_eligible_data:
                q_hat = hybrid_normalize(
                    np.asarray(csra_quality_scores, dtype=float),
                    cfg=self.cfg.contrib,
                )
                d_hat = hybrid_normalize(
                    np.asarray(csra_data_sizes, dtype=float),
                    mean_val=self.mean_data_size,
                    cfg=self.cfg.contrib,
                )
                r_hat = hybrid_normalize(
                    np.asarray(csra_reputations, dtype=float),
                    cfg=self.cfg.contrib,
                )
                for d, qv, dv, rv in zip(reward_eligible_data, q_hat, d_hat, r_hat):
                    reward_components[d["cid"]] = {
                        "quality": float(self.beta * qv),
                        "data": float(self.gamma * dv),
                        "reputation": float(self.delta * rv),
                    }

        pool_eth = self.cfg.blockchain.pool_eth_per_round
        rewards_map = apply_reward_policy(
            self.reward_policy,
            client_ids=reward_eligible_cids,
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
                "direction_anomaly": False, "cosine_to_reference": None,
                "reward_blocked": False, "risk_score": None,
                "anomaly_score_source": "",
                "data_commitment_anomaly": False,
                "data_size_mismatch": False,
                "low_quality_outlier": False,
                "inefficient_update": False,
                "authenticity_anomaly": False,
                "raw_update_norm": None,
                "raw_update_norm_z": None,
                "normalized_update_score": None,
                "reported_data_size": None,
                "server_known_data_size": self.server_data_sizes.get(cid),
                "suspicion_signal": 0.0,
                "suspicion_score": self._suspicion_scores.get(cid, 0.0),
                "suspicion_quarantine": False,
                "suspicion_reason": "",
            })

            if d is None:
                # Client không tham gia round này (lỗi hoặc dropout)
                client_type = self.client_types.get(cid, "honest")
                row = {
                    "round_num": server_round,
                    "client_id": cid,
                    "client_type": client_type,
                    "ground_truth_honest": client_type == "honest",
                    "quality": 0.0,
                    "data_size": 0,
                    "reported_data_size": 0,
                    "server_known_data_size": self.server_data_sizes.get(cid),
                    "w_new": 0.0,
                    "reputation": reputations_map.get(cid, 0.0),
                    "reward_eth": 0.0,
                    "reward_blocked": False,
                    "reward_eligible": False,
                    "is_honest": False,
                    "anomaly_score": None,
                    "robust_z": None,
                    "is_anomaly": False,
                    "detection_reason": "not_participating",
                    "direction_anomaly": False,
                    "cosine_to_reference": None,
                    "risk_score": None,
                    "anomaly_score_source": "",
                    "raw_update_norm": None,
                    "raw_update_norm_z": None,
                    "normalized_update_score": None,
                    "data_commitment_anomaly": False,
                    "data_size_mismatch": False,
                    "low_quality_outlier": False,
                    "inefficient_update": False,
                    "suspicion_signal": 0.0,
                    "suspicion_score": self._suspicion_scores.get(cid, 0.0),
                    "suspicion_quarantine": False,
                    "suspicion_reason": "",
                    "reward_component_quality": 0.0,
                    "reward_component_data": 0.0,
                    "reward_component_reputation": 0.0,
                }
            else:
                client_type = self.client_types.get(cid, "honest")
                reward_eligible = cid in reward_eligible_cids
                row = {
                    "round_num": server_round,
                    "client_id": cid,
                    "client_type": client_type,
                    "ground_truth_honest": client_type == "honest",
                    "quality": d["quality_score"],
                    "data_size": d["data_size"],
                    "reported_data_size": d["reported_data_size"],
                    "server_known_data_size": d.get("server_known_data_size"),
                    "w_new": w_new_map.get(cid, 0.0),
                    "reputation": reputations_map.get(cid, 0.0),
                    "reward_eth": rewards_map.get(cid, 0.0),
                    "reward_blocked": bool(
                        cid in reward_blocked_cids
                        or cid in reputation_blocked_cids
                    ),
                    "reward_eligible": reward_eligible,
                    "is_honest": reward_eligible,
                    "anomaly_score": det["anomaly_score"],
                    "robust_z": det["robust_z"],
                    "is_anomaly": det["is_anomaly"],
                    "detection_reason": det["detection_reason"],
                    "direction_anomaly": det["direction_anomaly"],
                    "cosine_to_reference": det["cosine_to_reference"],
                    "risk_score": det["risk_score"],
                    "anomaly_score_source": det["anomaly_score_source"],
                    "raw_update_norm": det["raw_update_norm"],
                    "raw_update_norm_z": det["raw_update_norm_z"],
                    "normalized_update_score": det["normalized_update_score"],
                    "data_commitment_anomaly": det["data_commitment_anomaly"],
                    "data_size_mismatch": det["data_size_mismatch"],
                    "low_quality_outlier": det["low_quality_outlier"],
                    "inefficient_update": det["inefficient_update"],
                    "authenticity_score": det.get("authenticity_score"),
                    "authenticity_anomaly": det.get("authenticity_anomaly"),
                    "suspicion_signal": det["suspicion_signal"],
                    "suspicion_score": det["suspicion_score"],
                    "suspicion_quarantine": det["suspicion_quarantine"],
                    "suspicion_reason": det["suspicion_reason"],
                    "alignment_score": det.get("alignment_score"),
                    "simplex_weight": det.get("simplex_weight"),
                    "reward_component_quality": reward_components[cid]["quality"],
                    "reward_component_data": reward_components[cid]["data"],
                    "reward_component_reputation": reward_components[cid]["reputation"],
                }
            pending_rows.append(row)

        self._pending_logs[server_round] = pending_rows

        n_valid = len(valid_cids)
        n_filtered = len(client_data) - n_valid
        logger.info(
            f"Round {server_round} — agg={self.aggregation_method} "
            f"reward={self.reward_policy} valid={n_valid}/{len(client_data)} "
            f"reward_eligible={len(reward_eligible_cids)}/{len(client_data)} "
            f"filtered={n_filtered} rep_blocked={len(reputation_blocked_cids)} "
            f"pool={pool_eth} ETH"
        )
        self.set_current_global_parameters(aggregated_ndarrays)
        return aggregated_params, {}

    # ─────────────────────────────────────────────────────────────────────────
    # Logging plumbing
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_round_logs(
        self,
        server_round: int,
        global_accuracy: Optional[float],
        global_loss: Optional[float] = None,
    ):
        for row in self._pending_logs.pop(server_round, []):
            self.exp_logger.log_round(
                dataset=self.cfg.experiment.dataset,
                scenario=self.cfg.experiment.scenario,
                aggregation_method=self.aggregation_method,
                reward_policy=self.reward_policy,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                mad_threshold=(
                    self.mad_threshold
                    if self.aggregation_method == "csra_dcd" else None
                ),
                cosine_threshold=(
                    self.cosine_threshold
                    if self.aggregation_method == "csra_dcd" else None
                ),
                direction_min_norm_z=(
                    self.direction_min_norm_z
                    if self.aggregation_method == "csra_dcd" else None
                ),
                min_honest_ratio=(
                    self.min_honest_ratio
                    if self.aggregation_method == "csra_dcd" else None
                ),
                fallback_hard_z=(
                    self.fallback_hard_z
                    if self.aggregation_method == "csra_dcd" else None
                ),
                suspicion_decay=self.suspicion_decay,
                suspicion_threshold=self.suspicion_threshold,
                low_quality_z_threshold=self.low_quality_z_threshold,
                low_quality_suspicion=self.low_quality_suspicion,
                zero_data_suspicion=self.zero_data_suspicion,
                anomaly_suspicion=self.anomaly_suspicion,
                authenticity_suspicion=self.authenticity_suspicion,
                low_authenticity_threshold=self.low_authenticity_threshold,
                high_update_norm_z_threshold=self.high_update_norm_z_threshold,
                inefficient_update_suspicion=self.inefficient_update_suspicion,
                alpha_law=self.alpha_law,
                beta_law=self.beta_law,
                sparsity_s=self.sparsity_s,
                capping_t=self.capping_t,
                attack_type=self.attack_type,
                seed=self.seed,
                persistent_clients=self.cfg.experiment.persistent_clients,
                num_clients=self.cfg.fl.n_clients,
                num_rounds=self.cfg.fl.n_rounds,
                local_epochs=self.cfg.fl.local_epochs,
                batch_size=self.cfg.fl.batch_size,
                learning_rate=self.cfg.fl.learning_rate,
                client_fraction=self.cfg.fl.fraction_fit,
                data_split=self.cfg.experiment.scenario,
                data_imbalance=self.cfg.experiment.data_imbalance,
                dirichlet_alpha=(
                    self.cfg.experiment.dirichlet_beta
                    if self.cfg.experiment.scenario == "K3"
                    else None
                ),
                global_accuracy=global_accuracy,
                global_loss=global_loss,
                **row,
            )

    def flush_pending_logs(self):
        """Force-flush bất kỳ pending log nào (gọi khi shutdown)."""
        for round_num in sorted(list(self._pending_logs.keys())):
            self._flush_round_logs(
                round_num,
                self._last_accuracy.get(round_num),
                self._last_loss.get(round_num),
            )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ):
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        acc = None
        global_loss = float(agg_loss) if agg_loss is not None else None
        weighted = [
            (res.num_examples, float(res.metrics["accuracy"]))
            for _, res in results
            if "accuracy" in res.metrics
        ]
        weighted_loss = [
            (res.num_examples, float(res.loss))
            for _, res in results
        ]
        total = sum(n for n, _ in weighted)
        if total > 0:
            acc = float(sum(n * a for n, a in weighted) / total)
            self._last_accuracy[server_round] = acc
            logger.info(f"Round {server_round} — global accuracy: {acc:.4f}")
        loss_total = sum(n for n, _ in weighted_loss)
        if global_loss is None and loss_total > 0:
            global_loss = float(sum(n * loss for n, loss in weighted_loss) / loss_total)
        if global_loss is not None:
            self._last_loss[server_round] = global_loss
        self._flush_round_logs(server_round, acc, global_loss)
        return agg_loss, metrics
