"""
run_experiment.py — Unified entrypoint cho toàn bộ ablation matrix.

Thay thế các file legacy:
  - run_experiment.py (cũ, FLBlockchainStrategy)
  - run_experiment_csra.py
  - run_experiment_trimmed.py

CLI mới:
  python experiments/run_experiment.py \\
      --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \\
      --aggregation csra_dcd --reward-policy csra \\
      --beta 0.5 --gamma 0.3 --delta 0.2 \\
      --attack free_rider --attack-client-ids 7,8 \\
      --seed 42 --n-rounds 50

Reference: docs/PLAN.md §7.4.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback

# Đảm bảo import từ project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from fl.aggregation_methods import AGGREGATION_NAMES
from fl.blockchain import BlockchainBridge
from fl.client_attacks import ATTACK_NAMES, make_client
from fl.config import ExperimentConfig, FLConfig, ProjectConfig
from fl.data_utils import (
    apply_labels,
    get_client_partitions,
    load_dataset,
    make_dataloader,
)
from fl.logger import ExperimentLogger, make_run_id
from fl.models import get_model, get_parameters
from fl.reward_policies import POLICY_NAMES
from fl.server_base import FLUnifiedStrategy
from fl.simulation_local import run_simulation_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified FL+Blockchain experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset & scenario
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--scenario", default="K1", choices=["K1", "K2", "K3"])
    p.add_argument("--dirichlet-alpha", type=float, default=None,
                   help="Dirichlet alpha (chỉ áp dụng cho K3)")

    # Method
    p.add_argument("--aggregation", required=True, choices=list(AGGREGATION_NAMES),
                   help="Aggregation method")
    p.add_argument("--reward-policy", required=True, choices=list(POLICY_NAMES),
                   help="Reward distribution policy")

    # CSRA reward weights
    p.add_argument("--beta", type=float, default=0.5,
                   help="Quality weight (csra only)")
    p.add_argument("--gamma", type=float, default=0.3,
                   help="Data size weight (csra only)")
    p.add_argument("--delta", type=float, default=0.2,
                   help="Reputation weight (csra only)")

    # Aggregation params
    p.add_argument("--trim-ratio", type=float, default=0.1,
                   help="TrimmedMean trim ratio (trimmed only)")
    p.add_argument("--mad-threshold", type=float, default=3.0,
                   help="MAD z threshold (csra_dcd only)")
    p.add_argument("--min-honest-ratio", type=float, default=0.5,
                   help="Min honest ratio failsafe (csra_dcd only)")

    # Attack
    p.add_argument("--attack", default="clean",
                   choices=["clean", *ATTACK_NAMES],
                   help="Attack type for designated clients")
    p.add_argument("--attack-client-ids", default="",
                   help="Comma-separated client IDs to be attackers "
                        "(empty = last 2 clients if attack != clean)")

    # FL setup
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--local-epochs", type=int, default=2)

    # Misc
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (bắt buộc)")
    p.add_argument("--no-blockchain", action="store_true",
                   help="Disable blockchain audit (mọi reward chỉ trong CSV)")
    p.add_argument("--log-dir", default="./results/logs")

    return p.parse_args()


def parse_attack_ids(spec: str, attack: str, n_clients: int) -> list[int]:
    """Parse --attack-client-ids. Default = 2 client cuối nếu có attack."""
    if attack == "clean":
        return []
    if spec.strip():
        ids = [int(x) for x in spec.split(",") if x.strip()]
        for cid in ids:
            if not 0 <= cid < n_clients:
                raise ValueError(
                    f"attack client_id {cid} out of range [0, {n_clients})"
                )
        return ids
    # Default: 2 client cuối
    default = list(range(max(0, n_clients - 2), n_clients))
    logger.info(f"No --attack-client-ids given, defaulting to {default}")
    return default


def build_client_types(
    n_clients: int,
    attack: str,
    attack_ids: list[int],
) -> dict[int, str]:
    types = {i: "honest" for i in range(n_clients)}
    if attack != "clean":
        for cid in attack_ids:
            types[cid] = attack
    return types


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Validate ─────────────────────────────────────────────────────────
    if args.reward_policy == "csra":
        weight_sum = args.beta + args.gamma + args.delta
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"beta + gamma + delta must equal 1.0, got {weight_sum:.6f} "
                f"(β={args.beta} γ={args.gamma} δ={args.delta})"
            )

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Config ───────────────────────────────────────────────────────────
    exp_cfg = ExperimentConfig(
        dataset=args.dataset,
        scenario=args.scenario,
        config_type="unified",  # legacy field, không dùng nữa
        seed=args.seed,
        log_dir=args.log_dir,
        # Clear legacy lists — new runner dùng client_attacks.py thay vì data noise
        noise_clients=[],
        lazy_client_ids=[],
    )
    if args.dirichlet_alpha is not None:
        exp_cfg.dirichlet_beta = args.dirichlet_alpha

    fl_cfg = FLConfig(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )
    cfg = ProjectConfig(fl=fl_cfg, experiment=exp_cfg)

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Attack setup ─────────────────────────────────────────────────────
    attack_ids = parse_attack_ids(args.attack_client_ids, args.attack, args.n_clients)
    client_types = build_client_types(args.n_clients, args.attack, attack_ids)

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info(
        f"Loading {args.dataset} | scenario={args.scenario} "
        f"| dirichlet={exp_cfg.dirichlet_beta if args.scenario == 'K3' else 'N/A'}"
    )
    splits, train_labels, mean_ds = get_client_partitions(
        args.dataset, args.n_clients, exp_cfg
    )
    train_dataset = load_dataset(args.dataset, train=True)
    apply_labels(train_dataset, train_labels)
    test_dataset = load_dataset(args.dataset, train=False)

    sizes = [len(s) for s in splits]
    logger.info(
        f"Client data sizes — min={min(sizes)} max={max(sizes)} "
        f"mean={mean_ds:.0f}"
    )
    logger.info(f"Client types: {client_types}")

    # ── Blockchain bridge ────────────────────────────────────────────────
    bridge: BlockchainBridge | None = None
    exp_id_chain = None
    if not args.no_blockchain:
        try:
            bridge = BlockchainBridge(cfg.blockchain, cfg.contrib)
            bridge.register_all_clients(args.n_clients)
            # Use beta as the headline weight stored on-chain for CSRA, else 0
            on_chain_alpha = args.beta if args.reward_policy == "csra" else 0.0
            exp_id_chain = bridge.start_experiment(
                args.dataset, args.scenario, on_chain_alpha,
                args.n_clients, args.n_rounds,
            )
            logger.info(f"Blockchain ready — experiment_id={exp_id_chain}")
        except Exception as e:
            logger.warning(
                f"Blockchain init failed: {e}. Running audit-only mode disabled."
            )
            bridge = None

    # ── Logger ───────────────────────────────────────────────────────────
    run_id = make_run_id(
        dataset=args.dataset,
        scenario=args.scenario,
        aggregation_method=args.aggregation,
        reward_policy=args.reward_policy,
        seed=args.seed,
        beta=args.beta if args.reward_policy == "csra" else 0.0,
        gamma=args.gamma if args.reward_policy == "csra" else 0.0,
        delta=args.delta if args.reward_policy == "csra" else 0.0,
        attack_type=args.attack,
        dirichlet_alpha=(exp_cfg.dirichlet_beta if args.scenario == "K3" else None),
    )
    exp_logger = ExperimentLogger(run_id, args.log_dir)
    fh = logging.FileHandler(f"{args.log_dir}/{run_id}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run ID: {run_id}")
    logger.info(
        f"agg={args.aggregation} reward={args.reward_policy} "
        f"β={args.beta:.2f} γ={args.gamma:.2f} δ={args.delta:.2f} "
        f"attack={args.attack} seed={args.seed}"
    )

    # ── Flower simulation ────────────────────────────────────────────────
    test_loader_shared = make_dataloader(
        test_dataset, np.arange(len(test_dataset)), fl_cfg.batch_size, shuffle=False,
    )

    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Factory: trả về NumPyClient (không wrap to_client cho local sim)."""
        idx = int(cid)
        train_loader = make_dataloader(
            train_dataset, splits[idx], fl_cfg.batch_size,
        )
        ctype = client_types[idx]
        return make_client(
            client_type=ctype,
            client_id=idx,
            dataset_name=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader_shared,
            fl_cfg=fl_cfg,
        )

    init_params = ndarrays_to_parameters(get_parameters(get_model(args.dataset)))

    # Khi reward != csra, β γ δ không có ý nghĩa — set về 0 để CSV và filename
    # consistent (chỉ run với reward=csra mới có giá trị thực).
    eff_beta = args.beta if args.reward_policy == "csra" else 0.0
    eff_gamma = args.gamma if args.reward_policy == "csra" else 0.0
    eff_delta = args.delta if args.reward_policy == "csra" else 0.0

    strategy = FLUnifiedStrategy(
        aggregation_method=args.aggregation,
        reward_policy=args.reward_policy,
        cfg=cfg,
        exp_logger=exp_logger,
        client_types=client_types,
        mean_data_size=mean_ds,
        seed=args.seed,
        attack_type=args.attack,
        # Aggregation params
        trim_ratio=args.trim_ratio,
        mad_threshold=args.mad_threshold,
        min_honest_ratio=args.min_honest_ratio,
        # Reward params
        beta=eff_beta,
        gamma=eff_gamma,
        delta=eff_delta,
        # Blockchain
        bridge=bridge,
        # Flower base
        fraction_fit=fl_cfg.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=args.n_clients,
        min_evaluate_clients=args.n_clients,
        min_available_clients=args.n_clients,
        initial_parameters=init_params,
    )

    try:
        run_simulation_local(
            strategy=strategy,
            client_fn=client_fn,
            num_clients=args.n_clients,
            n_rounds=args.n_rounds,
            initial_parameters=init_params,
        )
        logger.info(f"Simulation complete. CSV: {exp_logger.filepath}")

        if bridge and exp_id_chain is not None:
            try:
                bridge.end_experiment(exp_id_chain, args.n_rounds)
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving partial results")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        strategy.flush_pending_logs()
        exp_logger.close()
        logger.info("Logger closed cleanly")


if __name__ == "__main__":
    main()
