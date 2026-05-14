"""
run_experiment_csra.py — Entry point để chạy thực nghiệm so sánh với logic CSRA.
Sử dụng FLClientCSRA và FLCSRAStrategy.
"""
import argparse
import logging
import sys
import os
import traceback
import torch
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters

# Import từ project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.config import ProjectConfig, FLConfig, ExperimentConfig
from fl.data_utils import apply_labels, get_client_partitions, load_dataset, make_dataloader
from fl.models import get_model, get_parameters
from fl.blockchain import BlockchainBridge
from fl.server_csra import FLCSRAStrategy
from fl.client_csra import FLClientCSRA
from fl.logger import ExperimentLogger, make_run_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="CSRA FL-Blockchain Experiment Runner")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--scenario", default="K3",
                   choices=["K1", "K2", "K3"])
    p.add_argument("--config", default="C",
                   choices=["A", "B", "C"])
    p.add_argument("--alpha", default=0.5, type=float)
    p.add_argument("--n-clients", default=10, type=int)
    p.add_argument("--n-rounds", default=20, type=int)
    p.add_argument("--with-freeriders", action="store_true")
    p.add_argument("--free-rider-mode", default="noise", choices=["noise", "copy"],
                   help="Free-rider attack: noisy update or copy the global model unchanged")
    p.add_argument("--free-rider-noise-std", default=0.05, type=float,
                   help="Noise std for CSRA free-rider noisy updates")
    p.add_argument("--with-label-noise", action="store_true")
    p.add_argument("--no-blockchain", action="store_true")
    p.add_argument("--dynamic-alpha", action="store_true",
                   help="Enable runtime alpha schedule instead of fixed --alpha")
    p.add_argument("--mad-threshold", default=3.0, type=float,
                   help="Robust z-score threshold for CSRA-DCD")
    p.add_argument("--min-honest-ratio", default=0.5, type=float,
                   help="Fallback if fewer than this ratio of clients remain after DCD")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--log-dir", default="./results/logs")
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--local-epochs", default=2, type=int)
    p.add_argument("--dirichlet-alpha", default=None, type=float,
                   help="Dirichlet label-distribution alpha for K3 Non-IID")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    fl_cfg = FLConfig(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )
    exp_cfg = ExperimentConfig(
        dataset=args.dataset,
        scenario=args.scenario,
        config_type=args.config,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    if not (args.with_freeriders or args.with_label_noise):
        exp_cfg.noise_clients = []
    if not args.with_freeriders:
        exp_cfg.lazy_client_ids = []
    if args.dirichlet_alpha is not None:
        exp_cfg.dirichlet_beta = args.dirichlet_alpha
    cfg = ProjectConfig(fl=fl_cfg, experiment=exp_cfg)
    os.makedirs(args.log_dir, exist_ok=True)

    # Data
    splits, train_labels, mean_ds = get_client_partitions(args.dataset, args.n_clients, exp_cfg)
    train_dataset = load_dataset(args.dataset, train=True)
    apply_labels(train_dataset, train_labels)
    test_dataset = load_dataset(args.dataset, train=False)
    
    test_loader_shared = make_dataloader(test_dataset, np.arange(len(test_dataset)), fl_cfg.batch_size, shuffle=False)

    # Client types
    client_types = {i: "honest" for i in range(args.n_clients)}
    if args.with_freeriders:
        for i in exp_cfg.free_rider_ids:
            if i < args.n_clients:
                client_types[i] = "free_rider"
        for i in exp_cfg.lazy_client_ids:
            if i < args.n_clients:
                client_types[i] = "lazy"

    run_id = make_run_id(
        args.dataset,
        args.scenario,
        args.config + "-CSRA",
        args.alpha,
        exp_cfg.dirichlet_beta if args.scenario == "K3" else None,
    )
    exp_logger = ExperimentLogger(run_id, args.log_dir)

    fh = logging.FileHandler(f"{args.log_dir}/{run_id}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)

    # Blockchain
    bridge = None
    exp_id = None
    use_chain = (not args.no_blockchain) and (args.config != "A")
    if use_chain:
        try:
            bridge = BlockchainBridge(cfg.blockchain, cfg.contrib)
            bridge.register_all_clients(args.n_clients)
            exp_id = bridge.start_experiment(
                args.dataset, args.scenario, args.alpha,
                args.n_clients, args.n_rounds
            )
            logger.info(f"Blockchain ready, experiment_id={exp_id}")
        except Exception as e:
            logger.warning(f"Blockchain init failed: {e}. Running CSRA without on-chain reward.")
            bridge = None

    def client_fn(cid: str):
        idx = int(cid)
        train_loader = make_dataloader(train_dataset, splits[idx], fl_cfg.batch_size)
        return FLClientCSRA(
            client_id=idx,
            dataset_name=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader_shared,
            client_type=client_types[idx],
            fl_cfg=fl_cfg,
            free_rider_mode=args.free_rider_mode,
            free_rider_noise_std=args.free_rider_noise_std,
        ).to_client()

    init_params = ndarrays_to_parameters(get_parameters(get_model(args.dataset)))

    strategy = FLCSRAStrategy(
        bridge=bridge,
        exp_logger=exp_logger,
        cfg=cfg,
        alpha=args.alpha,
        mean_data_size=mean_ds,
        client_types=client_types,
        dynamic_alpha=args.dynamic_alpha,
        mad_threshold=args.mad_threshold,
        min_honest_ratio=args.min_honest_ratio,
        fraction_fit=fl_cfg.fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=init_params,
        min_fit_clients=args.n_clients,
        min_evaluate_clients=args.n_clients,
        min_available_clients=args.n_clients
    )

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.n_clients,
            config=fl.server.ServerConfig(num_rounds=args.n_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        if bridge and use_chain and exp_id is not None:
            try:
                bridge.end_experiment(exp_id, args.n_rounds)
            except Exception as e:
                logger.warning(f"Could not end on-chain experiment: {e}")

        logger.info("CSRA Experiment Finished.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving partial CSRA results")
    except Exception as e:
        logger.error(f"CSRA simulation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        strategy.flush_pending_logs()
        exp_logger.close()
        logger.info("Logger closed cleanly")

if __name__ == "__main__":
    main()
