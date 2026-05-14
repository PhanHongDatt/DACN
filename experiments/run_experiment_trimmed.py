"""
run_experiment_trimmed.py — TrimmedMean FL baseline runner.
This runner uses standard FL clients and no blockchain reward mechanism.
"""
import argparse
import logging
import os
import sys
import traceback

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.client import FLClient
from fl.config import ExperimentConfig, FLConfig, ProjectConfig
from fl.data_utils import apply_labels, get_client_partitions, load_dataset, make_dataloader
from fl.logger import ExperimentLogger, make_run_id
from fl.models import get_model, get_parameters
from fl.server_trimmed import TrimmedMeanStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="TrimmedMean FL Baseline Runner")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--scenario", default="K1", choices=["K1", "K2", "K3"])
    p.add_argument("--config", default="TrimmedMean")
    p.add_argument("--trim-ratio", default=0.1, type=float)
    p.add_argument("--n-clients", default=10, type=int)
    p.add_argument("--n-rounds", default=50, type=int)
    p.add_argument("--with-freeriders", action="store_true")
    p.add_argument("--with-label-noise", action="store_true")
    p.add_argument("--no-blockchain", action="store_true", help="Accepted for run_all.sh compatibility")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--log-dir", default="./results/logs")
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--local-epochs", default=2, type=int)
    p.add_argument("--dirichlet-alpha", default=None, type=float)
    return p.parse_args()


def build_client_types(n_clients, with_freeriders, exp_cfg):
    types = {i: "honest" for i in range(n_clients)}
    if with_freeriders:
        for i in exp_cfg.free_rider_ids:
            if i < n_clients:
                types[i] = "free_rider"
        for i in exp_cfg.lazy_client_ids:
            if i < n_clients:
                types[i] = "lazy"
    return types


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    fl_cfg = FLConfig(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )
    cfg = ProjectConfig(fl=fl_cfg, experiment=exp_cfg)
    os.makedirs(args.log_dir, exist_ok=True)

    logger.info("Loading %s, scenario=%s", args.dataset, args.scenario)
    splits, train_labels, mean_ds = get_client_partitions(args.dataset, args.n_clients, exp_cfg)
    train_dataset = load_dataset(args.dataset, train=True)
    apply_labels(train_dataset, train_labels)
    test_dataset = load_dataset(args.dataset, train=False)
    sizes = [len(split) for split in splits]
    logger.info("Data sizes per client: min=%s max=%s mean=%.0f", min(sizes), max(sizes), mean_ds)

    client_types = build_client_types(args.n_clients, args.with_freeriders, exp_cfg)
    logger.info("Client types: %s", client_types)

    run_id = make_run_id(
        args.dataset,
        args.scenario,
        args.config,
        0.0,
        exp_cfg.dirichlet_beta if args.scenario == "K3" else None,
    )
    exp_logger = ExperimentLogger(run_id, args.log_dir)

    fh = logging.FileHandler(f"{args.log_dir}/{run_id}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)

    test_loader_shared = make_dataloader(
        test_dataset,
        np.arange(len(test_dataset)),
        fl_cfg.batch_size,
        shuffle=False,
    )

    def client_fn(cid: str):
        idx = int(cid)
        train_loader = make_dataloader(train_dataset, splits[idx], fl_cfg.batch_size)
        return FLClient(
            client_id=idx,
            dataset_name=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader_shared,
            client_type=client_types[idx],
            fl_cfg=fl_cfg,
        ).to_client()

    init_params = ndarrays_to_parameters(get_parameters(get_model(args.dataset)))

    strategy = TrimmedMeanStrategy(
        exp_logger=exp_logger,
        cfg=cfg,
        client_types=client_types,
        trim_ratio=args.trim_ratio,
        fraction_fit=fl_cfg.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=args.n_clients,
        min_evaluate_clients=args.n_clients,
        min_available_clients=args.n_clients,
        initial_parameters=init_params,
    )

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.n_clients,
            config=fl.server.ServerConfig(num_rounds=args.n_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        logger.info("TrimmedMean simulation complete. Log: %s", exp_logger.filepath)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving partial TrimmedMean results")
    except Exception as e:
        logger.error("TrimmedMean simulation failed: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        strategy.flush_pending_logs()
        exp_logger.close()
        logger.info("Logger closed cleanly")


if __name__ == "__main__":
    main()
