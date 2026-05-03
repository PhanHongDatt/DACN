"""
run_experiment_csra.py — Entry point để chạy thực nghiệm so sánh với logic CSRA.
Sử dụng FLClientCSRA và FLCSRAStrategy.
"""
import argparse
import logging
import sys
import os
import torch
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters

# Import từ project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.config import ProjectConfig, FLConfig, ExperimentConfig
from fl.data_utils import get_client_partitions, load_dataset, make_dataloader
from fl.models import get_model, get_parameters
from fl.blockchain import BlockchainBridge
from fl.server_csra import FLCSRAStrategy
from fl.client_csra import FLClientCSRA
from fl.logger import ExperimentLogger, make_run_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--scenario", default="K3") # Mặc định K3 để thấy rõ sức mạnh CSRA
    p.add_argument("--config", default="C")
    p.add_argument("--alpha", default=0.5, type=float)
    p.add_argument("--n-clients", default=10, type=int)
    p.add_argument("--n-rounds", default=20, type=int) # Chạy ngắn 20 rounds để so sánh nhanh
    p.add_argument("--with-freeriders", action="store_true", default=True) # Luôn bật để test DCD
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(42)
    
    fl_cfg = FLConfig(n_clients=args.n_clients, n_rounds=args.n_rounds)
    exp_cfg = ExperimentConfig(dataset=args.dataset, scenario=args.scenario, config_type=args.config)
    cfg = ProjectConfig(fl=fl_cfg, experiment=exp_cfg)

    # Data
    splits, _, mean_ds = get_client_partitions(args.dataset, args.n_clients, exp_cfg)
    train_dataset = load_dataset(args.dataset, train=True)
    test_dataset = load_dataset(args.dataset, train=False)
    
    test_loader_shared = make_dataloader(test_dataset, np.arange(len(test_dataset)), fl_cfg.batch_size, shuffle=False)

    # Client types
    client_types = {i: "honest" for i in range(args.n_clients)}
    if args.with_freeriders:
        for i in exp_cfg.free_rider_ids: client_types[i] = "free_rider"

    # Blockchain
    bridge = BlockchainBridge(cfg.blockchain, cfg.contrib)
    bridge.register_all_clients(args.n_clients)
    exp_id = bridge.start_experiment(args.dataset, args.scenario, args.alpha, args.n_clients, args.n_rounds)

    # Logger
    run_id = make_run_id(args.dataset, args.scenario, args.config + "-CSRA", args.alpha)
    exp_logger = ExperimentLogger(run_id, "./results/logs")

    def client_fn(cid: str):
        idx = int(cid)
        train_loader = make_dataloader(train_dataset, splits[idx], fl_cfg.batch_size)
        return FLClientCSRA(
            client_id=idx,
            dataset_name=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader_shared,
            client_type=client_types[idx],
            fl_cfg=fl_cfg
        ).to_client()

    init_params = ndarrays_to_parameters(get_parameters(get_model(args.dataset)))

    strategy = FLCSRAStrategy(
        bridge=bridge,
        exp_logger=exp_logger,
        cfg=cfg,
        alpha=args.alpha,
        mean_data_size=mean_ds,
        client_types=client_types,
        initial_parameters=init_params,
        min_fit_clients=args.n_clients,
        min_available_clients=args.n_clients
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.n_clients,
        config=fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy=strategy
    )
    
    exp_logger.close()
    logger.info("CSRA Experiment Finished.")

if __name__ == "__main__":
    main()
