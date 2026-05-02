"""
run_experiment.py — Entry point chính. Mọi lỗi đều được bắt và log.

Ví dụ:
  python experiments/run_experiment.py --dataset mnist --scenario K1 --config C --alpha 0.5
  python experiments/run_experiment.py --dataset cifar10 --scenario K3 --config B --alpha 1.0 --n-rounds 100
  python experiments/run_experiment.py --dataset mnist --scenario K2 --config C --alpha 0.5 --with-freeriders
"""
import argparse
import logging
import sys
import os
import traceback

# Đảm bảo import từ project root dù gọi từ đâu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BUG1 FIX: import một lần duy nhất ở đầu file
from flwr.common import ndarrays_to_parameters
import flwr as fl
import numpy as np
import torch

from fl.config import ProjectConfig, FLConfig, ExperimentConfig
from fl.data_utils import get_client_partitions, load_dataset, make_dataloader
from fl.models import get_model, get_parameters
from fl.blockchain import BlockchainBridge
from fl.server import FLBlockchainStrategy
from fl.client import FLClient
from fl.logger import ExperimentLogger, make_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # FileHandler được thêm sau khi biết run_id
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FL-Blockchain Experiment Runner")
    p.add_argument("--dataset",        default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--scenario",       default="K1",
                   choices=["K1", "K2", "K3"])
    p.add_argument("--config",         default="C",
                   choices=["A", "B", "C"])
    p.add_argument("--alpha",          default=0.5,  type=float)
    p.add_argument("--n-clients",      default=10,   type=int)
    p.add_argument("--n-rounds",       default=50,   type=int)
    p.add_argument("--with-freeriders", action="store_true")
    p.add_argument("--no-blockchain",   action="store_true")
    p.add_argument("--seed",           default=42,   type=int)
    p.add_argument("--log-dir",        default="./results/logs")
    p.add_argument("--batch-size",     default=32,   type=int)
    p.add_argument("--local-epochs",   default=2,    type=int)
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

    # Seed toàn bộ
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Config ───────────────────────────────────────────────
    exp_cfg = ExperimentConfig(
        dataset     = args.dataset,
        scenario    = args.scenario,
        config_type = args.config,
        seed        = args.seed,
        log_dir     = args.log_dir,
    )
    fl_cfg = FLConfig(
        n_clients     = args.n_clients,
        n_rounds      = args.n_rounds,
        batch_size    = args.batch_size,
        local_epochs  = args.local_epochs,
    )
    cfg   = ProjectConfig(fl=fl_cfg, experiment=exp_cfg)
    alpha = 1.0 if args.config == "B" else args.alpha

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────
    logger.info(f"Loading {args.dataset}, scenario={args.scenario}")
    splits, _, mean_ds = get_client_partitions(args.dataset, args.n_clients, exp_cfg)
    train_dataset = load_dataset(args.dataset, train=True)
    test_dataset  = load_dataset(args.dataset, train=False)

    sizes = [len(s) for s in splits]
    logger.info(f"Data sizes per client: min={min(sizes)} max={max(sizes)} mean={mean_ds:.0f}")

    client_types = build_client_types(args.n_clients, args.with_freeriders, exp_cfg)
    logger.info(f"Client types: {client_types}")

    # ── Blockchain bridge ────────────────────────────────────
    bridge    = None
    use_chain = (not args.no_blockchain) and (args.config != "A")
    if use_chain:
        try:
            bridge = BlockchainBridge(cfg.blockchain, cfg.contrib)
            bridge.register_all_clients(args.n_clients)
            exp_id = bridge.start_experiment(
                args.dataset, args.scenario, alpha,
                args.n_clients, args.n_rounds
            )
            logger.info(f"Blockchain ready, experiment_id={exp_id}")
        except Exception as e:
            logger.warning(f"Blockchain init failed: {e}. Running without on-chain reward.")
            bridge = None

    # ── Logger ───────────────────────────────────────────────
    run_id     = make_run_id(args.dataset, args.scenario, args.config, alpha)
    exp_logger = ExperimentLogger(run_id, args.log_dir)

    # Thêm file handler để log ra file
    fh = logging.FileHandler(f"{args.log_dir}/{run_id}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"alpha={alpha}, n_clients={args.n_clients}, "
                f"n_rounds={args.n_rounds}, seed={args.seed}")

    # Cache test_loader ra ngoài client_fn — tránh tạo mới mỗi lần gọi
    # (DataLoader tạo nhiều lần liên tục qua 50 rounds gây memory leak nhẹ)
    test_loader_shared = make_dataloader(
        test_dataset,
        np.arange(len(test_dataset)),
        fl_cfg.batch_size,
        shuffle=False
    )

    def client_fn(cid: str):
        idx          = int(cid)
        train_loader = make_dataloader(train_dataset, splits[idx], fl_cfg.batch_size)
        return FLClient(
            client_id    = idx,
            dataset_name = args.dataset,
            train_loader = train_loader,
            test_loader  = test_loader_shared,
            client_type  = client_types[idx],
            fl_cfg       = fl_cfg,
        ).to_client()

    # BUG2 FIX: set initial_parameters chỉ 1 lần, trong constructor
    init_params = ndarrays_to_parameters(get_parameters(get_model(args.dataset)))

    strategy = FLBlockchainStrategy(
        bridge            = bridge,
        exp_logger        = exp_logger,
        cfg               = cfg,
        alpha             = alpha,
        mean_data_size    = mean_ds,
        client_types      = client_types,
        fraction_fit      = fl_cfg.fraction_fit,
        fraction_evaluate = 1.0,
        min_fit_clients          = args.n_clients,
        min_evaluate_clients     = args.n_clients,
        min_available_clients    = args.n_clients,
        initial_parameters       = init_params,   # chỉ set ở đây
    )

    # BUG6 FIX: dùng try/finally để đảm bảo logger luôn được đóng
    try:
        fl.simulation.start_simulation(
            client_fn        = client_fn,
            num_clients      = args.n_clients,
            config           = fl.server.ServerConfig(num_rounds=args.n_rounds),
            strategy         = strategy,
            client_resources = {"num_cpus": 1, "num_gpus": 0.0},
        )
        logger.info(f"Simulation complete. Log: {exp_logger.filepath}")

        # Kết thúc experiment on-chain
        if bridge and use_chain:
            try:
                bridge.end_experiment(exp_id, args.n_rounds)
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving partial results")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # BUG6 FIX: LUÔN đóng logger dù có exception hay không
        exp_logger.close()
        logger.info("Logger closed cleanly")


if __name__ == "__main__":
    main()
