"""
config.py — Cấu hình toàn cục cho dự án.
Tất cả hyperparameter tập trung ở đây, không hardcode trong code.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class BlockchainConfig:
    rpc_url:           str   = "http://127.0.0.1:8545"
    chain_id:          int   = 31337
    pool_eth_per_round: float = 1.0   # ETH deposit mỗi vòng
    gas_limit:         int   = 500_000
    # Fail-closed by default: if reputation cannot be read, the client is not
    # treated as eligible for reward. Set True only for environment debugging.
    reputation_fail_open: bool = False
    # Địa chỉ contract (tự động load từ contract_addresses.json sau deploy)
    store_address:     str   = ""
    dist_address:      str   = ""
    registry_address:  str   = ""

@dataclass
class FLConfig:
    n_clients:         int   = 10
    n_rounds:          int   = 50
    local_epochs:      int   = 2
    batch_size:        int   = 32
    learning_rate:     float = 0.01
    server_port:       int   = 8080
    fraction_fit:      float = 1.0   # tỷ lệ client tham gia mỗi vòng

@dataclass
class FedLAWConfig:
    alpha:             float = 0.1   # learning rate cho mô hình toàn cục (alpha)
    beta:              float = 0.01  # learning rate cho trọng số (beta)
    sparsity_s:        int   = 10    # số lượng client được giữ lại (s)
    capping_t:         float = 0.2   # chặn trên của trọng số một client (t)

@dataclass
class ContributionConfig:
    alpha:             float = 0.5   # trọng số Quality Score
    beta_reputation:   float = 0.9   # decay factor EWMA
    window_size:       int   = 10    # sliding window
    rep_threshold:     float = 0.1   # ngưỡng loại free-rider
    min_warmup_rounds: int   = 5     # warm-up trước khi lọc
    tau:               float = 1e-4  # ngưỡng phát hiện IID (Hybrid Norm)
    epsilon:           float = 1e-8  # smoothing epsilon

@dataclass
class ExperimentConfig:
    dataset:           str   = "mnist"          # mnist | fashion_mnist | cifar10
    scenario:          str   = "K1"             # K1 | K2 | K3
    config_type:       str   = "C"              # A | B | C
    alpha_sweep:       List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])
    noise_ratio:       float = 0.4
    noise_clients:     List[int] = field(default_factory=lambda: [8, 9])
    dirichlet_beta:    float = 0.1              # cho K3
    free_rider_ids:    List[int] = field(default_factory=lambda: [7, 8])
    lazy_client_ids:   List[int] = field(default_factory=lambda: [9])
    lazy_data_ratio:   float = 0.1              # lazy dùng 10% data
    # Schema v2: data size heterogeneity pattern
    data_imbalance:    str   = "lognormal"      # uniform | linear | lognormal | step
    persistent_clients: bool = False             # reuse local client objects across rounds
    seed:              int   = 42
    results_dir:       str   = "./results"
    log_dir:           str   = "./results/logs"

@dataclass
class ProjectConfig:
    blockchain: BlockchainConfig   = field(default_factory=BlockchainConfig)
    fl:         FLConfig           = field(default_factory=FLConfig)
    fedlaw:     FedLAWConfig       = field(default_factory=FedLAWConfig)
    contrib:    ContributionConfig = field(default_factory=ContributionConfig)
    experiment: ExperimentConfig   = field(default_factory=ExperimentConfig)
