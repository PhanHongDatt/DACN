"""
blockchain.py — Web3.py bridge giữa Flower FL và Solidity contracts.

Quy ước scale:
  - float quality/reputation → int × 1_000_000 trước khi gửi lên contract
  - ETH pool → wei khi gọi payable function
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from web3 import Web3
from web3.exceptions import ContractLogicError

from fl.config import BlockchainConfig, ContributionConfig
from fl.normalization import compute_w_new

logger = logging.getLogger(__name__)
SCALE = 1_000_000


class BlockchainBridge:
    def __init__(self, cfg: BlockchainConfig, contrib_cfg: ContributionConfig):
        self.cfg         = cfg
        self.contrib_cfg = contrib_cfg
        self.w3          = Web3(Web3.HTTPProvider(cfg.rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(
                f"Cannot connect to {cfg.rpc_url}. "
                "Run: npx hardhat node (terminal riêng)"
            )

        self.accounts = self.w3.eth.accounts
        if len(self.accounts) < 12:
            raise RuntimeError(
                f"Cần ít nhất 12 accounts (owner + 10 clients + 1 buffer), "
                f"hiện có {len(self.accounts)}. Kiểm tra hardhat.config.js"
            )
        self.owner = self.accounts[0]
        logger.info(f"Blockchain connected: chain_id={self.w3.eth.chain_id}, owner={self.owner[:10]}...")

        # Load contract addresses
        addr_file = Path(__file__).parent / "contract_addresses.json"
        if not addr_file.exists():
            raise FileNotFoundError(
                "fl/contract_addresses.json not found.\n"
                "Chạy: npx hardhat run scripts/deploy.js --network localhost"
            )
        addrs = json.loads(addr_file.read_text())
        logger.info(f"Loaded contract addresses from {addr_file}")

        self.store = self._load_contract("ContributionStore",  addrs["ContributionStore"])
        self.dist  = self._load_contract("RewardDistributor",  addrs["RewardDistributor"])
        self.reg   = self._load_contract("FLRegistry",         addrs["FLRegistry"])

    def _load_contract(self, name: str, address: str):
        # Thử cả hai đường dẫn ABI có thể có
        candidates = [
            Path(f"artifacts/contracts/{name}.sol/{name}.json"),
            Path(f"../artifacts/contracts/{name}.sol/{name}.json"),
        ]
        for p in candidates:
            if p.exists():
                abi = json.loads(p.read_text())["abi"]
                contract = self.w3.eth.contract(address=address, abi=abi)
                logger.debug(f"Loaded {name} at {address[:10]}...")
                return contract
        raise FileNotFoundError(
            f"ABI not found for {name}. Chạy: npx hardhat compile"
        )

    def _transact(self, fn, gas: int = 200_000, value_wei: int = 0,
                  retries: int = 3) -> bool:
        """
        Gửi transaction với retry logic.
        Trả về True nếu thành công, False nếu thất bại sau retries lần.
        """
        tx_kwargs = {"from": self.owner, "gas": gas}
        if value_wei > 0:
            tx_kwargs["value"] = value_wei

        for attempt in range(1, retries + 1):
            try:
                tx_hash = fn.transact(tx_kwargs)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                if receipt["status"] == 0:
                    logger.warning(f"Transaction reverted (attempt {attempt})")
                    return False
                return True
            except ContractLogicError as e:
                logger.error(f"Contract error (attempt {attempt}): {e}")
                return False
            except Exception as e:
                logger.warning(f"Transaction failed (attempt {attempt}): {e}")
                if attempt < retries:
                    time.sleep(1)
        return False

    def client_address(self, client_idx: int) -> str:
        """account[0] = owner. Clients bắt đầu từ account[1]."""
        return self.accounts[client_idx + 1]

    # ── Setup ─────────────────────────────────────────────────
    def register_all_clients(self, n_clients: int):
        """Đăng ký n_clients lên ContributionStore. Bỏ qua nếu đã đăng ký."""
        registered = 0
        for i in range(n_clients):
            addr = self.client_address(i)
            ok = self._transact(
                self.store.functions.registerClient(addr),
                gas=120_000
            )
            if ok:
                registered += 1
        logger.info(f"Registered {registered}/{n_clients} clients on-chain")

    def start_experiment(self, dataset: str, scenario: str,
                         alpha: float, n_clients: int, n_rounds: int) -> int:
        fn = self.reg.functions.startExperiment(
            dataset, scenario, int(alpha * SCALE), n_clients, n_rounds
        )
        ok = self._transact(fn, gas=500_000)
        return 0 if not ok else self.reg.functions.experimentCount().call() - 1

    def end_experiment(self, exp_id: int, actual_rounds: int):
        self._transact(
            self.reg.functions.endExperiment(exp_id, actual_rounds),
            gas=100_000
        )

    # ── Per-round ─────────────────────────────────────────────
    def submit_contribution(self, client_idx: int, quality: float,
                            data_size: int, mean_data_size: float,
                            round_num: int):
        """
        Ghi đóng góp 1 client lên ContributionStore.
        quality: float [0, ∞) — Δloss, được clip về [0,1] trước khi scale
        data_size: int — số mẫu thực tế client dùng
        """
        addr = self.client_address(client_idx)
        # Chuẩn hóa quality về [0,1] bằng tanh để không bao giờ vượt 1
        q_norm    = float(np.tanh(quality))
        q_scaled  = int(np.clip(q_norm, 0.0, 1.0) * SCALE)
        # Data commitment = d_k / D_mean, clip [0, 5] để tránh outlier
        dc        = float(data_size) / max(float(mean_data_size), 1.0)
        dc_scaled = int(np.clip(dc, 0.0, 5.0) * SCALE)

        self._transact(
            self.store.functions.recordContribution(
                addr, q_scaled, dc_scaled, round_num
            ),
            gas=150_000
        )

    def get_reputation(self, client_idx: int) -> Tuple[float, bool]:
        """Trả về (reputation ∈ [0,1], is_honest: bool)."""
        addr = self.client_address(client_idx)
        try:
            rep_scaled, is_honest = self.store.functions.getReputation(addr).call()
            return float(rep_scaled) / SCALE, bool(is_honest)
        except Exception as e:
            logger.warning(f"getReputation failed for client {client_idx}: {e}")
            return 0.0, True  # fail-open: không phạt client nếu lỗi đọc

    def filter_and_distribute(
        self,
        n_clients: int,
        quality_scores: np.ndarray,
        data_sizes: np.ndarray,
        alpha: float,
        mean_data_size: float,
        round_num: int,
        pool_eth: float
    ) -> Dict[int, float]:
        """
        Bước 1: lọc P_honest qua reputation (binary gate)
        Bước 2: tính W_new cho P_honest (Python, Hybrid Normalization)
        Bước 3: gửi W_new lên RewardDistributor → phân phối ETH

        BUG5 FIX: distributeRewards nhận đúng 3 params
        (honestClients, weightsScaled, round) — không truyền thừa.

        Returns: {client_idx: w_new} chỉ cho P_honest
        """
        # ── Lọc P_honest ─────────────────────────────────────
        honest_indices: List[int]  = []
        honest_addrs:  List[str]   = []

        for i in range(n_clients):
            _, is_honest = self.get_reputation(i)
            if is_honest:
                honest_indices.append(i)
                honest_addrs.append(self.client_address(i))

        if not honest_indices:
            logger.warning(f"Round {round_num}: P_honest rỗng — bỏ qua phân phối")
            return {}

        # ── Tính W_new (trong Python, không trong contract) ───
        q_arr = quality_scores[honest_indices]
        d_arr = data_sizes[honest_indices].astype(float)
        w_new = compute_w_new(q_arr, d_arr, alpha, mean_data_size, self.contrib_cfg)

        # Sanity check — log warning thay vì assert để không crash production
        w_sum = float(w_new.sum())
        if abs(w_sum - 1.0) > 1e-4:
            logger.warning(f"Round {round_num}: W_new sum={w_sum:.6f}, normalizing")
            w_new = w_new / (w_sum + 1e-10)
        w_new = np.clip(w_new, 0.0, 1.0)

        w_scaled = [int(w * SCALE) for w in w_new]

        # Đảm bảo tổng w_scaled chính xác = SCALE (tránh rounding error)
        scaled_sum = sum(w_scaled)
        if scaled_sum == 0:
            logger.error(f"Round {round_num}: w_scaled sum=0, bỏ qua distribute")
            return {}
        if scaled_sum != SCALE and w_scaled:
            w_scaled[-1] = max(0, w_scaled[-1] + (SCALE - scaled_sum))

        # ── Gửi lên contract ──────────────────────────────────
        pool_wei = self.w3.to_wei(pool_eth, "ether")
        ok = self._transact(
            self.dist.functions.distributeRewards(
                honest_addrs, w_scaled, round_num
            ),
            gas=self.cfg.gas_limit,
            value_wei=pool_wei
        )
        if not ok:
            logger.error(f"Round {round_num}: distributeRewards thất bại")

        logger.info(
            f"Round {round_num}: distributed {pool_eth} ETH to "
            f"{len(honest_indices)} clients, α={alpha:.2f}"
        )
        return {honest_indices[i]: float(w_new[i]) for i in range(len(honest_indices))}
