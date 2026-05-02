import numpy as np
import sys
import os

# Đảm bảo import được module fl
sys.path.append(os.getcwd())

from fl.blockchain import BlockchainBridge
from fl.config import BlockchainConfig, ContributionConfig

def test_full_blockchain_flow():
    """
    Kiểm tra luồng tích hợp với Blockchain mà không cần Ray.
    """
    print("Starting integration test...")
    b_cfg = BlockchainConfig()
    c_cfg = ContributionConfig()
    
    try:
        # Giả lập Bridge
        bridge = BlockchainBridge(b_cfg, c_cfg)
        print("✓ Bridge initialized.")
        
        # 1. Start experiment
        exp_id = bridge.start_experiment("mnist", "K1", 0.5, 3, 2)
        print(f"✓ Experiment started with ID: {exp_id}")
        
        # 2. Submit contributions
        print("Submitting contributions...")
        bridge.submit_contribution(0, 0.5, 1000, 1000, 1)
        bridge.submit_contribution(1, 0.4, 1000, 1000, 1)
        bridge.submit_contribution(2, 0.0, 0, 1000, 1)
        print("✓ Contributions submitted.")
        
        # 3. Check Reputation
        rep0, h0 = bridge.get_reputation(0)
        print(f"✓ Reputation check: client0_rep={rep0}, is_honest={h0}")
        
        # 4. Filter and Distribute
        print("Filtering and distributing rewards...")
        quality_scores = np.array([0.5, 0.4, 0.0])
        data_sizes = np.array([1000, 1000, 0])
        
        w_map = bridge.filter_and_distribute(
            n_clients=3,
            quality_scores=quality_scores,
            data_sizes=data_sizes,
            alpha=0.5,
            mean_data_size=1000.0,
            round_num=1,
            pool_eth=0.1 # Dùng lượng ETH nhỏ để test
        )
        
        print(f"✓ Rewards distributed. Weights: {w_map}")
        assert len(w_map) == 3
        assert abs(sum(w_map.values()) - 1.0) < 1e-5
        
        # 5. End experiment
        bridge.end_experiment(exp_id, 1)
        print("✓ Experiment ended.")
        print("\n>>> SUCCESS: Blockchain integration flow is STABLE. <<<")
        
    except Exception as e:
        print(f"\n>>> FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_full_blockchain_flow()
