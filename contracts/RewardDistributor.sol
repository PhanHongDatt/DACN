// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./ContributionStore.sol";

/**
 * @title RewardDistributor
 * @notice Nhận W_new weights từ Python (đã tính Hybrid Normalization),
 *         phân phối ETH pool cho P_honest theo tỷ lệ W_new.
 *
 * Reputation CHỈ dùng để lọc P_honest (binary gate).
 * W_new quyết định bao nhiêu reward — không double-count.
 */
contract RewardDistributor {
    ContributionStore public store;
    address public owner;
    mapping(uint256 => bool) public roundPaid;

    event RewardDistributed(uint256 indexed round, uint256 totalPool, uint256 nClients);
    event ClientRewarded(address indexed client, uint256 amount, uint256 weightScaled);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _store) {
        store = ContributionStore(_store);
        owner = msg.sender;
    }

    /**
     * @param honestClients  Danh sách địa chỉ trong P_honest (đã lọc từ Python)
     * @param weightsScaled  W_new × 1e6 tương ứng từng client
     * @param round          Số vòng hiện tại (để emit event)
     *
     * BUG5 FIX: 3 params đúng với cách gọi từ blockchain.py
     */
    function distributeRewards(
        address[] calldata honestClients,
        uint256[] calldata weightsScaled,
        uint256 round
    ) external payable onlyOwner {
        require(honestClients.length == weightsScaled.length, "Length mismatch");
        require(honestClients.length > 0, "No clients");
        require(msg.value > 0, "No ETH sent");
        require(!roundPaid[round], "Round already paid");

        uint256 weightSum = 0;
        for (uint i = 0; i < weightsScaled.length; i++) {
            require(honestClients[i] != address(0), "Zero client");
            for (uint j = 0; j < i; j++) {
                require(honestClients[i] != honestClients[j], "Duplicate client");
            }
            weightSum += weightsScaled[i];
        }
        require(weightSum > 0, "Zero weight sum");
        roundPaid[round] = true;

        uint256 totalPool = msg.value;
        uint256 distributed = 0;

        for (uint i = 0; i < honestClients.length; i++) {
            uint256 share;
            if (i == honestClients.length - 1) {
                share = totalPool - distributed; // tránh rounding loss
            } else {
                share = (totalPool * weightsScaled[i]) / weightSum;
            }
            distributed += share;
            payable(honestClients[i]).transfer(share);
            emit ClientRewarded(honestClients[i], share, weightsScaled[i]);
        }

        emit RewardDistributed(round, totalPool, honestClients.length);
    }
}
