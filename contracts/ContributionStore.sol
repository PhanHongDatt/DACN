// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ContributionStore
 * @notice Lưu lịch sử đóng góp của mỗi client bằng circular buffer
 *         kích thước cố định W_SIZE=10. Tránh gas O(R) của dynamic array.
 *
 * Reputation tính từ DATA COMMITMENT (d_k / D_mean), không phải quality
 * score, để tránh double-counting khi nhân vào reward.
 */
contract ContributionStore {
    uint8  public constant WINDOW_SIZE   = 10;
    uint8  public constant MIN_WARMUP    = 5;      // vòng warm-up trước khi lọc
    uint256 public constant REP_THRESHOLD = 100000; // 0.1 × 1e6

    struct ClientData {
        uint256[10] qualityHistory;    // quality score × 1e6 (từ Python)
        uint256[10] dataCommitHistory; // (d_k / D_mean) × 1e6
        uint8   currentIndex;
        uint256 roundsRecorded;
        bool    registered;
    }

    mapping(address => ClientData) private clients;
    address[] public clientList;
    address   public owner;

    event ContributionRecorded(address indexed client, uint256 round, uint256 quality, uint256 dataCommit);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() { owner = msg.sender; }

    function registerClient(address client) external onlyOwner {
        if (!clients[client].registered) {
            clients[client].registered = true;
            clientList.push(client);
        }
    }

    function recordContribution(
        address  client,
        uint256  qualityScaled,    // quality score × 1e6
        uint256  dataCommitScaled, // (d_k / D_mean) × 1e6
        uint256  round
    ) external onlyOwner {
        require(clients[client].registered, "Client not registered");
        ClientData storage cd = clients[client];
        uint8 idx = cd.currentIndex;
        cd.qualityHistory[idx]    = qualityScaled;
        cd.dataCommitHistory[idx] = dataCommitScaled;
        cd.currentIndex           = (idx + 1) % WINDOW_SIZE;
        cd.roundsRecorded++;
        emit ContributionRecorded(client, round, qualityScaled, dataCommitScaled);
    }

    /**
     * @notice Tính reputation từ data commitment history (EWMA, β=0.9)
     * @return rep       Reputation value × 1e6
     * @return isHonest  true nếu đủ điều kiện tham gia reward round
     */
    function getReputation(address client)
        external view returns (uint256 rep, bool isHonest)
    {
        ClientData storage cd = clients[client];
        if (cd.roundsRecorded < MIN_WARMUP) return (0, true); // warm-up: chưa lọc

        uint8   filled      = cd.roundsRecorded < WINDOW_SIZE
                              ? uint8(cd.roundsRecorded) : WINDOW_SIZE;
        uint256 beta        = 900000; // 0.9 × 1e6
        uint256 weight      = 1000000;
        uint256 weightedSum = 0;
        uint256 totalWeight = 0;

        for (uint8 i = 0; i < filled; i++) {
            uint8 idx = (cd.currentIndex + WINDOW_SIZE - 1 - i) % WINDOW_SIZE;
            weightedSum += (weight * cd.dataCommitHistory[idx]) / 1000000;
            totalWeight += weight;
            weight       = (weight * beta) / 1000000;
        }

        rep      = totalWeight > 0 ? (weightedSum * 1000000) / totalWeight : 0;
        isHonest = rep >= REP_THRESHOLD;
    }

    function getRoundsRecorded(address client) external view returns (uint256) {
        return clients[client].roundsRecorded;
    }

    function getClientCount() external view returns (uint256) {
        return clientList.length;
    }
}
