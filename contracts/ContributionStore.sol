// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ContributionStore
 * @notice Lưu lịch sử đóng góp của mỗi client bằng circular buffer
 *         kích thước cố định W_SIZE=10. Tránh gas O(R) của dynamic array.
 *
 * Reputation tính từ EWMA của capped DATA COMMITMENT và quality score.
 * Contract chỉ dùng fixed-point audit signals do Python gửi lên; ML detector
 * phức tạp vẫn nằm off-chain.
 */
contract ContributionStore {
    uint8  public constant WINDOW_SIZE   = 10;
    uint8  public constant MIN_WARMUP    = 5;      // vòng warm-up trước khi lọc
    uint256 public constant REP_THRESHOLD = 100000; // 0.1 × 1e6
    uint256 public constant SCALE = 1000000;
    uint256 public constant REP_DECAY = 900000; // 0.9 × 1e6
    uint256 public constant DATA_WEIGHT = 700000; // 70%
    uint256 public constant QUALITY_WEIGHT = 300000; // 30%

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

    function _capToScale(uint256 value) private pure returns (uint256) {
        return value > SCALE ? SCALE : value;
    }

    /**
     * @notice Tính reputation từ data commitment + quality history (EWMA, β=0.9)
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
        uint256 weight      = 1000000;
        uint256 weightedSum = 0;
        uint256 totalWeight = 0;

        for (uint8 i = 0; i < filled; i++) {
            uint8 idx = (cd.currentIndex + WINDOW_SIZE - 1 - i) % WINDOW_SIZE;
            uint256 dataScore = _capToScale(cd.dataCommitHistory[idx]);
            uint256 qualityScore = _capToScale(cd.qualityHistory[idx]);
            uint256 roundScore = (
                (dataScore * DATA_WEIGHT) + (qualityScore * QUALITY_WEIGHT)
            ) / SCALE;
            weightedSum += (weight * roundScore) / SCALE;
            totalWeight += weight;
            weight       = (weight * REP_DECAY) / SCALE;
        }

        rep      = totalWeight > 0 ? (weightedSum * SCALE) / totalWeight : 0;
        isHonest = rep >= REP_THRESHOLD;
    }

    function getRoundsRecorded(address client) external view returns (uint256) {
        return clients[client].roundsRecorded;
    }

    function getClientCount() external view returns (uint256) {
        return clientList.length;
    }
}
