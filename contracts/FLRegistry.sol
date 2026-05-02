// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FLRegistry
 * @notice Registry đơn giản để lưu metadata thực nghiệm on-chain:
 *         dataset, scenario, alpha, số vòng. Dùng cho logging và audit.
 */
contract FLRegistry {
    struct Experiment {
        string  dataset;    // "mnist", "fashion_mnist", "cifar10"
        string  scenario;   // "K1", "K2", "K3"
        uint256 alpha;      // α × 1e6
        uint256 nClients;
        uint256 nRounds;
        uint256 startTime;
        bool    active;
    }

    mapping(uint256 => Experiment) public experiments;
    uint256 public experimentCount;
    address public owner;

    event ExperimentStarted(uint256 indexed id, string dataset, string scenario);
    event ExperimentEnded(uint256 indexed id, uint256 totalRounds);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    constructor() { owner = msg.sender; }

    function startExperiment(
        string calldata dataset,
        string calldata scenario,
        uint256 alpha,
        uint256 nClients,
        uint256 nRounds
    ) external onlyOwner returns (uint256 id) {
        id = experimentCount++;
        experiments[id] = Experiment(dataset, scenario, alpha, nClients, nRounds, block.timestamp, true);
        emit ExperimentStarted(id, dataset, scenario);
    }

    function endExperiment(uint256 id, uint256 actualRounds) external onlyOwner {
        experiments[id].active = false;
        emit ExperimentEnded(id, actualRounds);
    }
}
