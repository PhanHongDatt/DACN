require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: { optimizer: { enabled: true, runs: 200 } }
  },
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
      accounts: { mnemonic: "test test test test test test test test test test test junk" }
    }
  },
  paths: {
    sources:   "./contracts",
    tests:     "./tests",
    cache:     "./cache",
    artifacts: "./artifacts"
  }
};
