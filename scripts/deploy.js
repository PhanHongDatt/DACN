/**
 * deploy.js — Hardhat deploy script
 * Chạy: npx hardhat run scripts/deploy.js --network localhost
 */
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);
  console.log("Balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)), "ETH");

  // 1. ContributionStore
  const Store = await ethers.getContractFactory("ContributionStore");
  const store = await Store.deploy();
  await store.waitForDeployment();
  console.log("ContributionStore:", await store.getAddress());

  // 2. RewardDistributor (depends on store)
  const Dist = await ethers.getContractFactory("RewardDistributor");
  const dist = await Dist.deploy(await store.getAddress());
  await dist.waitForDeployment();
  console.log("RewardDistributor:", await dist.getAddress());

  // 3. FLRegistry
  const Reg = await ethers.getContractFactory("FLRegistry");
  const reg = await Reg.deploy();
  await reg.waitForDeployment();
  console.log("FLRegistry:", await reg.getAddress());

  // Lưu addresses ra file để Python đọc
  const addresses = {
    ContributionStore:  await store.getAddress(),
    RewardDistributor:  await dist.getAddress(),
    FLRegistry:         await reg.getAddress(),
    deployer:           deployer.address,
    network:            (await ethers.provider.getNetwork()).name,
    deployedAt:         new Date().toISOString()
  };

  const outPath = path.join(__dirname, "../fl/contract_addresses.json");
  fs.writeFileSync(outPath, JSON.stringify(addresses, null, 2));
  console.log("\nAddresses saved to fl/contract_addresses.json");
  console.log(JSON.stringify(addresses, null, 2));
}

main().catch(e => { console.error(e); process.exit(1); });
