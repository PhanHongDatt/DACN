/**
 * healthcheck.js — Kiểm tra toàn bộ hệ thống trước khi chạy thực nghiệm.
 * Chạy: node scripts/healthcheck.js
 */
const { ethers } = require("hardhat");
const fs = require("fs");

async function main() {
  const issues = [];
  const ok = [];

  // 1. Check network
  try {
    const network = await ethers.provider.getNetwork();
    ok.push(`Network: chainId=${network.chainId}`);
  } catch (e) {
    issues.push(`Network unreachable: ${e.message}`);
  }

  // 2. Check accounts
  const signers = await ethers.getSigners();
  if (signers.length < 12) {
    issues.push(`Need >= 12 accounts, got ${signers.length}`);
  } else {
    ok.push(`Accounts: ${signers.length} available`);
  }

  // 3. Check balances
  const ownerBal = await ethers.provider.getBalance(signers[0].address);
  if (ownerBal < ethers.parseEther("10")) {
    issues.push(`Owner balance too low: ${ethers.formatEther(ownerBal)} ETH`);
  } else {
    ok.push(`Owner balance: ${ethers.formatEther(ownerBal)} ETH`);
  }

  // 4. Check contract addresses file
  const addrFile = "./fl/contract_addresses.json";
  if (!fs.existsSync(addrFile)) {
    issues.push("fl/contract_addresses.json not found — run: npm run deploy");
  } else {
    const addrs = JSON.parse(fs.readFileSync(addrFile));
    const contractNames = ["ContributionStore", "RewardDistributor", "FLRegistry"];
    ok.push(`Contracts: ${contractNames.join(", ")}`);

    // 5. Check contracts exist on chain
    for (const name of contractNames) {
      const addr = addrs[name];
      if (!addr || !addr.startsWith("0x")) {
        issues.push(`${name} address missing or invalid — run: npm run deploy`);
        continue;
      }
      const code = await ethers.provider.getCode(addr);
      if (code === "0x") {
        issues.push(`${name} has no code at ${addr} — redeploy needed`);
      }
    }
  }

  // 6. Check artifacts
  const abiFiles = [
    "artifacts/contracts/ContributionStore.sol/ContributionStore.json",
    "artifacts/contracts/RewardDistributor.sol/RewardDistributor.json",
    "artifacts/contracts/FLRegistry.sol/FLRegistry.json",
  ];
  for (const f of abiFiles) {
    if (!fs.existsSync(f)) {
      issues.push(`ABI missing: ${f} — run: npx hardhat compile`);
    }
  }
  if (issues.filter(i => i.includes("ABI")).length === 0) {
    ok.push("All ABIs present");
  }

  // Summary
  console.log("\n=== Healthcheck ===");
  ok.forEach(m => console.log(`  ✓ ${m}`));
  if (issues.length > 0) {
    issues.forEach(m => console.log(`  ✗ ${m}`));
    console.log(`\n  ${issues.length} issue(s) found. Fix before running experiments.\n`);
    process.exit(1);
  } else {
    console.log("\n  All checks passed. Ready to run experiments.\n");
  }
}

main().catch(e => { console.error(e); process.exit(1); });
