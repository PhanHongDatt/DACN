/**
 * fund_accounts.js — Cấp ETH cho 10 client accounts từ account[0]
 * Chạy sau deploy: npx hardhat run scripts/fund_accounts.js --network localhost
 */
const { ethers } = require("hardhat");

async function main() {
  const signers = await ethers.getSigners();
  const owner   = signers[0];

  console.log("Funding 10 client accounts with 10 ETH each...");
  for (let i = 1; i <= 10; i++) {
    const tx = await owner.sendTransaction({
      to:    signers[i].address,
      value: ethers.parseEther("10.0")
    });
    await tx.wait();
    const bal = ethers.formatEther(await ethers.provider.getBalance(signers[i].address));
    console.log(`  Account[${i}] ${signers[i].address}: ${bal} ETH`);
  }
  console.log("Done.");
}

main().catch(e => { console.error(e); process.exit(1); });
