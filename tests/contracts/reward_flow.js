const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FL reward contracts", function () {
  async function deployContracts() {
    const [owner, clientA, clientB] = await ethers.getSigners();

    const Store = await ethers.getContractFactory("ContributionStore");
    const store = await Store.deploy();
    await store.waitForDeployment();

    const Dist = await ethers.getContractFactory("RewardDistributor");
    const dist = await Dist.deploy(await store.getAddress());
    await dist.waitForDeployment();

    const Registry = await ethers.getContractFactory("FLRegistry");
    const registry = await Registry.deploy();
    await registry.waitForDeployment();

    return { owner, clientA, clientB, store, dist, registry };
  }

  it("records contributions and computes reputation after warmup", async function () {
    const { clientA, store } = await deployContracts();

    await store.registerClient(clientA.address);

    let [rep, isHonest] = await store.getReputation(clientA.address);
    expect(rep).to.equal(0n);
    expect(isHonest).to.equal(true);

    for (let round = 1; round <= 5; round++) {
      await store.recordContribution(clientA.address, 500000n, 1000000n, round);
    }

    [rep, isHonest] = await store.getReputation(clientA.address);
    expect(rep).to.equal(1000000n);
    expect(isHonest).to.equal(true);
    expect(await store.getRoundsRecorded(clientA.address)).to.equal(5n);
  });

  it("distributes the reward pool according to scaled weights", async function () {
    const { clientA, clientB, store, dist } = await deployContracts();

    await store.registerClient(clientA.address);
    await store.registerClient(clientB.address);

    const beforeA = await ethers.provider.getBalance(clientA.address);
    const beforeB = await ethers.provider.getBalance(clientB.address);
    const pool = ethers.parseEther("1.0");

    await dist.distributeRewards(
      [clientA.address, clientB.address],
      [300000n, 700000n],
      1,
      { value: pool }
    );

    const afterA = await ethers.provider.getBalance(clientA.address);
    const afterB = await ethers.provider.getBalance(clientB.address);

    expect(afterA - beforeA).to.equal(ethers.parseEther("0.3"));
    expect(afterB - beforeB).to.equal(ethers.parseEther("0.7"));
  });

  it("tracks experiment lifecycle metadata", async function () {
    const { registry } = await deployContracts();

    await registry.startExperiment("mnist", "K1", 500000n, 10, 50);
    expect(await registry.experimentCount()).to.equal(1n);

    const exp = await registry.experiments(0);
    expect(exp.dataset).to.equal("mnist");
    expect(exp.scenario).to.equal("K1");
    expect(exp.active).to.equal(true);

    await registry.endExperiment(0, 50);
    const ended = await registry.experiments(0);
    expect(ended.active).to.equal(false);
  });
});
