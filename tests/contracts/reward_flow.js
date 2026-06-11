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
    expect(rep).to.equal(850000n);
    expect(isHonest).to.equal(true);
    expect(await store.getRoundsRecorded(clientA.address)).to.equal(5n);
  });

  it("uses quality in reputation instead of data commitment only", async function () {
    const { clientA, clientB, store } = await deployContracts();

    await store.registerClient(clientA.address);
    await store.registerClient(clientB.address);

    for (let round = 1; round <= 5; round++) {
      await store.recordContribution(clientA.address, 1000000n, 1000000n, round);
      await store.recordContribution(clientB.address, 0n, 1000000n, round);
    }

    const [repA, honestA] = await store.getReputation(clientA.address);
    const [repB, honestB] = await store.getReputation(clientB.address);

    expect(repA).to.equal(1000000n);
    expect(repB).to.equal(700000n);
    expect(repB).to.be.lessThan(repA);
    expect(honestA).to.equal(true);
    expect(honestB).to.equal(true);
  });

  it("caps oversized data commitment in reputation", async function () {
    const { clientA, store } = await deployContracts();

    await store.registerClient(clientA.address);

    for (let round = 1; round <= 5; round++) {
      await store.recordContribution(clientA.address, 1000000n, 5000000n, round);
    }

    const [rep, isHonest] = await store.getReputation(clientA.address);

    expect(rep).to.equal(1000000n);
    expect(isHonest).to.equal(true);
  });

  it("marks repeated zero contribution as not honest after warmup", async function () {
    const { clientA, store } = await deployContracts();

    await store.registerClient(clientA.address);

    for (let round = 1; round <= 5; round++) {
      await store.recordContribution(clientA.address, 0n, 0n, round);
    }

    const [rep, isHonest] = await store.getReputation(clientA.address);

    expect(rep).to.equal(0n);
    expect(isHonest).to.equal(false);
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

  it("does not pay clients omitted from the honest client list", async function () {
    const { clientA, clientB, store, dist } = await deployContracts();

    await store.registerClient(clientA.address);
    await store.registerClient(clientB.address);

    const beforeA = await ethers.provider.getBalance(clientA.address);
    const beforeB = await ethers.provider.getBalance(clientB.address);
    const pool = ethers.parseEther("1.0");

    await dist.distributeRewards(
      [clientA.address],
      [1000000n],
      1,
      { value: pool }
    );

    const afterA = await ethers.provider.getBalance(clientA.address);
    const afterB = await ethers.provider.getBalance(clientB.address);

    expect(afterA - beforeA).to.equal(pool);
    expect(afterB - beforeB).to.equal(0n);
  });

  it("rejects double payout for the same round", async function () {
    const { clientA, store, dist } = await deployContracts();
    const pool = ethers.parseEther("1.0");

    await store.registerClient(clientA.address);
    await dist.distributeRewards(
      [clientA.address],
      [1000000n],
      7,
      { value: pool }
    );

    expect(await dist.roundPaid(7)).to.equal(true);
    await expect(
      dist.distributeRewards(
        [clientA.address],
        [1000000n],
        7,
        { value: pool }
      )
    ).to.be.revertedWith("Round already paid");
  });

  it("rejects invalid reward distribution calls", async function () {
    const { clientA, clientB, dist } = await deployContracts();
    const pool = ethers.parseEther("1.0");

    await expect(
      dist
        .connect(clientA)
        .distributeRewards([clientA.address], [1000000n], 1, { value: pool })
    ).to.be.revertedWith("Not owner");

    await expect(
      dist.distributeRewards([], [], 1, { value: pool })
    ).to.be.revertedWith("No clients");

    await expect(
      dist.distributeRewards([ethers.ZeroAddress], [1000000n], 1, { value: pool })
    ).to.be.revertedWith("Zero client");

    await expect(
      dist.distributeRewards(
        [clientA.address, clientA.address],
        [500000n, 500000n],
        1,
        { value: pool }
      )
    ).to.be.revertedWith("Duplicate client");

    await expect(
      dist.distributeRewards(
        [clientA.address, clientB.address],
        [1000000n],
        1,
        { value: pool }
      )
    ).to.be.revertedWith("Length mismatch");
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
