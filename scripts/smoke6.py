"""Smoke test 6 cells × MNIST × K1 × seed 42, n_rounds=3."""
import os
import subprocess
import sys
import time
from pathlib import Path

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

CELLS = [
    ("M1", "fedavg",   "equal",   []),
    ("M2", "fedavg",   "data",    []),
    ("M3", "fedavg",   "quality", []),
    ("M4", "fedavg",   "csra",    ["--beta", "0.5", "--gamma", "0.3", "--delta", "0.2"]),
    ("M5", "csra_dcd", "equal",   []),
    ("M6", "csra_dcd", "csra",    ["--beta", "0.5", "--gamma", "0.3", "--delta", "0.2"]),
]


def run_cell(name: str, agg: str, reward: str, extra: list[str]) -> tuple[int, float, str]:
    cmd = [
        sys.executable, "-m", "experiments.run_experiment",
        "--dataset", "mnist", "--scenario", "K1",
        "--aggregation", agg, "--reward-policy", reward,
        "--seed", "42", "--n-rounds", "3", "--n-clients", "5",
        "--local-epochs", "1", "--no-blockchain",
        "--log-dir", "results/smoke6",
    ] + extra
    print(f"=========== {name}: {agg}+{reward} ===========", flush=True)
    t0 = time.time()
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    dt = time.time() - t0
    last_acc = ""
    for line in (res.stdout or "").splitlines():
        if "global accuracy" in line.lower():
            last_acc = line.strip().split("]")[-1].strip()
    print(f"  rc={res.returncode}  time={dt:.1f}s  last={last_acc}", flush=True)
    if res.returncode != 0:
        print("STDERR tail:", (res.stderr or "")[-1200:])
    return res.returncode, dt, last_acc


def main() -> int:
    Path("results/smoke6").mkdir(parents=True, exist_ok=True)
    results = []
    for name, agg, reward, extra in CELLS:
        rc, dt, acc = run_cell(name, agg, reward, extra)
        if rc != 0:
            print(f"FAILED: {name}")
            return 1
        results.append((name, agg, reward, acc, dt))
    print("\n=== ALL 6 CELLS PASSED ===")
    for n, a, r, acc, dt in results:
        print(f"  {n}  {a}+{r:8s}  {acc}  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
