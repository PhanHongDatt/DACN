#!/bin/bash
# run_all.sh — VM-friendly experiment launcher for FL/Blockchain/CSRA.

set -uo pipefail

ROUNDS_WAS_SET="${ROUNDS+x}"
LOG_DIR="${LOG_DIR:-./results/logs}"
SEED="${SEED:-42}"
N_CLIENTS="${N_CLIENTS:-10}"
TRIM_RATIO="${TRIM_RATIO:-0.1}"
ROUNDS="${ROUNDS:-50}"

MODE="full"
DRY_RUN=0
RESUME=0
INCLUDE_CIFAR=1
CHECK_ENV=1
CHECK_CHAIN=1

PASS=0
FAIL=0
SKIP=0
START_TIME=$(date +%s)

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_all.sh [mode] [options]

Modes:
  --smoke       3 no-blockchain runs, 3 rounds by default. Fast sanity check.
  --quick       16 MNIST runs, 10 rounds by default. Good before full matrix.
  --full        70 VM-friendly runs, 50 rounds by default. Default mode.
  --cifar-only  6 CIFAR-10 stress runs, 50 rounds by default.

Options:
  --dry-run          Print commands only.
  --resume           Skip runs that already have matching CSV logs.
  --no-cifar         In --full mode, skip CIFAR-10 stress runs (64 runs total).
  --skip-env-check   Do not run scripts/check_python.py before execution.
  --skip-chain-check Do not check local Hardhat RPC/deployed addresses.
  -h, --help         Show this help.

Useful overrides:
  ROUNDS=20 SEED=123 LOG_DIR=./results/logs bash experiments/run_all.sh --quick
  bash experiments/run_all.sh --full --resume
  bash experiments/run_all.sh --smoke --dry-run

Before --quick/--full blockchain runs:
  npx hardhat node
  npx hardhat run scripts/deploy.js --network localhost
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke) MODE="smoke"; CHECK_CHAIN=0 ;;
    --quick) MODE="quick" ;;
    --full) MODE="full" ;;
    --cifar-only) MODE="cifar-only" ;;
    --dry-run) DRY_RUN=1; CHECK_ENV=0; CHECK_CHAIN=0 ;;
    --resume) RESUME=1 ;;
    --no-cifar) INCLUDE_CIFAR=0 ;;
    --skip-env-check) CHECK_ENV=0 ;;
    --skip-chain-check) CHECK_CHAIN=0 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "$ROUNDS_WAS_SET" ]; then
  case "$MODE" in
    smoke) ROUNDS=3 ;;
    quick) ROUNDS=10 ;;
    *) ROUNDS=50 ;;
  esac
fi

mkdir -p "$LOG_DIR"

da_code() {
  case "$1" in
    1|1.0) echo "100" ;;
    0.5|.5) echo "050" ;;
    0.1|.1) echo "010" ;;
    0.05|.05) echo "005" ;;
    *) python -c "print(f'{int(round(float(\"$1\") * 100)):03d}')" ;;
  esac
}

alpha_code() {
  python -c "print(f'{int(round(float(\"$1\") * 10)):02d}')"
}

check_environment() {
  if [ "$CHECK_ENV" -eq 0 ]; then
    return 0
  fi
  echo ">>> Checking Python runtime"
  python scripts/check_python.py || {
    echo ""
    echo "Environment check failed. Use Python 3.11 and install requirements before running experiments." >&2
    echo "For command preview only, use: bash experiments/run_all.sh --dry-run" >&2
    exit 1
  }
}

check_chain() {
  if [ "$CHECK_CHAIN" -eq 0 ]; then
    return 0
  fi
  echo ">>> Checking local Hardhat blockchain"
  python - <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

addr_file = Path("fl/contract_addresses.json")
if not addr_file.exists():
    print("Missing fl/contract_addresses.json. Run deploy script first.", file=sys.stderr)
    sys.exit(1)

try:
    json.loads(addr_file.read_text())
except Exception as exc:
    print(f"Cannot read contract addresses: {exc}", file=sys.stderr)
    sys.exit(1)

payload = b'{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
req = urllib.request.Request(
    "http://127.0.0.1:8545",
    data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        if resp.status != 200:
            raise RuntimeError(f"RPC status={resp.status}")
except Exception as exc:
    print(f"Cannot connect to Hardhat RPC: {exc}", file=sys.stderr)
    sys.exit(1)
PY
  if [ "$?" -ne 0 ]; then
    echo ""
    echo "Blockchain check failed. Start and deploy contracts first:" >&2
    echo "  npx hardhat node" >&2
    echo "  npx hardhat run scripts/deploy.js --network localhost" >&2
    echo "Or use --smoke for no-blockchain sanity checks." >&2
    exit 1
  fi
}

run() {
  local desc="$1"; shift
  local runner="${RUNNER:-experiments/run_experiment.py}"
  echo ""
  echo "━━━ $desc ━━━"
  echo "CMD: python $runner $*"

  if [ "$RESUME" -eq 1 ]; then
    local ds="" sc="" cfg="" alpha="" dir_alpha="" prev=""
    for arg in "$@"; do
      case "$prev" in
        --dataset)         ds="$arg" ;;
        --scenario)        sc="$arg" ;;
        --config)          cfg="$arg" ;;
        --alpha)           alpha="$(alpha_code "$arg")" ;;
        --dirichlet-alpha) dir_alpha="$arg" ;;
      esac
      prev="$arg"
    done

    if [ -z "$alpha" ]; then
      case "$cfg" in
        A|TrimmedMean) alpha="00" ;;
        B) alpha="10" ;;
        *) alpha="05" ;;
      esac
    fi

    local log_cfg="$cfg"
    if [[ "$runner" == *"run_experiment_csra.py" && "$cfg" == "C" ]]; then
      log_cfg="C-CSRA"
      alpha="05"
    fi
    if [[ "$runner" == *"run_experiment_trimmed.py" ]]; then
      log_cfg="${cfg:-TrimmedMean}"
      alpha="00"
    fi

    local dir_part=""
    if [ -n "$dir_alpha" ]; then
      dir_part="_da$(da_code "$dir_alpha")"
    fi

    local pattern="${LOG_DIR}/${ds}_${sc}_${log_cfg}_a${alpha}${dir_part}_*.csv"
    if ls $pattern 1>/dev/null 2>&1; then
      echo "SKIP (log đã tồn tại)"
      SKIP=$((SKIP+1))
      return 0
    fi
  fi

  [ "$DRY_RUN" -eq 1 ] && return 0

  if python "$runner" "$@"; then
    PASS=$((PASS+1))
    echo "OK"
  else
    FAIL=$((FAIL+1))
    echo "FAIL — tiếp tục run tiếp theo" >&2
  fi
}

summary() {
  local elapsed=$(( $(date +%s) - START_TIME ))
  echo ""
  echo "════════════════════════════════════════"
  echo "  KẾT QUẢ THỰC NGHIỆM"
  echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
  printf "  Thời gian: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  echo "════════════════════════════════════════"
}
trap summary EXIT

common_args() {
  echo --n-clients "$N_CLIENTS" --n-rounds "$ROUNDS" --seed "$SEED" --log-dir "$LOG_DIR"
}

run_method_pair() {
  local ds="$1"; shift
  local label="$1"; shift

  run "$ds/$label/FedAvg-Clean" \
    --dataset "$ds" "$@" --config A --alpha 0.0 --no-blockchain $(common_args)
  run "$ds/$label/BlockchainQuality-Clean" \
    --dataset "$ds" "$@" --config B --alpha 1.0 $(common_args)
  RUNNER="experiments/run_experiment_trimmed.py" run "$ds/$label/TrimmedMean-Clean" \
    --dataset "$ds" "$@" --config TrimmedMean --trim-ratio "$TRIM_RATIO" --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Clean" \
    --dataset "$ds" "$@" --config C --alpha 0.5 $(common_args)

  run "$ds/$label/FedAvg-Attack" \
    --dataset "$ds" "$@" --config A --alpha 0.0 --no-blockchain --with-freeriders $(common_args)
  run "$ds/$label/BlockchainQuality-Attack" \
    --dataset "$ds" "$@" --config B --alpha 1.0 --with-freeriders $(common_args)
  RUNNER="experiments/run_experiment_trimmed.py" run "$ds/$label/TrimmedMean-Attack" \
    --dataset "$ds" "$@" --config TrimmedMean --trim-ratio "$TRIM_RATIO" --with-freeriders --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Defense" \
    --dataset "$ds" "$@" --config C --alpha 0.5 --with-freeriders $(common_args)
}

run_smoke() {
  run "smoke/FedAvg-K1" \
    --dataset mnist --scenario K1 --config A --alpha 0.0 --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_trimmed.py" run "smoke/TrimmedMean-K1" \
    --dataset mnist --scenario K1 --config TrimmedMean --trim-ratio "$TRIM_RATIO" --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "smoke/CSRA-K3-Attack-NoChain" \
    --dataset mnist --scenario K3 --dirichlet-alpha 0.1 --config C --alpha 0.5 --with-freeriders --no-blockchain $(common_args)
}

run_quick() {
  run_method_pair "mnist" "K1-IID" --scenario K1
  run_method_pair "mnist" "K3-Dirichlet-a0.1" --scenario K3 --dirichlet-alpha 0.1
}

run_full() {
  local datasets=("mnist" "fashion_mnist")
  local dirichlet_alphas=("0.5" "0.1")

  for ds in "${datasets[@]}"; do
    run_method_pair "$ds" "K1-IID" --scenario K1
    run_method_pair "$ds" "K2-WeakNonIID" --scenario K2
    for da in "${dirichlet_alphas[@]}"; do
      run_method_pair "$ds" "K3-Dirichlet-a$da" --scenario K3 --dirichlet-alpha "$da"
    done
  done

  if [ "$INCLUDE_CIFAR" -eq 1 ]; then
    run_cifar_stress
  fi
}

run_cifar_stress() {
  local ds="cifar10"
  local label="K3-Dirichlet-a0.1-Stress"
  local args=(--scenario K3 --dirichlet-alpha 0.1)

  run "$ds/$label/FedAvg-Clean" \
    --dataset "$ds" "${args[@]}" --config A --alpha 0.0 --no-blockchain $(common_args)
  run "$ds/$label/BlockchainQuality-Clean" \
    --dataset "$ds" "${args[@]}" --config B --alpha 1.0 $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Clean" \
    --dataset "$ds" "${args[@]}" --config C --alpha 0.5 $(common_args)

  run "$ds/$label/FedAvg-Attack" \
    --dataset "$ds" "${args[@]}" --config A --alpha 0.0 --no-blockchain --with-freeriders $(common_args)
  run "$ds/$label/BlockchainQuality-Attack" \
    --dataset "$ds" "${args[@]}" --config B --alpha 1.0 --with-freeriders $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Defense" \
    --dataset "$ds" "${args[@]}" --config C --alpha 0.5 --with-freeriders $(common_args)
}

echo "════════════════════════════════════════"
echo "  FL/Blockchain/CSRA Experiment Launcher"
echo "  Mode: $MODE"
echo "  Rounds mỗi run: $ROUNDS"
echo "  Seed: $SEED"
echo "  Log dir: $LOG_DIR"
echo "  Resume: $RESUME  Dry-run: $DRY_RUN"
echo "════════════════════════════════════════"

check_environment
check_chain

case "$MODE" in
  smoke) run_smoke ;;
  quick) run_quick ;;
  full) run_full ;;
  cifar-only) run_cifar_stress ;;
esac
