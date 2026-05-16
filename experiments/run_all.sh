#!/bin/bash
# run_all.sh — Experiment launcher cho ablation matrix mới (schema v2).
#
# Thay thế hoàn toàn version legacy. Sử dụng CLI mới của unified runner
# (--aggregation, --reward-policy, --attack), filename schema v2.
#
# Reference: docs/PLAN.md §5 (Ma trận thực nghiệm), §9 (Lộ trình).

set -uo pipefail

# ── Defaults (có thể override qua biến môi trường) ───────────────────────────
ROUNDS_WAS_SET="${ROUNDS+x}"
LOG_DIR="${LOG_DIR:-./results/logs}"
N_CLIENTS="${N_CLIENTS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"         # Đã giảm xuống 1 (xem PLAN §10.3 đòn bẩy 1)
TRIM_RATIO="${TRIM_RATIO:-0.1}"
ROUNDS="${ROUNDS:-30}"                    # Đã giảm 50→30 (PLAN §10.3 đòn bẩy 3)
SEEDS="${SEEDS:-42,123,2024}"             # clean seeds, comma-separated
ATTACK_SEEDS="${ATTACK_SEEDS:-42,123}"    # attack seeds (ít hơn để tiết kiệm)
SWEEP_SEEDS="${SWEEP_SEEDS:-42,123,2024}" # β-sweep seeds

# CSRAReward default weights (xem PLAN.md §4.2)
BETA="${BETA:-0.5}"
GAMMA="${GAMMA:-0.3}"
DELTA="${DELTA:-0.2}"

# β sweep values (γ:δ tỉ lệ 3:2 cố định)
SWEEP_BETAS="${SWEEP_BETAS:-0.3,0.5,0.7}"

# Attack client IDs (default: 2 client cuối cho mọi attack)
ATTACK_IDS="${ATTACK_IDS:-8,9}"

# Data heterogeneity pattern — cần thiết cho M2/M4 trên K1/K2
# (xem PLAN.md §5 + commit message "data imbalance feature")
DATA_IMBALANCE="${DATA_IMBALANCE:-lognormal}"

# Parallelism (cho VM nhiều core)
# Khuyến nghị cho 8 core VM: PARALLEL=4 NUM_THREADS=2
PARALLEL="${PARALLEL:-1}"
NUM_THREADS="${NUM_THREADS:-0}"  # 0 = PyTorch default

# ── Mode ─────────────────────────────────────────────────────────────────────
MODE=""
DRY_RUN=0
RESUME=0
INCLUDE_CIFAR=1
INCLUDE_CLEAN=1
INCLUDE_ATTACK=1
INCLUDE_SWEEP=1
CHECK_ENV=1
CHECK_CHAIN=0           # blockchain mặc định off (audit-only, optional)
NO_BLOCKCHAIN_FLAG="--no-blockchain"

# Counters
PASS=0
FAIL=0
SKIP=0
START_TIME=$(date +%s)

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_all.sh <mode> [options]

Modes:
  --smoke         6 cells × MNIST × K1 × 1 seed × 3 rounds.
                  Sanity check sau refactor. Không blockchain.
  --quick         6 cells × MNIST × {K1, K3@0.1} × seed 42 × 10 rounds.
                  ~12 runs. Bước trung gian trước full matrix.
  --full          Full ablation matrix theo PLAN.md (~458 runs).
                  3 datasets × 4 scenarios × 6 cells × 3 seeds (clean)
                  + attack matrix + β sweep.

Subsets (modify --full / --quick):
  --clean-only       Bỏ qua attack matrix (chỉ clean runs).
  --attack-only      Bỏ qua clean matrix (chỉ attack runs).
  --sweep-only       Chỉ chạy β sweep.
  --no-cifar         Bỏ qua CIFAR-10 (cho VM yếu).
  --with-blockchain  Bật blockchain audit (default: tắt). Phải có Hardhat node.

Other options:
  --dry-run          In lệnh không chạy.
  --resume           Skip run đã có CSV trong LOG_DIR.
  --skip-env-check   Bỏ qua check Python env.
  -h, --help         Help.

Performance overrides (biến môi trường):
  PARALLEL=4 NUM_THREADS=2   # 4 cells song song, 2 thread/job (cho VM 8 core)
  ROUNDS=25 LOCAL_EPOCHS=1   # giảm rounds + epochs để chạy nhanh hơn
  SEEDS=42,123               # giảm seed count

Useful overrides (biến môi trường):
  SEEDS=42,123      ROUNDS=20  N_CLIENTS=10  LOG_DIR=./results/exp1
  BETA=0.5 GAMMA=0.3 DELTA=0.2
  ATTACK_IDS=7,8,9

Examples:
  bash experiments/run_all.sh --smoke --dry-run
  bash experiments/run_all.sh --quick
  bash experiments/run_all.sh --full --no-cifar --resume
  bash experiments/run_all.sh --full --attack-only --resume

  # VM 8 core: 4 parallel cells, 2 threads mỗi cell (rút ~4x time)
  PARALLEL=4 NUM_THREADS=2 \\
    bash experiments/run_all.sh --full --no-cifar --resume
EOF
}

# ── Parse args ───────────────────────────────────────────────────────────────
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)      MODE="smoke"; CHECK_CHAIN=0 ;;
    --quick)      MODE="quick" ;;
    --full)       MODE="full" ;;

    --clean-only)  INCLUDE_ATTACK=0; INCLUDE_SWEEP=0 ;;
    --attack-only) INCLUDE_CLEAN=0;  INCLUDE_SWEEP=0 ;;
    --sweep-only)  INCLUDE_CLEAN=0;  INCLUDE_ATTACK=0 ;;
    --no-cifar)    INCLUDE_CIFAR=0 ;;

    --with-blockchain)
      NO_BLOCKCHAIN_FLAG=""
      CHECK_CHAIN=1
      ;;

    --dry-run) DRY_RUN=1; CHECK_ENV=0; CHECK_CHAIN=0 ;;
    --resume) RESUME=1 ;;
    --skip-env-check)  CHECK_ENV=0 ;;
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

if [ -z "$MODE" ]; then
  echo "Error: must specify a mode (--smoke | --quick | --full)" >&2
  usage >&2
  exit 2
fi

# Default rounds per mode
if [ -z "$ROUNDS_WAS_SET" ]; then
  case "$MODE" in
    smoke) ROUNDS=3 ;;
    quick) ROUNDS=10 ;;
    *)     ROUNDS=30 ;;   # PLAN §10.3 đòn bẩy 3: giảm 50→30 cho MNIST/F-MNIST
  esac
fi

mkdir -p "$LOG_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

# Convert float to 2-digit code (0.5 → "50", 0.3 → "30")
weight_code() {
  python -c "print(f'{int(round(float(\"$1\") * 100)):02d}')"
}

# Convert dirichlet to 3-digit code (0.1 → "010")
da_code() {
  case "$1" in
    1|1.0) echo "100" ;;
    0.5|.5) echo "050" ;;
    0.1|.1) echo "010" ;;
    0.05|.05) echo "005" ;;
    *) python -c "print(f'{int(round(float(\"$1\") * 100)):03d}')" ;;
  esac
}

check_environment() {
  if [ "$CHECK_ENV" -eq 0 ]; then
    return 0
  fi
  echo ">>> Checking Python runtime"
  python scripts/check_python.py || {
    echo ""
    echo "Environment check failed. Install dependencies hoặc dùng --dry-run." >&2
    exit 1
  }
}

check_chain() {
  if [ "$CHECK_CHAIN" -eq 0 ]; then
    return 0
  fi
  echo ">>> Checking local Hardhat blockchain"
  python - <<'PY'
import json, sys, urllib.request
from pathlib import Path
addr_file = Path("fl/contract_addresses.json")
if not addr_file.exists():
    print("Missing fl/contract_addresses.json. Deploy contracts trước.", file=sys.stderr)
    sys.exit(1)
payload = b'{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
req = urllib.request.Request(
    "http://127.0.0.1:8545", data=payload,
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
    echo "Blockchain check failed. Start contracts hoặc bỏ --with-blockchain." >&2
    exit 1
  fi
}

# Tách CSV danh sách seeds thành array
parse_seeds() {
  echo "$1" | tr ',' '\n'
}

# ── Run function: 1 cell ─────────────────────────────────────────────────────
# Args:
#   $1 desc          : Mô tả run (in lúc chạy)
#   $2 dataset       : mnist | fashion_mnist | cifar10
#   $3 scenario      : K1 | K2 | K3
#   $4 dirichlet_alpha (string, empty cho non-K3)
#   $5 agg           : fedavg | trimmed | csra_dcd
#   $6 reward        : equal | data | quality | csra
#   $7 beta gamma delta (3 floats, set 0 nếu reward != csra)
#   $8 attack        : clean | free_rider | lazy | label_noise | sign_flip
#   $9 seed          : int
run() {
  local desc="$1"
  local ds="$2"
  local sc="$3"
  local da="$4"
  local agg="$5"
  local reward="$6"
  local beta="$7"
  local gamma="$8"
  local delta="$9"
  local attack="${10}"
  local seed="${11}"

  # ── Resume check: thử match filename ──────────────────────────────────────
  if [ "$RESUME" -eq 1 ]; then
    local b_code g_code d_code da_part=""
    b_code=$(weight_code "$beta")
    g_code=$(weight_code "$gamma")
    d_code=$(weight_code "$delta")
    if [ -n "$da" ]; then
      da_part="_da$(da_code "$da")"
    fi
    local pattern="${LOG_DIR}/${ds}_${sc}${da_part}_${agg}_${reward}_b${b_code}g${g_code}d${d_code}_s${seed}_${attack}_*.csv"
    if ls $pattern 1>/dev/null 2>&1; then
      echo "SKIP — $desc (CSV đã có)"
      SKIP=$((SKIP+1))
      return 0
    fi
  fi

  local cmd_args=(
    --dataset "$ds"
    --scenario "$sc"
    --aggregation "$agg"
    --reward-policy "$reward"
    --beta "$beta" --gamma "$gamma" --delta "$delta"
    --attack "$attack"
    --attack-client-ids "$ATTACK_IDS"
    --seed "$seed"
    --n-clients "$N_CLIENTS"
    --n-rounds "$ROUNDS"
    --local-epochs "$LOCAL_EPOCHS"
    --trim-ratio "$TRIM_RATIO"
    --log-dir "$LOG_DIR"
    --num-threads "$NUM_THREADS"
    --data-imbalance "$DATA_IMBALANCE"
  )
  if [ -n "$da" ]; then
    cmd_args+=(--dirichlet-alpha "$da")
  fi
  if [ -n "$NO_BLOCKCHAIN_FLAG" ]; then
    cmd_args+=($NO_BLOCKCHAIN_FLAG)
  fi

  echo ""
  echo "━━━ $desc ━━━"
  echo "CMD: python -m experiments.run_experiment ${cmd_args[*]}"

  [ "$DRY_RUN" -eq 1 ] && return 0

  if [ "$PARALLEL" -gt 1 ]; then
    # Parallel mode: chờ slot trống rồi launch background
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
      sleep 1
    done
    (
      if python -m experiments.run_experiment "${cmd_args[@]}" > /dev/null 2>&1; then
        echo "[OK]   $desc"
      else
        echo "[FAIL] $desc" >&2
      fi
    ) &
  else
    if python -m experiments.run_experiment "${cmd_args[@]}"; then
      PASS=$((PASS+1))
      echo "OK"
    else
      FAIL=$((FAIL+1))
      echo "FAIL — tiếp tục run tiếp theo" >&2
    fi
  fi
}

# 6 cells của ablation matrix (xem PLAN.md §5.1)
# Args: dataset, scenario, dirichlet, attack, seed, label
run_six_cells() {
  local ds="$1" sc="$2" da="$3" attack="$4" seed="$5" label="$6"

  run "$label/M1 FedAvg+EqualSplit"       "$ds" "$sc" "$da" fedavg   equal   0 0 0 "$attack" "$seed"
  run "$label/M2 FedAvg+DataSize"         "$ds" "$sc" "$da" fedavg   data    0 0 0 "$attack" "$seed"
  run "$label/M3 FedAvg+QualityOnly"      "$ds" "$sc" "$da" fedavg   quality 0 0 0 "$attack" "$seed"
  run "$label/M4 FedAvg+CSRAReward"       "$ds" "$sc" "$da" fedavg   csra    "$BETA" "$GAMMA" "$DELTA" "$attack" "$seed"
  run "$label/M5 CSRA-DCD+EqualSplit"     "$ds" "$sc" "$da" csra_dcd equal   0 0 0 "$attack" "$seed"
  run "$label/M6 CSRA-DCD+CSRAReward"     "$ds" "$sc" "$da" csra_dcd csra    "$BETA" "$GAMMA" "$DELTA" "$attack" "$seed"
}

# Reduced cells cho CIFAR-10 attack (PLAN §5.4 — chỉ 4 cells)
run_four_cells_cifar_attack() {
  local ds="$1" sc="$2" da="$3" attack="$4" seed="$5" label="$6"

  run "$label/M1 FedAvg+EqualSplit"      "$ds" "$sc" "$da" fedavg   equal   0 0 0 "$attack" "$seed"
  run "$label/M3 FedAvg+QualityOnly"     "$ds" "$sc" "$da" fedavg   quality 0 0 0 "$attack" "$seed"
  run "$label/M5 CSRA-DCD+EqualSplit"    "$ds" "$sc" "$da" csra_dcd equal   0 0 0 "$attack" "$seed"
  run "$label/M6 CSRA-DCD+CSRAReward"    "$ds" "$sc" "$da" csra_dcd csra    "$BETA" "$GAMMA" "$DELTA" "$attack" "$seed"
}

# β sweep cho CSRAReward (chỉ trên M4 — FedAvg+CSRA)
# Sweep β ∈ {0.3, 0.5, 0.7} với γ:δ = 3:2 cố định
run_beta_sweep_one() {
  local ds="$1" sc="$2" da="$3" seed="$4" beta="$5"
  # Khi β thay đổi, γ:δ giữ tỷ lệ 3:2:
  #   γ = (1 - β) × 3/5
  #   δ = (1 - β) × 2/5
  local gamma delta
  gamma=$(python -c "print(round((1 - $beta) * 3/5, 4))")
  delta=$(python -c "print(round((1 - $beta) * 2/5, 4))")
  run "sweep/$ds/$sc/β=$beta" "$ds" "$sc" "$da" fedavg csra "$beta" "$gamma" "$delta" clean "$seed"
}

# ── Mode implementations ─────────────────────────────────────────────────────

run_smoke() {
  # 6 cells × MNIST × K1 × 1 seed × 3 rounds
  local SEED_LIST="42"
  ROUNDS=${ROUNDS:-3}
  for seed in $(parse_seeds "$SEED_LIST"); do
    run_six_cells mnist K1 "" clean "$seed" "smoke[seed=$seed]"
  done
}

run_quick() {
  # 6 cells × MNIST × {K1, K3@0.1} × 1 seed × 10 rounds
  local SEED_LIST="42"
  ROUNDS=${ROUNDS:-10}
  for seed in $(parse_seeds "$SEED_LIST"); do
    run_six_cells mnist K1 ""    clean "$seed" "quick/K1[seed=$seed]"
    run_six_cells mnist K3 "0.1" clean "$seed" "quick/K3@0.1[seed=$seed]"
  done
}

run_full() {
  # Per PLAN.md §5.3:
  # - Clean: 6 cells × 3 ds × {K1, K2, K3@0.1, K3@0.5} × 3 seeds = 216
  # - Attack: 6 cells × {mnist, fashion} × {K2, K3@0.1} × 4 attacks × 2 seeds = 192
  # - Attack CIFAR: 4 cells × cifar × K3@0.1 × 4 attacks × 2 seeds = 32
  # - β sweep: 3 β × {mnist, fashion} × K3@0.1 × 3 seeds = 18

  local datasets=("mnist" "fashion_mnist")
  if [ "$INCLUDE_CIFAR" -eq 1 ]; then
    datasets+=("cifar10")
  fi
  local scenarios=("K1" "K2" "K3@0.5" "K3@0.1")
  local attacks=("free_rider" "lazy" "label_noise" "sign_flip")

  # ── 1. Clean matrix ──────────────────────────────────────────────────────
  if [ "$INCLUDE_CLEAN" -eq 1 ]; then
    echo ""
    echo "=== Clean matrix ==="
    for seed in $(parse_seeds "$SEEDS"); do
      for ds in "${datasets[@]}"; do
        for sc in "${scenarios[@]}"; do
          local scen da label
          case "$sc" in
            K3@*) scen="K3"; da="${sc#K3@}"; label="K3@$da" ;;
            *)    scen="$sc"; da=""; label="$sc" ;;
          esac
          run_six_cells "$ds" "$scen" "$da" clean "$seed" "clean/$ds/$label[seed=$seed]"
        done
      done
    done
  fi

  # ── 2. Attack matrix (MNIST + Fashion-MNIST: 6 cells) ────────────────────
  if [ "$INCLUDE_ATTACK" -eq 1 ]; then
    echo ""
    echo "=== Attack matrix (MNIST + Fashion-MNIST) ==="
    local attack_scenarios=("K2" "K3@0.1")
    for seed in $(parse_seeds "$ATTACK_SEEDS"); do
      for ds in mnist fashion_mnist; do
        for sc in "${attack_scenarios[@]}"; do
          local scen da
          case "$sc" in
            K3@*) scen="K3"; da="${sc#K3@}" ;;
            *)    scen="$sc"; da="" ;;
          esac
          for atk in "${attacks[@]}"; do
            run_six_cells "$ds" "$scen" "$da" "$atk" "$seed" \
              "attack/$ds/$sc/$atk[seed=$seed]"
          done
        done
      done
    done

    # ── 2b. Attack matrix (CIFAR-10: 4 cells, 1 scenario) ───────────────────
    if [ "$INCLUDE_CIFAR" -eq 1 ]; then
      echo ""
      echo "=== Attack matrix (CIFAR-10, reduced 4 cells) ==="
      for seed in $(parse_seeds "$ATTACK_SEEDS"); do
        for atk in "${attacks[@]}"; do
          run_four_cells_cifar_attack cifar10 K3 "0.1" "$atk" "$seed" \
            "attack/cifar10/K3@0.1/$atk[seed=$seed]"
        done
      done
    fi
  fi

  # ── 3. β sweep (CSRAReward only) ─────────────────────────────────────────
  if [ "$INCLUDE_SWEEP" -eq 1 ]; then
    echo ""
    echo "=== β sweep (FedAvg+CSRAReward, K3@0.1) ==="
    for seed in $(parse_seeds "$SWEEP_SEEDS"); do
      for ds in mnist fashion_mnist; do
        for beta in $(echo "$SWEEP_BETAS" | tr ',' '\n'); do
          # Bỏ qua β=0.5 nếu trùng default (đã chạy ở clean matrix M4)
          run_beta_sweep_one "$ds" K3 "0.1" "$seed" "$beta"
        done
      done
    done
  fi
}

summary() {
  local elapsed=$(( $(date +%s) - START_TIME ))
  echo ""
  echo "════════════════════════════════════════"
  echo "  KẾT QUẢ THỰC NGHIỆM"
  echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
  printf "  Thời gian: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  echo "  Log dir: $LOG_DIR"
  echo "════════════════════════════════════════"
}
trap summary EXIT

echo "════════════════════════════════════════"
echo "  FL Reward Refactor — Experiment Launcher"
echo "  Mode    : $MODE"
echo "  Rounds  : $ROUNDS    Local epochs: $LOCAL_EPOCHS"
echo "  Seeds   : $SEEDS  (attack: $ATTACK_SEEDS)"
echo "  Log dir : $LOG_DIR"
echo "  Cells   : 6 (M1-M6)"
echo "  Subsets : clean=$INCLUDE_CLEAN attack=$INCLUDE_ATTACK sweep=$INCLUDE_SWEEP cifar=$INCLUDE_CIFAR"
echo "  Resume  : $RESUME    Dry-run: $DRY_RUN"
echo "  Chain   : $([ -z "$NO_BLOCKCHAIN_FLAG" ] && echo "on" || echo "off")"
echo "  Parallel: $PARALLEL  (threads/job: $NUM_THREADS)"
echo "  Data imb: $DATA_IMBALANCE"
echo "════════════════════════════════════════"

# Cleanup background jobs on Ctrl+C
trap 'echo ""; echo "Interrupted — killing background jobs..."; kill $(jobs -p) 2>/dev/null; exit 1' INT TERM

check_environment
check_chain

case "$MODE" in
  smoke) run_smoke ;;
  quick) run_quick ;;
  full)  run_full ;;
esac

# Đợi mọi background job kết thúc (parallel mode)
if [ "$PARALLEL" -gt 1 ]; then
  echo ""
  echo "Đang chờ các parallel job hoàn tất..."
  wait
fi
