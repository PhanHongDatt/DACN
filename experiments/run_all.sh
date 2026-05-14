#!/bin/bash
# run_all.sh — VM-friendly experiment matrix for FL/Blockchain/CSRA comparison.
# Default scope: 70 runs at 50 rounds.

set -uo pipefail

MODE="${1:-}"
LOG_DIR="${LOG_DIR:-./results/logs}"
ROUNDS="${ROUNDS:-50}"
SEED="${SEED:-42}"
N_CLIENTS="${N_CLIENTS:-10}"
TRIM_RATIO="${TRIM_RATIO:-0.1}"

MAIN_DATASETS=("mnist" "fashion_mnist")
DIRICHLET_ALPHAS=("0.5" "0.1")

PASS=0
FAIL=0
SKIP=0
START_TIME=$(date +%s)

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

run() {
  local desc="$1"; shift
  local runner="${RUNNER:-experiments/run_experiment.py}"
  echo ""
  echo "━━━ $desc ━━━"
  echo "CMD: python $runner $*"

  if [ "$MODE" = "--resume" ]; then
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
      alpha="00"
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

  [ "$MODE" = "--dry-run" ] && return 0

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
    --dataset "$ds" "$@" --config A --no-blockchain $(common_args)
  run "$ds/$label/BlockchainQuality-Clean" \
    --dataset "$ds" "$@" --config B --alpha 1.0 $(common_args)
  RUNNER="experiments/run_experiment_trimmed.py" run "$ds/$label/TrimmedMean-Clean" \
    --dataset "$ds" "$@" --config TrimmedMean --trim-ratio "$TRIM_RATIO" --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Clean" \
    --dataset "$ds" "$@" --config C --alpha 0.5 $(common_args)

  run "$ds/$label/FedAvg-Attack" \
    --dataset "$ds" "$@" --config A --no-blockchain --with-freeriders $(common_args)
  run "$ds/$label/BlockchainQuality-Attack" \
    --dataset "$ds" "$@" --config B --alpha 1.0 --with-freeriders $(common_args)
  RUNNER="experiments/run_experiment_trimmed.py" run "$ds/$label/TrimmedMean-Attack" \
    --dataset "$ds" "$@" --config TrimmedMean --trim-ratio "$TRIM_RATIO" --with-freeriders --no-blockchain $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Defense" \
    --dataset "$ds" "$@" --config C --alpha 0.5 --with-freeriders $(common_args)
}

run_cifar_stress() {
  local ds="cifar10"
  local label="K3-Dirichlet-a0.1-Stress"
  local args=(--scenario K3 --dirichlet-alpha 0.1)

  run "$ds/$label/FedAvg-Clean" \
    --dataset "$ds" "${args[@]}" --config A --no-blockchain $(common_args)
  run "$ds/$label/BlockchainQuality-Clean" \
    --dataset "$ds" "${args[@]}" --config B --alpha 1.0 $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Clean" \
    --dataset "$ds" "${args[@]}" --config C --alpha 0.5 $(common_args)

  run "$ds/$label/FedAvg-Attack" \
    --dataset "$ds" "${args[@]}" --config A --no-blockchain --with-freeriders $(common_args)
  run "$ds/$label/BlockchainQuality-Attack" \
    --dataset "$ds" "${args[@]}" --config B --alpha 1.0 --with-freeriders $(common_args)
  RUNNER="experiments/run_experiment_csra.py" run "$ds/$label/CSRA-Defense" \
    --dataset "$ds" "${args[@]}" --config C --alpha 0.5 --with-freeriders $(common_args)
}

echo "════════════════════════════════════════"
echo "  KHỞI CHẠY MA TRẬN THỰC NGHIỆM VM"
echo "  Main datasets: ${MAIN_DATASETS[*]}"
echo "  Main scenarios: K1, K2, K3(alpha=${DIRICHLET_ALPHAS[*]})"
echo "  Stress dataset: cifar10/K3(alpha=0.1)"
echo "  Methods: FedAvg, BlockchainQuality, TrimmedMean, CSRA"
echo "  Rounds mỗi run: $ROUNDS"
echo "  Seed: $SEED"
echo "════════════════════════════════════════"

for ds in "${MAIN_DATASETS[@]}"; do
  run_method_pair "$ds" "K1-IID" --scenario K1
  run_method_pair "$ds" "K2-WeakNonIID" --scenario K2
  for da in "${DIRICHLET_ALPHAS[@]}"; do
    run_method_pair "$ds" "K3-Dirichlet-a$da" --scenario K3 --dirichlet-alpha "$da"
  done
done

run_cifar_stress
