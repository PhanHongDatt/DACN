#!/bin/bash
# run_all.sh — Chạy toàn bộ thực nghiệm FL-Blockchain
#
# Sử dụng:
#   bash experiments/run_all.sh           # chạy thật
#   bash experiments/run_all.sh --dry-run # in lệnh, không thực thi
#   bash experiments/run_all.sh --resume  # bỏ qua run đã có log
#
# Khuyến nghị: chạy trong tmux để tránh mất session khi SSH disconnect
#   tmux new -s exp
#   bash experiments/run_all.sh 2>&1 | tee results/run_all.log

set -uo pipefail   # -e bỏ vì muốn tiếp tục khi 1 run fail

DRY="${1:-}"
LOG_DIR="./results/logs"
ROUNDS=50
PASS=0
FAIL=0
SKIP=0
START_TIME=$(date +%s)

mkdir -p "$LOG_DIR"

run() {
  local desc="$1"; shift
  echo ""
  echo "━━━ $desc ━━━"
  echo "CMD: python experiments/run_experiment.py $*"

  # Resume: bỏ qua nếu đã có log cho run này
  if [ "${DRY}" = "--resume" ]; then
    local ds sc cfg alpha
    for arg in "$@"; do
      case "$prev" in
        --dataset)  ds="$arg" ;;
        --scenario) sc="$arg" ;;
        --config)   cfg="$arg" ;;
        --alpha)    alpha="${arg//.}" ;;
      esac
      prev="$arg"
    done
    prev=""
    local pattern="${LOG_DIR}/${ds:-}_${sc:-}_${cfg:-}_a${alpha:-}*.csv"
    if ls $pattern 1>/dev/null 2>&1; then
      echo "SKIP (log đã tồn tại)"
      SKIP=$((SKIP+1))
      return 0
    fi
  fi

  [ "${DRY}" = "--dry-run" ] && return 0

  if python experiments/run_experiment.py "$@"; then
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
  echo "  Kết quả: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
  printf "  Thời gian: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  echo "  Log: $LOG_DIR"
  echo "════════════════════════════════════════"
}
trap summary EXIT


echo "════════════════════════════════════════"
echo "  FL-Blockchain Experiment Suite"
echo "  Log dir: $LOG_DIR"
echo "  Mode: ${DRY:-run}"
echo "════════════════════════════════════════"

# ══ BLOCK 1: Baseline A — không blockchain ══════════════════
for ds in mnist fashion_mnist; do
  for sc in K1 K2 K3; do
    run "$ds/$sc/A" --dataset $ds --scenario $sc --config A \
      --alpha 0.5 --no-blockchain --n-rounds $ROUNDS --log-dir $LOG_DIR
  done
done

# ══ BLOCK 2: Config B — Yang & Li gốc (alpha=1.0) ═══════════
for ds in mnist fashion_mnist; do
  for sc in K1 K2 K3; do
    run "$ds/$sc/B" --dataset $ds --scenario $sc --config B \
      --alpha 1.0 --n-rounds $ROUNDS --log-dir $LOG_DIR
  done
done

# ══ BLOCK 3: Config C — Sensitivity Analysis alpha ══════════
for ds in mnist fashion_mnist; do
  for sc in K1 K2 K3; do
    for alpha in 0.0 0.3 0.5 0.7 1.0; do
      run "$ds/$sc/C/a$alpha" --dataset $ds --scenario $sc --config C \
        --alpha $alpha --n-rounds $ROUNDS --log-dir $LOG_DIR
    done
  done
done

# ══ BLOCK 4: Nhóm 3 — Free-rider simulation ═════════════════
for ds in mnist fashion_mnist; do
  for sc in K1 K2 K3; do
    run "$ds/$sc/B+FR" --dataset $ds --scenario $sc --config B \
      --alpha 1.0 --with-freeriders --n-rounds $ROUNDS --log-dir $LOG_DIR
    run "$ds/$sc/C+FR" --dataset $ds --scenario $sc --config C \
      --alpha 0.5 --with-freeriders --n-rounds $ROUNDS --log-dir $LOG_DIR
  done
done

# ══ BLOCK 5: CIFAR-10 stress test (100 rounds CNN) ══════════
for alpha in 0.0 0.3 0.5 0.7 1.0; do
  run "cifar10/K3/C/a$alpha" --dataset cifar10 --scenario K3 --config C \
    --alpha $alpha --n-rounds 100 --log-dir $LOG_DIR
done
run "cifar10/K3/B" --dataset cifar10 --scenario K3 --config B \
  --alpha 1.0 --n-rounds 100 --log-dir $LOG_DIR
run "cifar10/K3/C+FR" --dataset cifar10 --scenario K3 --config C \
  --alpha 0.5 --with-freeriders --n-rounds 100 --log-dir $LOG_DIR
run "cifar10/K3/B+FR" --dataset cifar10 --scenario K3 --config B \
  --alpha 1.0 --with-freeriders --n-rounds 100 --log-dir $LOG_DIR

