#!/bin/bash
# run_all.sh — Chạy thực nghiệm FL-Blockchain theo trình tự logic so sánh
# Trình tự: Baseline (Cơ sở) -> Vulnerability (Tấn công) -> Solution (Cải tiến)

set -uo pipefail

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
  echo "  KẾT QUẢ THỰC NGHIỆM TỔNG THỂ"
  echo "  Trình tự: Baseline -> Vulnerability -> Solution"
  echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
  printf "  Thời gian: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  echo "════════════════════════════════════════"
}
trap summary EXIT

echo "════════════════════════════════════════"
echo "  KHỞI CHẠY HỆ THỐNG THỰC NGHIỆM"
echo "════════════════════════════════════════"

# ──────── PHASE 1: BASELINE REFERENCE (Tiền đề so sánh) ────────
# Mục tiêu: Xác định hiệu năng tối ưu của hệ thống khi không có tấn công.
echo ">>> PHASE 1: Establishing Baselines (Config A & B)"
for ds in mnist fashion_mnist; do
  for sc in K1 K3; do
    # Config A: FL truyền thống (không Blockchain)
    run "$ds/$sc/A-Baseline" --dataset $ds --scenario $sc --config A --no-blockchain --n-rounds $ROUNDS
    # Config B: Blockchain cơ bản (Baseline reward)
    run "$ds/$sc/B-Baseline" --dataset $ds --scenario $sc --config B --alpha 1.0 --n-rounds $ROUNDS
  done
done

# ──────── PHASE 2: SYSTEM VULNERABILITY (Kịch bản tấn công) ────────
# Mục tiêu: Chứng minh Config B (Baseline) bị ảnh hưởng bởi Free-riders.
echo ">>> PHASE 2: Simulating Attacks on Baseline (Config B + Attacks)"
for ds in mnist fashion_mnist; do
  for sc in K1 K3; do
    run "$ds/$sc/B-Attack" --dataset $ds --scenario $sc --config B --alpha 1.0 --with-freeriders --n-rounds $ROUNDS
  done
done

# ──────── PHASE 3: PROPOSED IMPROVEMENT (Giải pháp cải tiến) ────────
# Mục tiêu: Chứng minh Config C vượt trội về tính công bằng và khả năng chống chịu.
echo ">>> PHASE 3: Evaluating Proposed Solution (Config C)"
for ds in mnist fashion_mnist; do
  for sc in K1 K3; do
    # 3.1: Chạy với Alpha tối ưu (0.5) trong điều kiện bình thường
    run "$ds/$sc/C-Normal" --dataset $ds --scenario $sc --config C --alpha 0.5 --n-rounds $ROUNDS
    
    # 3.2: Chạy trong điều kiện có Tấn công (Chứng minh khả năng loại bỏ kẻ xấu)
    run "$ds/$sc/C-Defense" --dataset $ds --scenario $sc --config C --alpha 0.5 --with-freeriders --n-rounds $ROUNDS
    
    # 3.3: Phân tích độ nhạy Alpha (Sensitivity Analysis) - chỉ chạy trên K3 để tiết kiệm thời gian
    if [ "$sc" = "K3" ]; then
      for a in 0.0 0.3 0.7; do
        run "$ds/K3/C-Alpha-$a" --dataset $ds --scenario K3 --config C --alpha $a --n-rounds $ROUNDS
      done
    fi
  done
done

# ──────── BONUS: CIFAR-10 STRESS TEST ────────
echo ">>> BONUS: Stress Test with CIFAR-10 (Non-IID K3)"
run "cifar10/K3/B-Attack" --dataset cifar10 --scenario K3 --config B --alpha 1.0 --with-freeriders --n-rounds 50
run "cifar10/K3/C-Defense" --dataset cifar10 --scenario K3 --config C --alpha 0.5 --with-freeriders --n-rounds 50
