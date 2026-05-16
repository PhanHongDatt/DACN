#!/bin/bash
# run_pipeline.sh — One-shot script: auto-deploy blockchain + run experiments.
#
# Usage:
#   bash scripts/run_pipeline.sh --quick                       # Test nhanh, no blockchain
#   bash scripts/run_pipeline.sh --full --no-cifar --resume    # Full matrix
#   bash scripts/run_pipeline.sh --quick --with-blockchain     # Demo blockchain audit
#
# Env vars (forward sang run_all.sh):
#   PARALLEL=4 NUM_THREADS=2  → recommend cho VM 8 core
#   ROUNDS=25 LOCAL_EPOCHS=1
#
# Tự động:
#   1. Khởi động Hardhat node (nếu cần)
#   2. Compile contracts (nếu chưa)
#   3. Deploy + fund + healthcheck
#   4. Chạy run_all.sh với args bạn truyền
#   5. Cleanup Hardhat node khi xong / Ctrl+C

set -uo pipefail

cd "$(dirname "$0")/.."

HARDHAT_PID=""
HARDHAT_LOG="/tmp/hardhat-pipeline.log"
WAS_RUNNING=0

cleanup() {
  if [ -n "$HARDHAT_PID" ] && [ "$WAS_RUNNING" -eq 0 ]; then
    if kill -0 "$HARDHAT_PID" 2>/dev/null; then
      echo ""
      echo ">>> Stopping Hardhat node (PID $HARDHAT_PID)..."
      kill "$HARDHAT_PID" 2>/dev/null
      wait "$HARDHAT_PID" 2>/dev/null
      echo "Hardhat stopped."
    fi
  elif [ "$WAS_RUNNING" -eq 1 ]; then
    echo ""
    echo ">>> Hardhat node đã chạy trước script này — không tắt."
  fi
}
trap cleanup EXIT INT TERM

# ── Detect blockchain mode ───────────────────────────────────────────────────
NEED_CHAIN=0
PARALLEL_VAL="${PARALLEL:-1}"
for arg in "$@"; do
  if [ "$arg" = "--with-blockchain" ]; then
    NEED_CHAIN=1
  fi
done

# ── Sanity check parallel + blockchain ───────────────────────────────────────
if [ "$NEED_CHAIN" -eq 1 ] && [ "$PARALLEL_VAL" -gt 1 ]; then
  echo "⚠️  Cảnh báo: PARALLEL=$PARALLEL_VAL với --with-blockchain có thể"
  echo "   gây nonce conflict (tất cả tx dùng chung owner account)."
  echo "   Khuyến nghị: PARALLEL=1 khi bật blockchain, hoặc dùng --no-blockchain"
  echo "   cho full matrix (CSV vẫn ghi đầy đủ data)."
  echo ""
  read -p "Tiếp tục? [y/N] " yn
  case "$yn" in
    [Yy]*) echo "Tiếp tục..." ;;
    *) exit 1 ;;
  esac
fi

# ── Hardhat node setup (chỉ khi --with-blockchain) ───────────────────────────
chain_is_up() {
  python - <<'PY' 2>/dev/null
import sys, urllib.request
try:
    req = urllib.request.Request(
        "http://127.0.0.1:8545",
        data=b'{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

if [ "$NEED_CHAIN" -eq 1 ]; then
  echo "════════════════════════════════════════"
  echo "  Phase 1: Hardhat blockchain setup"
  echo "════════════════════════════════════════"

  # 1.1. Check node already running
  if chain_is_up; then
    echo ">>> Hardhat node đã chạy ở port 8545. Sẽ tái sử dụng."
    WAS_RUNNING=1
  else
    echo ">>> Khởi động Hardhat node ở background..."
    npx hardhat node > "$HARDHAT_LOG" 2>&1 &
    HARDHAT_PID=$!
    echo "    PID: $HARDHAT_PID  log: $HARDHAT_LOG"

    # Đợi node ready (max 30s)
    for i in {1..30}; do
      sleep 1
      if chain_is_up; then
        echo ">>> Hardhat node ready (sau ${i}s)."
        break
      fi
      if [ "$i" -eq 30 ]; then
        echo "❌ Hardhat không khởi động được. Xem $HARDHAT_LOG" >&2
        exit 1
      fi
    done
  fi

  # 1.2. Compile contracts (nếu chưa)
  if [ ! -d "artifacts/contracts" ]; then
    echo ""
    echo ">>> Compile contracts..."
    npx hardhat compile
  else
    echo ">>> Contracts đã compile (artifacts/ tồn tại)."
  fi

  # 1.3. Deploy
  echo ""
  echo ">>> Deploy contracts..."
  npx hardhat run scripts/deploy.js --network localhost

  # 1.4. Fund test accounts
  echo ""
  echo ">>> Fund test accounts..."
  npx hardhat run scripts/fund_accounts.js --network localhost

  # 1.5. Healthcheck
  echo ""
  echo ">>> Healthcheck..."
  npx hardhat run scripts/healthcheck.js --network localhost

  echo ""
  echo "✅ Blockchain setup hoàn tất."
else
  echo ">>> Bỏ qua blockchain setup (no --with-blockchain flag)."
fi

# ── Phase 2: Run experiments ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Phase 2: Run experiments"
echo "  Forward args: $*"
echo "════════════════════════════════════════"

bash experiments/run_all.sh "$@"

echo ""
echo "✅ Pipeline hoàn tất. Logs trong: ${LOG_DIR:-./results/logs}"
