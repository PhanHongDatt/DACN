# Hướng dẫn cài đặt môi trường máy ảo

## Yêu cầu hệ thống
- Ubuntu 22.04 LTS (khuyến nghị) hoặc 20.04
- RAM: tối thiểu 8GB (16GB để chạy song song)
- CPU: 4 cores
- Disk: 10GB trống

## Bước 1 — Cài đặt Python 3.11
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-pip python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

## Bước 2 — Cài đặt Node.js 20 (cho Hardhat)
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version  # phải là v20.x
```

## Bước 3 — Cài Python dependencies
```bash
pip install -r requirements.txt
```

## Bước 4 — Cài Node dependencies và compile contracts
```bash
npm install
npm run compile
# Kiểm tra: artifacts/ phải có 3 file JSON
ls artifacts/contracts/
```

## Bước 5 — Khởi động Hardhat local node
```bash
# Terminal 1 (để mở)
npx hardhat node
# Xuất ra 20 accounts với 10000 ETH mỗi account
```

## Bước 6 — Deploy contracts (Terminal 2)
```bash
npm run deploy
# Xuất ra địa chỉ 3 contract và lưu vào fl/contract_addresses.json
cat fl/contract_addresses.json

npm run fund
# Cấp 10 ETH cho 10 client accounts
```

## Bước 7 — Chạy unit tests
```bash
pytest tests/ -v
# Tất cả phải PASS trước khi chạy thực nghiệm
```

## Bước 8 — Chạy thực nghiệm đầu tiên (smoke test)
```bash
python experiments/run_experiment.py \
  --dataset mnist --scenario K1 --config C --alpha 0.5 \
  --n-clients 10 --n-rounds 5
# 5 vòng để kiểm tra pipeline hoạt động
```

## Bước 9 — Chạy toàn bộ thực nghiệm
```bash
# Dry-run: in lệnh mà không thực thi
bash experiments/run_all.sh --dry-run

# Chạy thật (ước tính 8-12 giờ tùy cấu hình máy)
bash experiments/run_all.sh 2>&1 | tee results/run_all.log
```

## Bước 10 — Phân tích kết quả
```bash
python experiments/analyze_results.py
# Xuất summary CSV và plots vào results/
```

## Lưu ý quan trọng
- Hardhat node phải đang chạy trước khi chạy bất kỳ experiment nào
- Nếu Hardhat node restart, phải deploy lại contracts (Bước 6)
- Log CSV mỗi run được lưu tự động vào results/logs/
- Dùng `--no-blockchain` để chạy Config A mà không cần node
