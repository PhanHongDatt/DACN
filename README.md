# FL-Blockchain Reward Distribution

Dự án: **Cải tiến cơ chế phân phối phần thưởng dựa trên đóng góp đa chiều trong hệ thống Federated Learning kết hợp Blockchain**.

Hệ thống mô phỏng Federated Learning bằng Flower, ghi nhận đóng góp và phân phối reward qua smart contract Solidity chạy trên Hardhat. Phiên bản hiện tại tập trung so sánh thuật toán đề xuất **CSRA + Blockchain minh bạch** với các baseline phù hợp: **FedAvg**, **BlockchainQuality** và **TrimmedMean**.

---

## 1. Tổng Quan Hệ Thống

Mục tiêu chính của hệ thống:

- Huấn luyện mô hình FL trên `MNIST`, `Fashion-MNIST`, `CIFAR-10`.
- Hỗ trợ kịch bản dữ liệu `IID`, `Weak Non-IID`, `Dirichlet Non-IID`.
- Mô phỏng client bất thường: `free_rider`, `lazy`, label-noise.
- Ghi log contribution/reward/reputation/accuracy theo từng round.
- So sánh CSRA-DCD với các baseline về accuracy, fairness, reward leakage và detection.

### Kiến Trúc

```mermaid
flowchart TB
    CLI["Experiment Launcher<br/>experiments/run_all.sh"] --> Runners["Experiment Runners<br/>run_experiment.py<br/>run_experiment_csra.py<br/>run_experiment_trimmed.py"]

    Runners --> Data["Data Pipeline<br/>fl/data_utils.py<br/>K1 IID, K2 Weak Non-IID, K3 Dirichlet"]
    Runners --> Sim["Flower Simulation"]

    Data --> Clients["FL Clients<br/>FLClient / FLClientCSRA"]
    Clients --> Sim

    Sim --> FedAvg["FedAvg / BlockchainQuality<br/>fl/server.py"]
    Sim --> CSRA["CSRA Strategy<br/>fl/server_csra.py"]
    Sim --> Trimmed["TrimmedMean Baseline<br/>fl/server_trimmed.py"]

    CSRA --> DCD["CSRA-DCD<br/>Update Delta L2 Norm + MAD"]
    DCD --> Filter["Filter abnormal clients<br/>Exclude aggregation + reward"]
    Filter --> RWA["Reputation-weighted aggregation"]

    FedAvg --> Bridge["BlockchainBridge<br/>fl/blockchain.py"]
    CSRA --> Bridge

    Bridge --> Store["ContributionStore.sol<br/>Contribution + Reputation"]
    Bridge --> Reward["RewardDistributor.sol<br/>ETH Reward Distribution"]
    Bridge --> Registry["FLRegistry.sol<br/>Experiment Metadata"]

    FedAvg --> Logs["CSV Logs<br/>results/logs"]
    CSRA --> Logs
    Trimmed --> Logs

    Logs --> Analysis["Analysis Pipeline<br/>analyze_results.py<br/>analysis/loader.py, stats.py, plots.py"]
    Analysis --> Outputs["Reports + Plots<br/>results/summary_metrics.csv<br/>results/fairness_metrics.csv<br/>results/analysis_report.md<br/>results/plots"]
```

### Luồng Một Round FL

1. Server gửi global parameters cho các clients.
2. Client huấn luyện local model và trả về parameters + metadata.
3. Với CSRA, client gửi thêm `anomaly_score = ||local_params - global_params||_2`.
4. Server CSRA dùng MAD robust z-score để phát hiện update bất thường.
5. Client bị DCD flag sẽ bị loại khỏi aggregation và không nhận reward.
6. Blockchain ghi contribution, reputation và phân phối reward cho nhóm hợp lệ.
7. Logger ghi CSV theo round để phân tích offline.

---

## 2. Yêu Cầu Môi Trường

Khuyến nghị:

- Python `3.11`
- Node.js `20.x`
- RAM tối thiểu `8GB`, khuyến nghị `16GB`
- Git Bash/WSL/Linux shell để chạy `experiments/run_all.sh`

Kiểm tra nhanh môi trường:

```bash
python scripts/check_python.py
npm run compile
npm test
```

---

## 3. Cài Đặt

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

npm install
npm run compile
```

Trên Windows, nên dùng Python 3.11 virtualenv và chạy script Bash bằng Git Bash hoặc WSL.

---

## 4. Khởi Động Blockchain Local

Mở terminal 1:

```bash
npx hardhat node
```

Mở terminal 2:

```bash
npm run deploy
npm run fund
npm run healthcheck
```

Nếu chỉ chạy smoke test không blockchain, có thể bỏ qua bước này.

---

## 5. Chạy Thực Nghiệm

### 5.1. Chạy Nhanh Bằng Launcher

Xem hướng dẫn:

```bash
bash experiments/run_all.sh --help
```

Các mode chính:

```bash
# In lệnh, không chạy
bash experiments/run_all.sh --full --dry-run

# Smoke test: 3 runs, không cần blockchain, mặc định 3 rounds
bash experiments/run_all.sh --smoke

# Quick test: 16 runs MNIST, mặc định 10 rounds
bash experiments/run_all.sh --quick --resume

# Full matrix: 70 runs, mặc định 50 rounds/run
bash experiments/run_all.sh --full --resume

# Full nhưng bỏ CIFAR-10 stress để giảm tải VM: 64 runs
bash experiments/run_all.sh --full --no-cifar --resume

# Chỉ chạy CIFAR-10 stress: 6 runs
bash experiments/run_all.sh --cifar-only --resume
```

Có thể override bằng biến môi trường:

```bash
ROUNDS=20 SEED=123 LOG_DIR=./results/logs bash experiments/run_all.sh --quick --resume
```

### 5.2. Chạy Một Experiment Đơn Lẻ

FedAvg không blockchain:

```bash
python experiments/run_experiment.py \
  --dataset mnist --scenario K1 --config A --alpha 0.0 --no-blockchain \
  --n-rounds 10
```

BlockchainQuality:

```bash
python experiments/run_experiment.py \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --config B --alpha 1.0 --n-rounds 10
```

CSRA defense:

```bash
python experiments/run_experiment_csra.py \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --config C --alpha 0.5 --with-freeriders --n-rounds 10
```

TrimmedMean baseline:

```bash
python experiments/run_experiment_trimmed.py \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --config TrimmedMean --trim-ratio 0.1 --no-blockchain --n-rounds 10
```

---

## 6. Kịch Bản Và Baseline

### Datasets

- `mnist`
- `fashion_mnist`
- `cifar10`

### Data Scenarios

- `K1`: IID.
- `K2`: Weak Non-IID.
- `K3`: Dirichlet Non-IID, dùng `--dirichlet-alpha`, khuyến nghị `0.5` và `0.1`.

### Methods

| Method | Runner | Blockchain | Mục đích |
| --- | --- | --- | --- |
| `A` FedAvg | `run_experiment.py` | Không | Baseline FL truyền thống |
| `B` BlockchainQuality | `run_experiment.py` | Có | FedAvg + reward theo quality/data/reputation |
| `TrimmedMean` | `run_experiment_trimmed.py` | Không | Robust aggregation baseline |
| `C-CSRA-Opt` | `run_experiment_csra.py` | Có | Thuật toán đề xuất CSRA-DCD + blockchain minh bạch |

---

## 7. CSRA Trong Hệ Thống

### CSRA-DCD

CSRA-DCD hiện dùng update delta norm:

```text
delta_i = local_params_i - global_params
anomaly_score_i = ||delta_i||_2
```

Server dùng MAD robust z-score để phát hiện anomaly:

```text
z_i = |score_i - median(score)| / (1.4826 * MAD)
```

Nếu `z_i > --mad-threshold`, client bị đánh dấu anomaly. Client bị anomaly:

- Không tham gia aggregation.
- Contribution được zero hóa trước khi ghi reward flow.
- Bị exclude khỏi `filter_and_distribute` nên không nhận reward round đó.
- Được log vào CSV qua `is_anomaly`, `anomaly_score`, `robust_z`, `detection_reason`.

### Alpha Reward

`alpha` điều chỉnh trọng số giữa quality và data-size trong reward:

```text
W_new = alpha * quality_norm + (1 - alpha) * data_size_norm
```

Mặc định CSRA dùng alpha tĩnh. Có thể bật alpha động bằng:

```bash
python experiments/run_experiment_csra.py ... --dynamic-alpha
```

---

## 8. Phân Tích Kết Quả

Log CSV được ghi tại:

```text
results/logs/
```

Chạy phân tích:

```bash
python analyze_results.py --report
```

Output chính:

- `results/summary_metrics.csv`
- `results/fairness_metrics.csv`
- `results/analysis_report.md`
- `results/plots/*.png`

Metrics đáng chú ý:

- Final accuracy, peak accuracy.
- Convergence round: round đầu tiên đạt `95%` peak accuracy và ổn định trong 5 rounds.
- Jain index, Gini, fairness gap.
- Reward-quality correlation.
- False positive rate.
- Attack detection rate.
- Reward leakage.

---

## 9. Cấu Trúc Thư Mục

```text
contracts/      Smart contracts Solidity
fl/             Core FL, clients, strategies, blockchain bridge, metrics
experiments/    Experiment runners và run_all launcher
analysis/       Loader, statistics, plots, report generator
scripts/        Deploy, fund, healthcheck, environment check
tests/          Unit tests, contract tests, integration tests
results/        Logs, summaries, plots
docs/           Setup notes
```

---

## 10. Kiểm Thử

```bash
python -m pytest tests/unit -q
npm test
python -m compileall fl experiments analysis tests -q
```

Kiểm tra dry-run matrix:

```bash
bash experiments/run_all.sh --smoke --dry-run
bash experiments/run_all.sh --quick --dry-run
bash experiments/run_all.sh --full --dry-run
```

---

## 11. Giới Hạn Hiện Tại

- `quality_score` hiện vẫn do client báo cáo dựa trên delta loss local; server-side validation quality nên được xem là hướng cải tiến tiếp theo.
- Full matrix 70 runs có thể tốn nhiều giờ trên VM, nên chạy `--smoke` và `--quick` trước.
- Blockchain local cần Hardhat node và deployed contracts khi chạy các config có reward on-chain.

---

**Tác giả:** Phan Hồng Đạt - 23520266
