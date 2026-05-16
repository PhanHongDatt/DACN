# FL-Blockchain Reward Distribution

Dự án: **Cải tiến cơ chế phân phối phần thưởng dựa trên đóng góp đa chiều trong hệ thống Federated Learning kết hợp Blockchain**.

Hệ thống mô phỏng Federated Learning bằng Flower, ghi nhận đóng góp và phân phối reward qua smart contract Solidity chạy trên Hardhat.

> **Schema v2 — refactor reward policies** _(đang triển khai trên branch `refactor/reward-policies`)_
>
> Pipeline được tái thiết kế thành 2 chiều độc lập:
> - **Aggregation method**: `fedavg | trimmed | csra_dcd`
> - **Reward policy**: `equal | data | quality | csra` (CSRA = 3-chiều `β·quality + γ·data + δ·reputation`)
>
> Blockchain đóng vai trò audit/log/distribute layer, không phải hyperparameter so sánh.
>
> Chi tiết: [`docs/PLAN.md`](docs/PLAN.md).

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
    CLI["Launcher<br/>experiments/run_all.sh"] --> Runner["Unified Runner<br/>experiments/run_experiment.py"]

    Runner --> Data["Data Pipeline<br/>fl/data_utils.py"]
    Runner --> Sim["Local Sequential Sim<br/>fl/simulation_local.py"]

    Data --> Clients["Client Hierarchy<br/>fl/client_attacks.py<br/>(Honest, FreeRider, Lazy,<br/>LabelNoise, SignFlip)"]
    Clients --> Sim

    Sim --> Strategy["FLUnifiedStrategy<br/>fl/server_base.py"]

    Strategy --> Agg["Aggregation Method<br/>fl/aggregation_methods.py<br/>(fedavg | trimmed | csra_dcd)"]
    Strategy --> Reward["Reward Policy<br/>fl/reward_policies.py<br/>(equal | data | quality | csra)"]

    Agg --> Filter["Anomaly Mask<br/>(post-filter valid clients)"]
    Filter --> Reward

    Reward --> Bridge["BlockchainBridge (optional)<br/>fl/blockchain.py<br/>audit-only mode"]

    Bridge --> Store["ContributionStore.sol"]
    Bridge --> RewardC["RewardDistributor.sol"]
    Bridge --> Registry["FLRegistry.sol"]

    Strategy --> Logs["CSV Logs<br/>results/logs/<br/>schema v2"]

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

### 5.2. Chạy Một Experiment Đơn Lẻ (Schema v2)

Mọi experiment dùng chung **một runner duy nhất** với cờ `--aggregation` và `--reward-policy`.

M1 — FedAvg + EqualSplit (baseline trần):

```bash
python -m experiments.run_experiment \
  --dataset mnist --scenario K1 \
  --aggregation fedavg --reward-policy equal \
  --seed 42 --n-rounds 10 --no-blockchain
```

M4 — FedAvg + CSRAReward 3-chiều (ablation: chỉ reward formula):

```bash
python -m experiments.run_experiment \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --aggregation fedavg --reward-policy csra \
  --beta 0.5 --gamma 0.3 --delta 0.2 \
  --seed 42 --n-rounds 10 --no-blockchain
```

M6 — CSRA-DCD + CSRAReward (hệ thống đầy đủ với attack):

```bash
python -m experiments.run_experiment \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --aggregation csra_dcd --reward-policy csra \
  --beta 0.5 --gamma 0.3 --delta 0.2 \
  --attack free_rider --attack-client-ids 8,9 \
  --seed 42 --n-rounds 10 --no-blockchain
```

TrimmedMean baseline:

```bash
python -m experiments.run_experiment \
  --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
  --aggregation trimmed --reward-policy equal \
  --trim-ratio 0.1 --seed 42 --n-rounds 10 --no-blockchain
```

Toàn bộ flags: `python -m experiments.run_experiment --help`.

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

### Methods (Schema v2 — Ablation 2 chiều)

Mọi cấu hình đều dùng `run_experiment.py` duy nhất với `--aggregation` × `--reward-policy`:

| ID | Aggregation | Reward Policy | Vai trò |
| --- | --- | --- | --- |
| **M1** | `fedavg` | `equal` | Baseline trần |
| **M2** | `fedavg` | `data` | Bias data quantity |
| **M3** | `fedavg` | `quality` | Bias quality (nhạy noise) |
| **M4** | `fedavg` | `csra` | Ablation: chỉ reward formula |
| **M5** | `csra_dcd` | `equal` | Ablation: chỉ filtering |
| **M6** | `csra_dcd` | `csra` | **Hệ thống đề xuất** |

Attack types: `clean | free_rider | lazy | label_noise | sign_flip` (via `--attack`).

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

**Sweep β cho CSRAReward:**

```bash
for beta in 0.3 0.5 0.7; do
  gamma=$(python -c "print(round((1 - $beta) * 3/5, 4))")
  delta=$(python -c "print(round((1 - $beta) * 2/5, 4))")
  python -m experiments.run_experiment \
    --dataset mnist --scenario K3 --dirichlet-alpha 0.1 \
    --aggregation fedavg --reward-policy csra \
    --beta $beta --gamma $gamma --delta $delta \
    --seed 42 --n-rounds 10 --no-blockchain
done
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
fl/             Core modules:
                  - reward_policies.py    (4 reward policies)
                  - aggregation_methods.py (3 aggregation strategies)
                  - client_attacks.py      (Honest + 4 attack subclasses)
                  - server_base.py         (Unified strategy)
                  - simulation_local.py    (Sequential sim, no ray)
                  - blockchain.py          (Audit-only bridge)
                  - logger.py              (CSV schema v2)
                  - data_utils.py, models.py, metrics.py, normalization.py, config.py
experiments/    run_experiment.py (unified) + run_all.sh (launcher)
analysis/       Loader, statistics, plots, report generator
scripts/        Deploy, fund, healthcheck, environment check
tests/          Unit tests, contract tests, integration tests
results/        Logs, summaries, plots
docs/           PLAN.md (refactor plan), SETUP.md (env)
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
