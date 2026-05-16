# Kế Hoạch Cải Tiến Hệ Thống — FL + Blockchain Reward Distribution

**Đề tài:** Cải tiến cơ chế phân phối phần thưởng dựa trên đóng góp đa chiều trong hệ thống Học máy Liên kết kết hợp Blockchain

**Tác giả:** Phan Hồng Đạt — 23520266
**Ngày lập kế hoạch:** 2026-05-16
**Phạm vi:** Refactor toàn diện + chạy lại từ đầu để có kết quả tốt nhất

---

## 1. Bối Cảnh & Vấn Đề Của Thiết Kế Cũ

### 1.1. Đánh giá thiết kế hiện tại

Phiên bản hiện tại có 4 method:

| Config | Aggregation | Reward | Blockchain |
|---|---|---|---|
| A | FedAvg | None | No |
| B | FedAvg | α·Q + (1−α)·D | Yes |
| TrimmedMean | TrimmedMean | None | No |
| C-CSRA | CSRA-DCD filtering | α·Q + (1−α)·D | Yes |

### 1.2. Vấn đề lý thuyết

**Confound 1:** Config B bao gồm 3 yếu tố cùng lúc (blockchain on-chain, reputation gating, multi-factor reward). Không tách được phần improvement nào do thứ nào.

**Confound 2:** So sánh C-CSRA vs B không cho biết lợi ích đến từ filtering (CSRA-DCD) hay từ reward formula (CSRA).

**Confound 3:** Blockchain bị nhầm với thuật toán. Câu hỏi "blockchain có hữu ích không" không phải mục tiêu nghiên cứu của đề tài này.

### 1.3. Hệ quả

Báo cáo cũ không trả lời rõ được: **"Cải tiến cơ chế reward đến đâu thì là do CSRA?"**

---

## 2. Định Hướng Mới

### 2.1. Câu hỏi nghiên cứu chính

> **Cơ chế phân phối phần thưởng đề xuất dựa trên đóng góp đa chiều (CSRA) có công bằng, ổn định và chống reward leakage tốt hơn các baseline phân phối đơn giản (EqualSplit, DataSize, QualityOnly) hay không, khi đặt trong cùng một aggregation method?**

### 2.2. Vai trò của Blockchain

Blockchain được tái định vị là **lớp hạ tầng minh bạch hóa**, không phải hyperparameter:

- Lưu contribution, reputation, reward trên-chain để audit.
- Bật cho **tất cả** configs có reward (không bật/tắt như tham số).
- Không so sánh "có blockchain vs không blockchain" — không có ý nghĩa khoa học trong phạm vi đề tài.

### 2.3. Tại sao hướng này khớp tên đề tài

Tên đề tài: *"Cải tiến cơ chế phân phối phần thưởng dựa trên đóng góp đa chiều trong hệ thống Học máy Liên kết kết hợp Blockchain"*

| Cụm từ | Vai trò trong thiết kế mới |
|---|---|
| Cải tiến cơ chế phân phối phần thưởng | **Trọng tâm** — so sánh 4 reward policies |
| Đóng góp đa chiều | **CSRAReward 3-chiều** (quality + data + reputation) |
| Học máy Liên kết | Flower simulation, datasets, scenarios |
| Kết hợp Blockchain | **Hạ tầng** — audit layer, không phải đối tượng so sánh |

---

## 3. Thiết Kế Ablation Hai Chiều

### 3.1. Tách độc lập aggregation × reward

**Aggregation method** quyết định cách hợp nhất parameters:

| Method | Mô tả |
|---|---|
| `FedAvg` | Trung bình có trọng số theo data size (chuẩn) |
| `TrimmedMean` | Robust aggregation (trim ratio = 0.1) |
| `CSRA-DCD` | FedAvg + filter client bất thường bằng MAD robust z-score trên `‖Δᵢ‖₂` |

**Reward policy** quyết định cách chia reward cho clients hợp lệ:

| Policy | Công thức | Mục đích |
|---|---|---|
| `EqualSplit` | `r_i = R/n` | Baseline tối thiểu (mọi người bằng nhau) |
| `DataSize` | `r_i = R · d_i / Σd_j` | Baseline theo lượng dữ liệu |
| `QualityOnly` | `r_i = R · q_i / Σq_j` | Baseline theo chất lượng |
| `CSRAReward` | `r_i = R · W_i / ΣW_j` (xem §4) | **Đề xuất chính** |

### 3.2. Semantic: reward chỉ trên valid clients (post-filter)

Sau khi aggregation method (đặc biệt CSRA-DCD) loại các client bị flag, **reward chỉ phân phối cho phần còn lại**:

```
valid_clients = participants − flagged_by_filter
r_i = compute_reward(valid_clients, reward_policy)  ∀ i ∈ valid_clients
r_i = 0  ∀ i ∈ flagged_by_filter
```

Hệ quả ablation:
- `FedAvg + X` vs `CSRA-DCD + X` (cùng reward) → **tách effect của filtering**
- `Y + EqualSplit` vs `Y + CSRAReward` (cùng aggregation) → **tách effect của reward formula**

---

## 4. Công Thức CSRAReward 3-Chiều

### 4.1. Định nghĩa

Cho client `i` ở round `t`, định nghĩa weight đóng góp đa chiều:

$$W_i = \beta \cdot \tilde{q}_i + \gamma \cdot \tilde{d}_i + \delta \cdot \tilde{r}_i$$

Với:
- $\tilde{q}_i = q_i / \sum_j q_j$ — quality score chuẩn hoá ($q_i$ là Δloss local)
- $\tilde{d}_i = d_i / \sum_j d_j$ — data size chuẩn hoá
- $\tilde{r}_i = \rho_i / \sum_j \rho_j$ — reputation chuẩn hoá ($\rho_i$ tích luỹ qua rounds)
- $\beta + \gamma + \delta = 1$, mỗi tham số $\in [0, 1]$

Reward thực tế phân phối:

$$r_i = R \cdot \frac{W_i}{\sum_{j \in \text{valid}} W_j}$$

### 4.2. Tham số mặc định và sweep

| Tham số | Default | Sweep range | Giữ ratio γ:δ |
|---|---|---|---|
| β (quality) | 0.5 | {0.3, 0.5, 0.7} | — |
| γ (data) | 0.3 | tự tính | 3:2 với δ |
| δ (reputation) | 0.2 | tự tính | 2:3 với γ |

Sweep cụ thể:
| β | γ | δ |
|---|---|---|
| 0.3 | 0.42 | 0.28 |
| 0.5 | 0.30 | 0.20 |
| 0.7 | 0.18 | 0.12 |

### 4.3. Defend trong báo cáo

**Câu hỏi 1: Tại sao thêm reputation vào reward formula?**

- Quality score (Δloss) là tín hiệu **1 round**, dễ nhiễu (ví dụ: 1 batch không may, learning rate noise).
- Reputation tích luỹ qua nhiều rounds → đóng vai trò **smoothing**, giảm phương sai reward.
- Khi quality nhiễu cao, reputation giúp reward ổn định hơn.

**Câu hỏi 2: Tại sao β > γ > δ?**

Ordering theo độ tin cậy của tín hiệu:
- β (quality) cao nhất: tín hiệu **trực tiếp** của vòng hiện tại
- γ (data size) trung bình: proxy **cố định**, không phản ánh chất lượng
- δ (reputation) thấp nhất: tín hiệu **lịch sử**, có thể lỗi thời

**Câu hỏi 3: Tại sao tổng = 1?**

Để $\sum_i W_i$ chuẩn hoá ổn định, tránh reward bị scale ngẫu nhiên theo magnitude của các tín hiệu thành phần.

---

## 5. Ma Trận Thực Nghiệm

### 5.1. Ma trận chính: 6 cells

| ID | Aggregation | Reward | Vai trò |
|---|---|---|---|
| **M1** | FedAvg | EqualSplit | Baseline trần (không filtering, không reward thông minh) |
| **M2** | FedAvg | DataSize | Bias theo lượng dữ liệu |
| **M3** | FedAvg | QualityOnly | Bias theo chất lượng (nhạy noise) |
| **M4** | FedAvg | **CSRAReward** | Ablation: tách riêng reward formula |
| **M5** | CSRA-DCD | EqualSplit | Ablation: tách riêng filtering |
| **M6** | **CSRA-DCD** | **CSRAReward** | **Hệ thống đề xuất — claim chính** |

### 5.2. So sánh quan trọng trong báo cáo

| So sánh | Đo lường |
|---|---|
| M4 vs {M1, M2, M3} | CSRAReward công bằng hơn baselines đơn giản (chưa cần filter) |
| M5 vs M1 | Filtering giảm reward leakage (chưa cần reward formula) |
| M6 vs M4 | Lợi ích bổ sung của filtering khi đã có CSRAReward |
| M6 vs M5 | Lợi ích bổ sung của reward formula khi đã có filter |
| M6 vs M1 | **Tổng lợi ích của hệ thống đầy đủ** |

### 5.3. Coverage thực nghiệm

| Trục | Giá trị |
|---|---|
| Datasets | MNIST, Fashion-MNIST, CIFAR-10 |
| Scenarios clean | K1 (IID), K2 (Weak Non-IID), K3@α=0.1, K3@α=0.5 |
| Scenarios attack | K2, K3@α=0.1 (2 điều kiện khó nhất) |
| Attack types | free-rider, lazy, label-noise, sign-flip |
| Seeds (clean) | 3 (cố định: 42, 123, 2024) |
| Seeds (attack) | 2 (cố định: 42, 123) |

### 5.4. Tổng số runs

| Phần | Công thức tính | Runs |
|---|---|---|
| Clean matrix | 6 cells × 3 datasets × 4 scenarios × 3 seeds | **216** |
| Attack matrix (MNIST + F-MNIST) | 6 cells × 2 datasets × 2 scenarios × 4 attacks × 2 seeds | **192** |
| Attack matrix (CIFAR-10) | 4 cells (M1, M3, M5, M6) × 1 dataset × 1 scenario × 4 attacks × 2 seeds | **32** |
| β sweep (CSRAReward only) | 3 β × 2 datasets × 1 scenario × 3 seeds | **18** |
| **TỔNG** | | **458 runs** |

### 5.5. Ước lượng thời gian compute

| Dataset | Thời gian/run | Tổng |
|---|---|---|
| MNIST | ~3-5 phút | ~10-17 giờ |
| Fashion-MNIST | ~3-5 phút | ~10-17 giờ |
| CIFAR-10 | ~30-45 phút | ~52-78 giờ |
| **Continuous total** | | **~3-5 ngày** |
| **Realistic (có debug, retry)** | | **~7-10 ngày** |

---

## 6. Metrics Báo Cáo

### 6.1. Bốn nhóm metric chính

**Nhóm A — Accuracy:**
- `final_accuracy`: accuracy round cuối
- `peak_accuracy`: max accuracy đạt được
- `convergence_round`: round đầu tiên đạt 95% peak, giữ trong 5 rounds

**Nhóm B — Fairness:**
- `jain_index` (↑ tốt hơn)
- `gini_coefficient` (↓ tốt hơn)
- `reward_variance` (↓ ổn định hơn)
- `fairness_gap` = `mean |r_i/R − c_i/C|` (↓ tốt hơn)

**Nhóm C — Reward Correctness:**
- `reward_quality_correlation` (Pearson)
- `reward_data_correlation` (Pearson)
- `reward_concentration_topk` — % reward tập trung vào top-k clients
- `reward_reputation_correlation` (kiểm tra CSRA có honour reputation không)

**Nhóm D — Attack Robustness:**
- `reward_leakage` = total_malicious_reward / total_reward
- `attack_detection_rate` = TPR
- `false_positive_rate` = FPR
- `reward_ratio` = mean(honest_reward) / mean(malicious_reward)
- `eii` (Economic Incentive Index)

### 6.2. Statistical tests

- **Mann-Whitney U** (unpaired): so sánh cặp config trên cùng dataset/scenario
- **Wilcoxon signed-rank** (paired): so sánh cặp config trên cùng seed
- α = 0.05, hai chiều
- Effect size: rank-biserial correlation

### 6.3. Plots cần sinh

| Nhóm | Plots |
|---|---|
| Accuracy | `baseline_accuracy_curve_*.png`, `convergence_round_*.png`, `convergence_scatter_*.png` |
| Fairness | `fairness_boxplot_*.png`, `fairness_jain_gini_*.png`, `fairness_reward_vs_quality_*.png` |
| Reward | `reward_concentration_*.png`, `reward_correlation_heatmap_*.png` |
| Attack | `attack_accuracy_*.png`, `attack_reward_share_*.png`, `leakage_by_attack_type_*.png` |
| Sweep | `beta_sensitivity_*.png` (3 subplots: accuracy, jain, leakage) |

---

## 7. Refactor Code

### 7.1. Cấu trúc module mới

```
fl/
├── reward_policies.py        # MỚI — 4 reward functions
├── aggregation_methods.py    # MỚI — 3 aggregation strategies
├── server_base.py            # MỚI — base class chung
├── client.py                 # giữ
├── client_csra.py            # giữ (cho anomaly_score reporting)
├── client_attacks.py         # MỚI — gom attack clients (free_rider, lazy, label_noise, sign_flip)
├── blockchain.py             # giữ — chỉ làm audit
├── logger.py                 # SỬA — thêm cột mới
├── metrics.py                # SỬA — thêm reward_concentration_topk, reward_data_correlation
├── data_utils.py             # giữ
├── models.py                 # giữ
├── config.py                 # SỬA — thêm constants cho aggregation/reward enums
└── normalization.py          # giữ

experiments/
├── run_experiment.py         # GỘP — 1 entrypoint duy nhất, flags --aggregation --reward-policy
├── run_all.sh                # SỬA — gọi entrypoint mới
└── (xóa run_experiment_csra.py, run_experiment_trimmed.py)

analysis/
├── loader.py                 # SỬA — regex filename mới, parse cột mới
├── stats.py                  # SỬA — group theo (aggregation, reward_policy)
├── plots.py                  # SỬA — đổi naming, thêm plots mới
├── report.py                 # SỬA — sections theo ablation 2 chiều
└── style.py                  # SỬA — color mapping cho 6 cells
```

### 7.2. API `fl/reward_policies.py`

```python
"""4 reward policies, mỗi hàm trả về dict {client_id: reward_eth}."""
import numpy as np
from typing import Dict, Sequence

def equal_split(
    client_ids: Sequence[int],
    total_reward: float,
) -> Dict[int, float]: ...

def data_size_reward(
    client_ids: Sequence[int],
    data_sizes: Sequence[int],
    total_reward: float,
) -> Dict[int, float]: ...

def quality_reward(
    client_ids: Sequence[int],
    quality_scores: Sequence[float],
    total_reward: float,
) -> Dict[int, float]: ...

def csra_reward(
    client_ids: Sequence[int],
    quality_scores: Sequence[float],
    data_sizes: Sequence[int],
    reputations: Sequence[float],
    total_reward: float,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
) -> Dict[int, float]:
    """
    W_i = β·q̃_i + γ·d̃_i + δ·ρ̃_i
    r_i = R · W_i / Σ W_j
    """
    ...
```

**Invariants được test:**
- `sum(rewards.values()) ≈ total_reward` (trừ EqualSplit có lẻ số)
- `all(r >= 0)`
- `β + γ + δ ≈ 1` (trong csra_reward)
- Khi mọi input bằng nhau → fallback về EqualSplit

### 7.3. API `fl/aggregation_methods.py`

```python
"""3 aggregation strategies, mỗi hàm nhận parameters list, trả về aggregated parameters."""

def fedavg_aggregation(
    client_params: list[ClientParameters],
    weights: list[float],
) -> Parameters: ...

def trimmed_mean_aggregation(
    client_params: list[ClientParameters],
    trim_ratio: float = 0.1,
) -> Parameters: ...

def csra_dcd_aggregation(
    client_params: list[ClientParameters],
    anomaly_scores: list[float],
    mad_threshold: float = 3.0,
) -> tuple[Parameters, list[bool]]:
    """Returns (aggregated_params, is_anomaly_mask)."""
    ...
```

### 7.4. Unified runner

```bash
python experiments/run_experiment.py \
    --dataset mnist \
    --scenario K3 --dirichlet-alpha 0.1 \
    --aggregation csra_dcd \
    --reward-policy csra \
    --beta 0.5 --gamma 0.3 --delta 0.2 \
    --attack free_rider \
    --seed 42 \
    --n-rounds 50
```

Flags chính:
- `--aggregation`: `fedavg | trimmed_mean | csra_dcd`
- `--reward-policy`: `equal | data_size | quality | csra`
- `--attack`: `none | free_rider | lazy | label_noise | sign_flip`
- `--beta`, `--gamma`, `--delta`: chỉ áp dụng với `--reward-policy csra`
- `--seed`: bắt buộc (không random ngầm)

### 7.5. Server base class

```python
class FLServerBase(fl.server.strategy.Strategy):
    def __init__(
        self,
        aggregation: Callable,
        reward_policy: Callable,
        reward_kwargs: dict,
        blockchain: BlockchainBridge | None = None,
        logger: ExperimentLogger | None = None,
    ): ...

    def aggregate_fit(self, server_round, results, failures):
        # 1. extract client params + metadata
        # 2. apply aggregation → (agg_params, anomaly_mask)
        # 3. valid_clients = [c for c, anom in zip(clients, anomaly_mask) if not anom]
        # 4. apply reward_policy on valid_clients → rewards dict
        # 5. flagged clients get reward = 0
        # 6. blockchain.distribute_rewards(rewards) if enabled
        # 7. logger.log_round(...)
        # 8. return agg_params
```

→ Loại bỏ `server.py`, `server_csra.py`, `server_trimmed.py` → còn 1 file `server_base.py`.

---

## 8. Schema Logger Mới

### 8.1. Filename format

```
<dataset>_<scenario>[_da<dirichlet>]_<agg>_<reward>_b<β>g<γ>d<δ>_s<seed>[_<attack>]_<timestamp>.csv
```

Ví dụ:
- `mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_143022.csv`
- `cifar10_K3_da010_csra_dcd_csra_b50g30d20_s2024_free_rider_20260521_091533.csv`

Quy ước viết tắt:
- Aggregation: `fedavg`, `trimmed`, `csra_dcd`
- Reward: `equal`, `data`, `quality`, `csra`
- β,γ,δ viết dưới dạng 2 chữ số (b50 = β=0.50). Cho reward không phải CSRA thì `b00g00d00`.

### 8.2. Cột CSV mới

| Cột | Kiểu | Mô tả |
|---|---|---|
| `run_id` | str | Filename không có `.csv` |
| `dataset` | str | mnist / fashion_mnist / cifar10 |
| `scenario` | str | K1 / K2 / K3 |
| `dirichlet_alpha` | float | 0.0 nếu K1/K2 |
| **`aggregation_method`** | str | **MỚI** — fedavg / trimmed / csra_dcd |
| **`reward_policy`** | str | **MỚI** — equal / data / quality / csra |
| **`beta`, `gamma`, `delta`** | float | **MỚI** — chỉ có giá trị với CSRAReward |
| **`attack_type`** | str | **MỚI** — none / free_rider / lazy / label_noise / sign_flip |
| **`seed`** | int | **MỚI** — seed cố định |
| `round_num` | int | |
| `client_id` | int | |
| `client_type` | str | honest / free_rider / lazy / label_noise / malicious |
| `is_honest` | bool | |
| `data_size` | int | |
| `quality_score` | float | |
| `reputation` | float | |
| `w_new` | float | weight tổng dùng cho aggregation |
| `reward_eth` | float | |
| `is_anomaly` | bool | bị flag bởi CSRA-DCD filter |
| `anomaly_score` | float | `‖Δᵢ‖₂` |
| `robust_z` | float | MAD-based z-score |
| `detection_reason` | str | "accepted" / "filtered_anomaly" |
| `global_accuracy` | float | accuracy round đó |
| `run_rounds_observed` | int | tổng số rounds của run này |

### 8.3. Migration

**Không có migration.** Toàn bộ logs cũ trong `results/logs/` sẽ bị **xoá** trước khi chạy lại. Backup vào `results/logs_legacy/` nếu cần đối chiếu.

---

## 9. Lộ Trình Thực Hiện

### 9.1. Tổng quan 5 giai đoạn

| GĐ | Tên | Output | Ngày làm việc |
|---|---|---|---|
| **W1** | Refactor core module | `reward_policies.py` + `aggregation_methods.py` + `server_base.py` + smoke test 6 cells | 2-3 |
| **W2** | Refactor logger + loader + tests | New schema + unit tests PASS | 1-2 |
| **W3** | Implement attacks + blockchain audit-only | 4 attack clients hoạt động + blockchain ghi log đầy đủ | 1 |
| **W4** | Chạy full matrix | 458 CSV files trong `results/logs/` | 7-10 |
| **W5** | Analysis pipeline + báo cáo | `analysis_report.md` + plots cho luận văn | 2-3 |

**Tổng:** ~14-19 ngày làm việc.

### 9.2. Chi tiết W1 — Refactor core

| Task | Thứ tự | Test |
|---|---|---|
| Tạo `fl/reward_policies.py` với 4 hàm | 1 | `pytest tests/unit/test_reward_policies.py` |
| Tạo `fl/aggregation_methods.py` với 3 hàm | 2 | `pytest tests/unit/test_aggregation_methods.py` |
| Tạo `fl/server_base.py` (base + 1 class) | 3 | Smoke test 1 run MNIST K1 M1 |
| Refactor `experiments/run_experiment.py` thành unified runner | 4 | 6 cells × MNIST × K1 × seed 42 = 6 runs PASS |
| Xoá `run_experiment_csra.py`, `run_experiment_trimmed.py` | 5 | — |

### 9.3. Chi tiết W2 — Logger + Loader

| Task | Test |
|---|---|
| Sửa `fl/logger.py` — thêm cột mới | unit test ghi/đọc đúng |
| Sửa `analysis/loader.py` — regex filename mới | parse 6 smoke logs từ W1 |
| Sửa `analysis/stats.py` — group key mới | `compute_summary_metrics` chạy clean |
| Viết `tests/unit/test_reward_policies.py` | 80%+ coverage |
| Viết `tests/unit/test_aggregation_methods.py` | 80%+ coverage |
| Cập nhật `tests/unit/test_analysis_loader.py` | PASS |

### 9.4. Chi tiết W3 — Attacks + Blockchain

| Task | Test |
|---|---|
| Implement `client_attacks.py` với 4 attack classes | Mỗi loại smoke 1 round |
| `FreeRiderClient` — trả về global params, không train | `anomaly_score ≈ 0` |
| `LazyClient` — train 1 epoch thay vì 5 | `quality_score` thấp |
| `LabelNoiseClient` — flip 30% labels | `anomaly_score` cao |
| `SignFlipClient` — `delta_i = −delta_correct` | `anomaly_score` rất cao |
| Refactor `blockchain.py` — chỉ làm distribute, không quyết định reward weights | log đầy đủ |

### 9.5. Chi tiết W4 — Chạy matrix

Thứ tự ưu tiên (chạy MNIST/F-MNIST trước, CIFAR sau):

1. **Clean MNIST + F-MNIST** (144 runs, ~12 giờ): 6 cells × 2 ds × 4 sc × 3 seeds
2. **Attack MNIST + F-MNIST** (192 runs, ~16 giờ): 6 cells × 2 ds × 2 sc × 4 attacks × 2 seeds
3. **β sweep** (18 runs, ~1.5 giờ): 3 β × 2 ds × K3@0.1 × 3 seeds
4. **Clean CIFAR-10** (72 runs, ~36-54 giờ): 6 cells × 1 ds × 4 sc × 3 seeds
5. **Attack CIFAR-10** (32 runs, ~16-24 giờ): 4 cells × 1 ds × 1 sc × 4 attacks × 2 seeds

Mỗi batch kết thúc bằng smoke `analyze_results.py` để bắt regression sớm.

### 9.6. Chi tiết W5 — Analysis + báo cáo

| Task | Output |
|---|---|
| Sửa `plots.py` — đổi naming, thêm 4 plot mới | 25+ PNG files trong `results/plots/` |
| Sửa `report.py` — sections theo ablation 2 chiều | `analysis_report.md` đầy đủ |
| Export LaTeX tables | `results/latex/*.tex` |
| Statistical tests | `stat_tests.csv` |
| Viết phần "Discussion" trong report (manual) | Trong luận văn |

---

## 10. Quản Trị Rủi Ro

### 10.1. Rủi ro kỹ thuật

| Rủi ro | Mức độ | Mitigation |
|---|---|---|
| CIFAR-10 quá chậm, không kịp deadline | Cao | Chuẩn bị plan B: chỉ chạy CIFAR-10 với 2 cells quan trọng nhất (M1, M6) |
| Server base class phá compatibility | Trung bình | Giữ git branch riêng cho refactor, merge khi smoke test PASS |
| Sign-flip attack quá mạnh, mọi config fail | Trung bình | Thử với 10-20% sign-flip clients thay vì 100% |
| Blockchain audit log quá lớn | Thấp | Giới hạn log mỗi round chứ không mỗi client (giảm 10×) |

### 10.2. Rủi ro thiết kế

| Rủi ro | Mitigation |
|---|---|
| β=0.5, γ=0.3, δ=0.2 không tối ưu | Đã có β sweep; trong báo cáo trình bày sweep + chọn best |
| Reputation 3-dim không cải thiện đáng kể vs 2-dim | Plan B: chạy thêm M4'  với β=α, γ=1−α, δ=0 (về lại 2-dim) làm baseline so sánh |
| Reward = 0 cho flagged → vi phạm gì trong báo cáo? | Documented rõ trong §3.2 — đây là design decision có lý do (chống reward leakage) |

### 10.3. Rủi ro thời gian

- Refactor (W1-W3) chạy chậm hơn dự kiến: cắt scope đến **CSR-CSRA core** trước, attack/sweep cuối cùng.
- Compute (W4) gặp lỗi: dùng `--resume` flag trong `run_all.sh` để chạy tiếp từ chỗ failed.
- Phân tích (W5) phát hiện regression: rollback git, retest.

---

## 11. Kết Quả Kỳ Vọng

### 11.1. Hypothesis chính

| Hypothesis | So sánh | Kỳ vọng |
|---|---|---|
| H1: CSRAReward công bằng hơn EqualSplit | M4 vs M1 (clean) | fairness_gap M4 < M1, jain M4 > M1 |
| H2: CSRAReward giảm bias data | M4 vs M2 (clean) | reward_data_corr M4 < M2 |
| H3: CSRAReward ổn định hơn QualityOnly | M4 vs M3 (clean) | reward_variance M4 < M3 |
| H4: CSRA-DCD giảm leakage | M5 vs M1 (attack) | leakage M5 < M1 |
| H5: Hệ thống đầy đủ tốt nhất | M6 vs {M1..M5} (attack) | leakage M6 thấp nhất, accuracy không thấp hơn |
| H6: FPR chấp nhận được | M5, M6 (clean) | FPR < 5% |

### 11.2. Tiêu chí thành công

Báo cáo được xem là thành công nếu:
- Ít nhất **4/6 hypotheses** được khẳng định với p < 0.05.
- M6 có **reward_leakage < 20%** trên K3@0.1 với mọi attack type.
- M5, M6 có **FPR < 5%** trên clean runs (tránh false alarm).
- CSRA-DCD không **giảm final_accuracy** quá 2% so với FedAvg trên clean runs.

---

## 12. Phụ Lục — Quy Ước Naming

### 12.1. Trong code

| Đối tượng | Tên |
|---|---|
| Aggregation method enum | `Aggregation.FEDAVG`, `Aggregation.TRIMMED_MEAN`, `Aggregation.CSRA_DCD` |
| Reward policy enum | `RewardPolicy.EQUAL`, `RewardPolicy.DATA_SIZE`, `RewardPolicy.QUALITY`, `RewardPolicy.CSRA` |
| Attack type enum | `AttackType.NONE`, `AttackType.FREE_RIDER`, `AttackType.LAZY`, `AttackType.LABEL_NOISE`, `AttackType.SIGN_FLIP` |

### 12.2. Trong báo cáo (LaTeX/Markdown)

| Combination | Label hiển thị |
|---|---|
| M1 | FedAvg + EqualSplit |
| M2 | FedAvg + DataSize |
| M3 | FedAvg + QualityOnly |
| M4 | FedAvg + CSRAReward |
| M5 | CSRA-DCD + EqualSplit |
| M6 | **CSRA-DCD + CSRAReward** |

### 12.3. Trong filename CSV

Đã quy định ở §8.1.

---

## 13. Checklist Pre-flight

Trước khi start W1, đảm bảo:

- [ ] Git working tree sạch (`git status` không có thay đổi)
- [ ] Branch `refactor/reward-policies` được tạo từ `main`
- [ ] Backup `results/logs/` hiện tại sang `results/logs_legacy/`
- [ ] `pytest tests/unit -q` PASS trên branch hiện tại
- [ ] Hardhat node + contracts deployed local
- [ ] Disk có ≥ 5GB trống cho logs mới
- [ ] Confirm GVHD (nếu cần) về phạm vi 6-cell ablation

---

**Tài liệu này là contract giữa designer và implementer trong project. Mọi thay đổi scope phải update lại file này trước khi code.**
