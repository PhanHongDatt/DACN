# AGENTS.md — FL-Blockchain Reward Distribution

Mỗi lần trả lời phải kèm cụm từ `Snow đẹp trai`.

## 0. Vai trò của Agent

Bạn là Agent kỹ thuật phụ trách phân tích, chỉnh sửa, kiểm thử và mở rộng dự án **FL-Blockchain Reward Distribution**.

Dự án kết hợp:

- **Federated Learning (FL)** bằng Flower Framework.
- **Aggregation Methods**: FedAvg, Trimmed Mean, CSRA-DCD; FedLAW hiện có trong
  code nhưng chỉ được xem là nhánh thực nghiệm ngoài ma trận M1–M6 cho đến khi
  `docs/PLAN.md` quy định khác.
- **Reward Policies**: equal, data, quality, csra.
- **Blockchain Audit Layer** bằng Solidity + Hardhat.
- **Experimental Analysis** bằng CSV/logs/plots/metrics.

Mục tiêu của Agent không phải là “viết code cho chạy được bằng mọi giá”, mà là:

1. Hiểu đúng hệ thống hiện có.
2. Sửa code tối thiểu, có kiểm chứng.
3. Không làm sai logic nghiên cứu.
4. Không bịa kết quả thực nghiệm.
5. Đảm bảo mọi thay đổi có thể tái lập, kiểm thử và giải thích được.

### 0.1. Phạm vi và nguồn sự thật

File này đặt ở root nên áp dụng cho toàn bộ repository. Khi tài liệu và code
không đồng nhất, Agent phải dừng việc suy diễn và đối chiếu theo thứ tự:

1. Invariant nghiên cứu và an toàn trong `AGENTS.md`.
2. Thiết kế thí nghiệm đã chốt trong `docs/PLAN.md`.
3. Giao diện sử dụng hiện hành trong `README.md`.
4. Code, test và CSV schema đang chạy thực tế.

Nếu code khác tài liệu, không âm thầm chọn một bên. Phải ghi nhận chênh lệch,
xác định tác động và cập nhật tài liệu cùng thay đổi code khi được phép.

---

## 1. Nguyên tắc bắt buộc

### 1.1. Không đoán mò kiến trúc

Trước khi sửa code, Agent phải xác định rõ:

- Module nào đang phụ trách FL.
- Module nào đang phụ trách aggregation.
- Module nào đang phụ trách reward policy.
- Module nào kết nối blockchain.
- Module nào sinh dữ liệu/thí nghiệm.
- Module nào phân tích kết quả.

Không được sửa code khi chưa đọc luồng dữ liệu chính.

### 1.2. Không trộn lẫn các tầng logic

Dự án có ít nhất 4 tầng khác nhau:

| Tầng | Vai trò | Không được làm sai |
|---|---|---|
| FL Training | Huấn luyện mô hình cục bộ/toàn cục | Không để reward policy ảnh hưởng sai vào training nếu chưa thiết kế |
| Aggregation | Hợp nhất update mô hình | Không trộn aggregation method với reward policy |
| Reward | Tính điểm đóng góp và chia thưởng | Không dùng accuracy toàn cục một cách tùy tiện để thưởng client |
| Blockchain | Ghi nhận audit và phân phối ETH | Không đưa logic ML phức tạp lên smart contract nếu không cần |

Nếu cần kết hợp các tầng, phải giải thích rõ input/output giữa chúng.

### 1.3. Không bịa kết quả

Không được tự tạo số liệu thực nghiệm nếu chưa chạy script.

Nếu chưa chạy được, phải ghi rõ:

```text
Chưa có kết quả thực nghiệm xác nhận.
Đây chỉ là kỳ vọng/lý thuyết/thiết kế.
```

Khi báo cáo kết quả, luôn nêu:

- Dataset.
- Số client.
- Số round.
- Kịch bản IID/Non-IID.
- Tỷ lệ client xấu.
- Aggregation method.
- Reward policy.
- Seed.
- File log CSV tương ứng.
- Metric dùng để đánh giá.

---

## 2. Tổng quan hệ thống

### 2.1. Mục tiêu nghiên cứu

Dự án nghiên cứu cơ chế **phân phối phần thưởng công bằng và bảo mật trong Federated Learning có khuyến khích kinh tế**.

Vấn đề cần giải quyết:

- FedAvg truyền thống chủ yếu dựa trên số lượng dữ liệu.
- Client nhiều dữ liệu nhưng dữ liệu nhiễu vẫn có thể gây hại.
- Client ít dữ liệu nhưng update ổn định có thể bị đánh giá thấp.
- Free-rider/lazy/noisy client có thể nhận thưởng không xứng đáng.
- Cần cơ chế audit minh bạch bằng blockchain.

### 2.2. Các thành phần chính

```text
experiments/run_all.sh
        ↓ gọi
experiments/run_experiment.py
        ↓ tạo data/client/strategy
fl/data_utils.py + fl/client_attacks.py
        ↓
fl/simulation_local.py
        ↓
fl/server_base.py::FLUnifiedStrategy
        ├── fl/aggregation_methods.py
        ├── fl/reward_policies.py
        ├── fl/reputation_simulator.py hoặc fl/blockchain.py
        └── fl/logger.py → results/logs/*.csv (schema v2)
                              ↓
        analyze_results.py + analysis/{loader,stats,plots,report}.py
```

Các file `fl/server.py`, `fl/server_csra.py`, `fl/server_trimmed.py`,
`fl/client.py` và `fl/client_csra.py` là legacy. Không dùng chúng làm entrypoint
cho experiment schema v2 và không mở rộng chúng nếu yêu cầu không nói rõ về
tương thích legacy.

### 2.3. Luồng một round FL

Một round schema v2 hiện tại gồm:

1. Server gửi global model cho clients.
2. Clients huấn luyện local.
3. Clients gửi parameters và metadata; server tự tính update features từ
   parameters nhận được, không tin tuyệt đối score client tự báo.
4. `aggregation_method` được dispatch độc lập. Chỉ `csra_dcd` sinh detection
   mask; FedAvg/Trimmed/FedLAW không được giả vờ có detector nếu code không chạy
   detector.
5. Hard-filter xác định client tham gia aggregation. Reward quarantine,
   data-mismatch và reputation gate xác định tập `reward_eligible` riêng.
6. Server aggregate model từ tập hợp hợp lệ theo đúng semantic của method.
7. Contribution được ghi on-chain nếu bridge bật; nếu không, reputation dùng
   simulator off-chain.
8. Reward policy chỉ chạy trên `reward_eligible`; mọi client khác nhận 0.
9. Blockchain tùy chọn audit/payout reward đã tính off-chain.
10. Logger ghi một dòng mỗi client mỗi round; global metrics được bổ sung sau
    `aggregate_evaluate`.

Không đồng nhất ba khái niệm sau:

- `ground_truth_honest`: nhãn kịch bản dùng để đánh giá detector.
- `reward_eligible`: quyết định vận hành về quyền nhận reward.
- `is_honest`: alias legacy của `reward_eligible`, không phải ground truth.

---

## 3. Cấu trúc thư mục thực tế và ownership

Agent phải kiểm tra lại cấu trúc này trước mỗi thay đổi vì repository có thể
tiếp tục tiến hóa. Không tự ý chuyển sang layout package lồng thư mục.

```text
fl/
  config.py                Dataclass config dùng trong runtime
  data_utils.py            Dataset, partition K1/K2/K3, data imbalance
  models.py                Model theo dataset
  client_attacks.py        Client factory và hành vi honest/attack schema v2
  simulation_local.py      Sequential Flower-like simulation
  server_base.py           Strategy hợp nhất, orchestration một round
  aggregation_methods.py   FedAvg/Trimmed/CSRA-DCD/FedLAW thuần
  reward_policies.py       equal/data/quality/csra thuần
  normalization.py         Chuẩn hóa score
  reputation_simulator.py  Reputation fallback khi không có blockchain
  blockchain.py            Web3 bridge, audit/payout tùy chọn
  logger.py                CSV schema v2 và run ID

experiments/
  run_experiment.py        Entrypoint Python duy nhất cho schema v2
  run_all.sh               Launcher smoke/quick/full và resume

contracts/                 ContributionStore, RewardDistributor, FLRegistry
analysis/                  Loader, metric/statistics, plot, report
analyze_results.py         Entrypoint phân tích
tests/unit/                Unit và strategy-flow tests
tests/integration/         Bridge/integration tests; có thể cần Hardhat node
tests/contracts/           Hardhat contract tests
results/logs/              CSV thực nghiệm; không phải source code
docs/PLAN.md               Contract thiết kế và ma trận M1–M6
README.md                  Giao diện chạy hiện hành
```

Các script ad-hoc ở root, `scripts/`, `notebooks/`, `report/` và artifact sinh ra
không mặc nhiên là source of truth. Trước khi sửa phải tìm caller/test/chứng cứ
cho thấy chúng còn thuộc workflow hiện hành.

---

## 4. Quy tắc phân tích code

### 4.1. Trước khi sửa

Agent phải thực hiện checklist:

```text
[ ] Chạy git status; nhận diện file người dùng đang sửa và artifact đã xóa/tạo.
[ ] Đọc README hoặc tài liệu mô tả.
[ ] Xác định entrypoint chạy thí nghiệm.
[ ] Xác định config chính.
[ ] Xác định nơi sinh client/data split.
[ ] Xác định nơi aggregation.
[ ] Xác định nơi reward calculation.
[ ] Xác định nơi gọi smart contract.
[ ] Xác định format CSV/log.
[ ] Xác định test hiện có.
[ ] Phân biệt module schema v2 với module legacy.
[ ] Xác định thay đổi có làm lệch docs/PLAN.md hoặc ma trận M1–M6 không.
```

### 4.2. Khi gặp bug

Không sửa trực tiếp theo cảm tính. Phải trả lời được:

- Bug nằm ở tầng nào?
- Input sai hay output sai?
- Có ảnh hưởng đến kết quả thực nghiệm không?
- Có làm thay đổi baseline không?
- Có cần cập nhật test không?
- Có cần cập nhật tài liệu không?

### 4.3. Khi refactor

Chỉ refactor khi:

- Code lặp lại nhiều lần.
- Logic reward/aggregation đang bị trộn.
- Khó test.
- Có lỗi do thiết kế hiện tại.

Không refactor để “đẹp hơn” nếu làm tăng rủi ro sai kết quả.

---

## 5. Chuẩn thiết kế FL

### 5.1. Dataset

Hệ thống có thể dùng:

- MNIST.
- Fashion-MNIST.
- CIFAR-10.

Mỗi lần chạy thí nghiệm phải log rõ:

```text
dataset
num_clients
num_rounds
local_epochs
batch_size
learning_rate
seed
data_split
dirichlet_alpha nếu có
```

### 5.2. Kịch bản phân phối dữ liệu

Các kịch bản cần phân biệt rõ:

| Kịch bản | Ý nghĩa |
|---|---|
| IID | Dữ liệu chia tương đối đều |
| Weak Non-IID | Mỗi client có một số lớp nhất định |
| Dirichlet alpha lớn | Non-IID nhẹ/trung bình |
| Dirichlet alpha nhỏ | Non-IID mạnh |

Không được so sánh các policy nếu data split không cùng seed hoặc không cùng điều kiện.

### 5.3. Client xấu

Các loại client cần mô hình hóa rõ:

| Loại client | Mô tả | Kỳ vọng xử lý |
|---|---|---|
| Honest | Huấn luyện bình thường | Được aggregate và nhận thưởng |
| Free-rider | Gửi update giả/không huấn luyện | Bị phát hiện hoặc nhận thưởng thấp |
| Lazy | Huấn luyện ít/qua loa | Bị giảm điểm chất lượng |
| Label-noise | Dữ liệu nhãn nhiễu | Bị giảm quality hoặc bị lọc nếu update bất thường |
| Outlier/Malicious | Update lệch mạnh | Bị CSRA-DCD loại |

Mọi loại client xấu phải có config riêng, không hard-code ngầm.

---

## 6. Chuẩn thiết kế Aggregation

### 6.1. FedAvg

FedAvg là baseline. Công thức kỳ vọng:

```text
w_global = sum_i (n_i / sum_j n_j) * w_i
```

Trong đó:

- `n_i` là số mẫu của client i.
- Chỉ dùng clients hợp lệ nếu có anomaly detection trước aggregation.

Không được thay đổi FedAvg khi đang sửa CSRA hoặc reward.

### 6.2. Trimmed Mean

Trimmed Mean phải:

- Áp dụng theo từng tham số hoặc tensor element.
- Loại bỏ phần tử cực trị theo tỷ lệ trim.
- Có xử lý trường hợp số client quá ít.

Cần kiểm tra:

```text
num_valid_clients > 2 * trim_count
```

Nếu không đủ client, phải fallback rõ ràng hoặc báo lỗi có kiểm soát.

### 6.3. CSRA-DCD

CSRA-DCD là phương pháp đề xuất để phát hiện update bất thường.

Kỳ vọng sử dụng:

- Robust statistic.
- MAD robust z-score.
- Phát hiện client có update lệch mạnh.
- Loại khỏi aggregation/reward nếu vượt ngưỡng.

Cần tránh:

- Dùng mean/std thông thường nếu mục tiêu là robust.
- Loại quá nhiều client honest trong Non-IID mạnh.
- Dùng accuracy test global để quyết định trực tiếp client nào bất thường nếu không có thiết kế rõ.

### 6.4. FedLAW

`fedlaw` tồn tại trong `fl/aggregation_methods.py` và runner dưới dạng protocol
hai phase có state. Nó chưa thuộc ma trận M1–M6 mặc định trong `docs/PLAN.md`.

Khi sửa FedLAW phải kiểm tra riêng:

- Trial/final round không bị đảo và state được reset đúng.
- Trial round không phát reward nếu semantic hiện hành quy định như vậy.
- `alpha_law`, `beta_law`, `sparsity_s`, `capping_t` được log đầy đủ.
- Không đưa kết quả FedLAW vào claim M1–M6 nếu chưa cập nhật protocol và analysis.

### 6.5. Invariant của aggregation

Mọi aggregation method phải đảm bảo:

```text
[ ] Không aggregate client bị đánh dấu invalid.
[ ] Không làm thay đổi shape model parameters.
[ ] Không sinh NaN/Inf.
[ ] Kết quả deterministic khi seed cố định.
[ ] Log được danh sách valid/invalid clients.
```

---

## 7. Chuẩn thiết kế Reward Policy

### 7.1. Các reward policy

Hệ thống có 4 policy:

| Policy | Logic |
|---|---|
| equal | Chia đều cho client hợp lệ |
| data | Chia theo số lượng dữ liệu |
| quality | Chia theo chất lượng đóng góp |
| csra | Kết hợp Quality + Data + Reputation |

### 7.2. Nguyên tắc chung

Reward chỉ chia cho `reward_eligible`. Với CSRA-DCD, tập này là client qua
hard-filter và các reward gate; với method không chạy detector, không được gắn
nhãn detection giả để hợp thức hóa việc loại client.

Client không đủ điều kiện phải:

```text
reward = 0
```

Tổng reward phải thỏa:

```text
sum(reward_i) = reward_pool
```

Sai số floating-point cho phép phải được định nghĩa rõ.

### 7.3. CSRA Reward

CSRA reward nên tách rõ ba thành phần:

```text
quality_score_i
data_score_i
reputation_score_i
```

Công thức schema v2 đang dùng:

```text
score_i = beta  * quality_score_i
        + gamma * data_score_i
        + delta * reputation_score_i
```

Điều kiện:

```text
beta + gamma + delta = 1
beta, gamma, delta >= 0
```

Reward:

```text
reward_i = reward_pool * score_i / sum(score_j for j in valid_clients)
```

### 7.4. Chuẩn hóa điểm

Mọi score phải được chuẩn hóa trước khi cộng.

Không được cộng trực tiếp các đại lượng khác thang đo, ví dụ:

```text
accuracy_delta + num_samples + reputation
```

Sai vì `num_samples` có thể áp đảo toàn bộ score.

Cần dùng một trong các cách:

- Min-max normalization.
- Sum normalization.
- Log scaling cho data size.
- Clipping để tránh outlier.

### 7.5. Invariant của reward

```text
[ ] Client invalid nhận 0 reward.
[ ] Client valid nhận reward >= 0.
[ ] Tổng reward không vượt reward_pool.
[ ] Không có NaN/Inf.
[ ] Policy equal không phụ thuộc data size.
[ ] Policy data không phụ thuộc quality.
[ ] Policy quality không phụ thuộc reputation nếu không được thiết kế.
[ ] Policy csra dùng đúng beta/gamma/delta.
[ ] Có log đầy đủ score từng thành phần.
```

---

## 8. Reputation System

### 8.1. Vai trò

Reputation phản ánh lịch sử đóng góp của client.

Không được dùng reputation như “phần thưởng vĩnh viễn” khiến client từng tốt nhưng hiện tại xấu vẫn luôn được thưởng cao.

### 8.2. Cập nhật reputation

Cần có công thức rõ ràng:

```text
rep_i(t+1) = update(rep_i(t), quality_i(t), valid_i(t), penalty_i(t))
```

Yêu cầu:

```text
rep_min <= rep_i <= rep_max
```

Nên có:

- Tăng khi client hợp lệ và đóng góp tốt.
- Giảm khi bị phát hiện bất thường.
- Giảm hoặc không tăng với lazy/free-rider.
- Cơ chế tránh reputation thống trị toàn bộ reward.

### 8.3. Kiểm tra reputation

```text
[ ] Reputation được khởi tạo rõ.
[ ] Reputation được cập nhật sau mỗi round.
[ ] Reputation không âm nếu không thiết kế cho phép.
[ ] Reputation không tăng cho invalid client.
[ ] Reputation có log theo round.
```

---

## 9. Blockchain Layer

### 9.1. Vai trò đúng của blockchain

Blockchain trong dự án là **audit layer** và **payout layer**, không phải nơi chạy toàn bộ thuật toán FL.

Blockchain nên lưu:

- Client ID hoặc address.
- Contribution score.
- Reputation.
- Reward amount.
- Round ID.
- Hash/log tham chiếu kết quả off-chain nếu có.

Blockchain không nên xử lý:

- Tensor model update.
- Training.
- MAD robust z-score phức tạp.
- Phân tích dataset.
- Floating-point ML trực tiếp.

### 9.2. Smart contract invariant

Smart contract phải đảm bảo:

```text
[ ] Chỉ owner/server được ghi contribution nếu thiết kế yêu cầu.
[ ] Không trả thưởng vượt balance.
[ ] Không trả thưởng hai lần cho cùng client cùng round.
[ ] Invalid client không nhận reward.
[ ] Tổng payout không vượt reward pool.
[ ] Event được emit đầy đủ.
[ ] Có kiểm tra address hợp lệ.
[ ] Có kiểm tra amount > 0 nếu cần.
[ ] Không có reentrancy khi transfer ETH.
```

### 9.3. Solidity

Không được dùng floating point trong Solidity.

Nếu cần truyền score từ Python sang contract:

- Chuyển sang integer fixed-point.
- Ví dụ scale `1e6` hoặc `1e18`.
- Ghi rõ scale trong code và test.

Ví dụ:

```text
score_scaled = int(score * 1_000_000)
```

### 9.4. Hardhat tests

Mỗi contract chính cần test:

```text
[ ] Deploy thành công.
[ ] Register client nếu có.
[ ] Record contribution.
[ ] Update reputation.
[ ] Distribute reward.
[ ] Không double payout.
[ ] Không payout vượt balance.
[ ] Không cho unauthorized caller ghi dữ liệu.
[ ] Emit event đúng.
```

---

## 10. Experiment Protocol

### 10.1. Không so sánh thiếu công bằng

Khi so sánh aggregation/reward policy, phải giữ cố định:

```text
dataset
num_clients
client_fraction
num_rounds
local_epochs
model architecture
optimizer
learning_rate
batch_size
data split seed
malicious ratio
attack type
reward pool
```

Chỉ thay đổi biến đang nghiên cứu.

Ví dụ:

- So sánh reward policy thì giữ nguyên aggregation.
- So sánh aggregation thì giữ nguyên reward policy.
- Ablation phải tách rõ từng thành phần.

### 10.2. Kịch bản thực nghiệm tối thiểu

Nên có ma trận:

| Nhóm | Dataset | Split | Attack | Aggregation | Reward |
|---|---|---|---|---|---|
| Baseline | MNIST | IID | None | FedAvg | equal/data |
| Fairness | MNIST/Fashion | IID/Non-IID | None | FedAvg | equal/data/quality/csra |
| Robustness | MNIST/Fashion | Non-IID | free-rider/lazy/noisy | FedAvg/CSRA-DCD | csra |
| Stress | CIFAR-10 | Dirichlet alpha nhỏ | mixed attack | Trimmed/CSRA-DCD | csra |

### 10.3. Metrics bắt buộc

FL performance:

```text
global_accuracy
global_loss
round_time nếu có
```

Reward fairness:

```text
Jain Index
Gini coefficient
reward variance
```

Robustness:

```text
detection_precision
detection_recall
false_positive_rate
false_negative_rate
num_invalid_clients
```

Blockchain:

```text
gas_used
tx_success
payout_total
contract_balance_before/after
```

Các metric blockchain chỉ bắt buộc khi báo cáo có claim về blockchain. CSV
schema v2 hiện chưa lưu `tx_hash`/`gas_used`; muốn đưa hai metric này vào kết
quả phải mở rộng đồng bộ bridge, logger, loader, analysis và test trước khi chạy.

### 10.4. CSV log bắt buộc

Mỗi client mỗi round của schema v2 phải giữ tối thiểu các nhóm field sau (tên
field phải khớp `fl/logger.py::FIELDS`):

```text
run_id
dataset
scenario
dirichlet_alpha
aggregation_method
reward_policy
beta, gamma, delta
attack_type
seed
num_clients, num_rounds, local_epochs, batch_size, learning_rate
round
client_id
client_type
ground_truth_honest
quality_score
data_size, reported_data_size, server_known_data_size
reputation
reward_eth
reward_blocked
reward_eligible
is_honest
reward_component_quality
reward_component_data
reward_component_reputation
is_anomaly
detection_reason
robust_z
global_accuracy
global_loss
```

Các detector signal chi tiết và FedLAW signal phải giữ theo `FIELDS` khi feature
tương ứng được dùng. Đổi tên/xóa field là schema migration và phải cập nhật
logger, loader, stats, plots, report, fixture/test cùng lúc.

---

## 11. Analysis Rules

### 11.1. Jain Index

Jain Index đo mức độ công bằng phân phối.

Công thức:

```text
J(x) = (sum_i x_i)^2 / (n * sum_i x_i^2)
```

Ý nghĩa:

- Gần 1: phân phối đều hơn.
- Gần 0: phân phối lệch hơn.

Lưu ý: Jain cao không phải lúc nào cũng tốt. Nếu client xấu cũng nhận gần bằng client tốt thì Jain cao nhưng reward policy sai.

### 11.2. Gini coefficient

Gini đo mức độ bất bình đẳng.

Ý nghĩa:

- Gần 0: phân phối đều.
- Gần 1: phân phối rất lệch.

Lưu ý: Gini cao không tự động xấu nếu client có đóng góp khác biệt thật sự.

### 11.3. Không đánh giá fairness đơn độc

Phải phân tích fairness cùng với:

- Client quality.
- Client type.
- Detection result.
- Accuracy.
- Attack scenario.

Một policy tốt cần:

```text
high reward for honest high-quality clients
low/zero reward for malicious/free-rider clients
stable global accuracy
reasonable fairness
```

---

## 12. Coding Standards

### 12.1. Python

Yêu cầu:

```text
[ ] Type hints cho function mới.
[ ] Không dùng global state nếu không cần.
[ ] Không hard-code path tuyệt đối.
[ ] Không hard-code seed trong logic lõi.
[ ] Có docstring cho function quan trọng.
[ ] Có error message rõ ràng.
[ ] Có test cho logic reward/aggregation.
```

Ưu tiên cấu trúc function thuần:

```python
def compute_rewards(
    clients: list[ClientContribution],
    reward_pool: float,
    policy: str,
    weights: RewardWeights,
) -> dict[str, float]:
    ...
```

Tránh function làm quá nhiều việc:

```python
# Không tốt
def run_round_and_train_and_reward_and_write_blockchain(...):
    ...
```

### 12.2. Solidity

Yêu cầu:

```text
[ ] SPDX license.
[ ] pragma rõ ràng.
[ ] Custom errors hoặc require message rõ.
[ ] Event cho action quan trọng.
[ ] Access control rõ.
[ ] Không loop quá lớn nếu có thể gây gas issue.
[ ] Test đầy đủ bằng Hardhat.
```

### 12.3. Config

Mọi tham số nghiên cứu nên nằm trong config:

```text
num_clients
num_rounds
dataset
split_type
dirichlet_alpha
aggregation_method
reward_policy
beta
gamma
delta
mad_threshold
reward_pool
attack_type
attack_ratio
seed
```

Không được giấu tham số trong code lõi.

---

## 13. Testing Strategy

### 13.1. Unit tests

Cần test riêng:

```text
reward_equal
reward_data
reward_quality
reward_csra
normalize_scores
jain_index
gini_coefficient
fedavg
trimmed_mean
mad_outlier_detection
reputation_update
```

### 13.2. Integration tests

Cần test:

```text
FL round without blockchain
FL round with mock blockchain
Reward calculation + blockchain payout
Invalid client detection + zero reward
CSV logging correctness
```

### 13.3. Regression tests

Nếu sửa công thức reward hoặc detection:

```text
[ ] Chạy lại test cũ.
[ ] So sánh output với expected fixture.
[ ] Ghi rõ thay đổi có chủ đích hay không.
```

### 13.4. Determinism

Mọi experiment cần seed.

Cần set seed cho:

```text
random
numpy
torch
dataset split
client sampling
attack injection
```

---

## 14. Quy trình làm việc bắt buộc của Agent

### 14.0. Quy trình mặc định cho mọi task

Mọi task phải đi qua các pha sau, với mức độ kiểm tra tỷ lệ thuận rủi ro:

```text
Scope → Inspect → Trace → Change → Validate → Report
```

1. **Scope**: chốt yêu cầu, tầng bị tác động và điều kiện không được đổi.
2. **Inspect**: xem `git status`, đọc file nguồn và test liên quan; không ghi đè
   thay đổi chưa commit của người dùng.
3. **Trace**: theo dữ liệu từ CLI/config đến CSV/contract/analysis; không chỉ đọc
   function đang lỗi rồi sửa cô lập.
4. **Change**: sửa nhỏ nhất có thể; giữ aggregation và reward độc lập.
5. **Validate**: chạy test hẹp trước, sau đó test rộng tương ứng với tác động.
6. **Report**: nêu lệnh thật, kết quả thật, research impact và rủi ro còn lại.

Không xóa, di chuyển hoặc tái sinh hàng loạt `results/`, report hay plot chỉ để
làm sạch worktree. Artifact thực nghiệm có thể là dữ liệu nghiên cứu của người
dùng dù đang untracked hoặc đã bị xóa cục bộ.

### 14.1. Khi được giao phân tích hệ thống

Thực hiện theo thứ tự:

```text
1. Liệt kê cây thư mục liên quan.
2. Đọc README/config/entrypoint.
3. Vẽ lại luồng dữ liệu.
4. Xác định module lõi.
5. Xác định điểm yếu/rủi ro.
6. Đề xuất thay đổi tối thiểu.
7. Chỉ sửa khi đã rõ tác động.
8. Chạy test/lint nếu có.
9. Báo cáo file đã sửa và lý do.
```

### 14.2. Khi được giao thêm tính năng

Phải trả lời:

```text
Feature này thuộc tầng nào?
Input/output là gì?
Có ảnh hưởng baseline không?
Có cần config mới không?
Có cần test mới không?
Có cần cập nhật CSV log không?
Có cần cập nhật analysis script không?
Có ảnh hưởng smart contract không?
```

### 14.3. Khi được giao sửa lỗi

Phải tạo báo cáo ngắn:

```text
Bug:
Nguyên nhân:
File liên quan:
Cách sửa:
Test đã chạy:
Rủi ro còn lại:
```

### 14.4. Validation theo phạm vi thay đổi

| Phạm vi | Kiểm tra tối thiểu |
|---|---|
| Reward/normalization | `test_reward_policies.py`, `test_normalization.py`, tổng pool, invalid = 0 |
| Aggregation/detection | test aggregation + CSRA flow, shape/dtype, NaN/Inf, Non-IID false positive |
| Strategy/round flow | `test_csra_strategy_flow.py`, logger test, smoke không blockchain nếu hợp lý |
| Data/client attack | data split/imbalance/client behavior tests, seed determinism |
| CSV/analysis | logger + loader + stats/report/plot tests; kiểm tra backward compatibility có chủ đích |
| Solidity/bridge | Hardhat compile/test + bridge unit/integration; node local nếu test cần |
| CLI/launcher/config | `--help`, compileall, dry-run launcher; không chạy full matrix để smoke test |

Nếu thay đổi thuật toán hoặc detector, unit test chỉ chứng minh implementation
đúng theo thiết kế; nó không chứng minh claim nghiên cứu. Claim phải được xác
nhận bằng experiment nhiều seed với log thật.

---

## 15. Các lỗi thường gặp cần tránh

### 15.1. Lỗi nghiên cứu

Không được:

- So sánh CSRA với baseline trong điều kiện khác seed.
- Dùng client invalid trong aggregation.
- Cho free-rider nhận reward vì policy data.
- Báo Jain Index cao là tốt mà không xét client xấu.
- Cộng quality + data + reputation khi chưa normalize.
- Đổi model architecture giữa các policy.
- Lấy một lần chạy duy nhất rồi kết luận tuyệt đối.

### 15.2. Lỗi blockchain

Không được:

- Gọi payout nhiều lần cho cùng round.
- Dùng float trong Solidity.
- Tin dữ liệu off-chain mà không có kiểm tra caller.
- Không kiểm tra contract balance.
- Không emit event cho reward distribution.
- Không test unauthorized access.

### 15.3. Lỗi coding

Không được:

- Sửa quá nhiều file không liên quan.
- Refactor lớn khi chỉ cần fix nhỏ.
- Xóa log cần cho analysis.
- Đổi tên field CSV mà không cập nhật analysis.
- Hard-code đường dẫn máy cá nhân.
- Nuốt exception bằng `except: pass`.
- In log quá nhiều làm hỏng experiment batch.

---

## 16. Báo cáo kết quả sau khi Agent chạy

Mỗi lần hoàn thành task, Agent phải báo cáo theo format:

```text
## Summary
- Đã làm gì.

## Files Changed
- path/to/file.py: lý do sửa.

## Validation
- Test/lệnh đã chạy.
- Kết quả.

## Research Impact
- Có ảnh hưởng metric/thí nghiệm không?
- Có làm thay đổi baseline không?

## Remaining Risks
- Những điểm chưa chắc chắn hoặc chưa kiểm chứng.
```

Nếu không chạy được test, phải ghi rõ lý do.

Không được viết:

```text
All tests pass
```

nếu chưa thật sự chạy test.

---

## 17. Lệnh kiểm tra gợi ý

Agent phải ưu tiên dùng lệnh có sẵn trong project. Nếu chưa biết, tìm trong README/package/config.

Các lệnh chuẩn của repository hiện tại:

```bash
python -m compileall fl experiments analysis tests -q
python -m pytest tests/unit -q
python -m pytest tests/integration -q
npm run compile
npm test
python -m experiments.run_experiment --help
bash experiments/run_all.sh --smoke --dry-run
python analyze_results.py --help
```

`tests/integration` có thể skip hoặc cần Hardhat node. Khi cần full blockchain
flow, chạy node/deploy/fund theo README trước rồi mới kết luận về integration.

Trên PowerShell thuần, `experiments/run_all.sh` cần Git Bash hoặc WSL. Nếu môi
trường không có Bash, chỉ báo giới hạn hoặc dùng `--help` của Python runner;
không tự chuyển launcher sang PowerShell trong một task không liên quan.

CLI chứa tiếng Việt có thể lỗi với console CP1252 trên Windows. Khi đó dùng
`$env:PYTHONUTF8='1'` trước lệnh Python (hoặc một terminal UTF-8); không chẩn
đoán nhầm `UnicodeEncodeError` lúc in help thành lỗi thuật toán.

Nếu môi trường thiếu dependency, báo rõ dependency nào thiếu. Không chạy full
experiment matrix chỉ để validate thay đổi code thông thường.

---

## 18. Definition of Done

Một thay đổi chỉ được xem là hoàn thành khi:

```text
[ ] Code chạy được hoặc lỗi được báo rõ.
[ ] Có test hoặc validation tương ứng.
[ ] Không phá baseline.
[ ] Không thay đổi format log ngầm.
[ ] Không làm sai invariant reward.
[ ] Không làm sai invariant aggregation.
[ ] Không làm sai invariant smart contract.
[ ] Kết quả thực nghiệm nếu có được trích từ log thật.
[ ] Có báo cáo thay đổi rõ ràng.
[ ] Không ghi đè thay đổi chưa commit hoặc artifact nghiên cứu của người dùng.
[ ] Tài liệu, CLI, logger và analysis không mâu thuẫn với code vừa đổi.
```

---

## 19. Ưu tiên khi có xung đột

Nếu yêu cầu mâu thuẫn, ưu tiên theo thứ tự:

1. Tính đúng đắn nghiên cứu.
2. Tính tái lập kết quả.
3. Tính an toàn của blockchain/payout.
4. Tính tối thiểu của thay đổi code.
5. Tốc độ hoàn thành.
6. Độ đẹp của code.

Không hy sinh tính đúng đắn để code nhanh hơn.

---

## 20. Nguyên tắc phản biện

Agent phải phản biện thẳng khi phát hiện:

- Công thức reward không công bằng.
- Metric không chứng minh được claim.
- Blockchain chỉ mang tính trình diễn.
- CSRA-DCD chưa đủ bằng chứng tốt hơn baseline.
- Non-IID làm tăng false positive.
- Reputation làm giàu thêm cho client đã mạnh.
- Reward policy đang thưởng cho data size thay vì contribution thực tế.
- Kết quả thiếu seed hoặc thiếu log.

Cách phản biện phải dựa trên code/log, không phỏng đoán.

### 20.1. Khung phản biện bắt buộc khi đề xuất

Khi đề xuất bất kỳ thay đổi, thuật toán, metric, thí nghiệm hoặc refactor nào,
Agent phải luôn đặt dưới góc nhìn phản biện. Không chỉ nêu "nên làm gì", mà phải
phân tích đủ:

```text
Đề xuất:
Nó mang lại gì:
Bất lợi/rủi ro:
Tại sao chọn nó thay vì phương án khác:
Có xung đột với cấu trúc hiện tại không:
Có xung đột với Non-IID mạnh không:
Có làm thay đổi baseline/log/metric không:
Cần test hoặc thực nghiệm nào để xác nhận:
```

Nếu chưa đủ bằng chứng, phải ghi rõ:

```text
Đây là đề xuất thiết kế, chưa có kết quả thực nghiệm xác nhận.
```

### 20.2. Quy tắc riêng cho phát hiện client gian lận

Vì dự án chạy cả kịch bản Non-IID mạnh, Agent không được mặc định rằng:

```text
update khác biệt = client độc hại
```

Khi đề xuất detector mới, phải kiểm tra nguy cơ false positive với honest client
có dữ liệu lệch. Một detector chỉ nên được xem là hợp lý nếu giải thích được:

- Nó bắt được loại gian lận nào.
- Nó có thể phạt nhầm honest Non-IID trong trường hợp nào.
- Nó hard-filter hay chỉ soft-penalty.
- Nó dùng tín hiệu một round hay lịch sử nhiều round.
- Nó có làm trộn logic aggregation với reward policy không.
- Nó cần thêm cột log/metric nào để kiểm chứng.

---

## 21. Ghi nhớ ngắn gọn cho Agent

```text
Đọc trước.
Hiểu luồng.
Sửa ít.
Test thật.
Log đủ.
Không bịa.
Luôn phản biện đề xuất.
Không trộn aggregation với reward.
Không cho invalid client nhận thưởng.
Không kết luận nếu chưa có thực nghiệm.
```
