# FL-Blockchain Reward Distribution

Dự án: **Cải tiến cơ chế phân phối phần thưởng dựa trên đóng góp đa chiều trong hệ thống Học máy Liên kết (Federated Learning) kết hợp Blockchain**

Hệ thống tích hợp mô hình Học máy liên kết (Flower) với Blockchain (Hardhat/Ethereum) để ghi nhận đóng góp của các thực thể tham gia (clients) dựa trên các tiêu chí đa chiều: Chất lượng mô hình, Độ ổn định và Danh tiếng, từ đó thực hiện phân phối phần thưởng (ETH) một cách minh bạch và công bằng.

---

## 1. Yêu cầu hệ thống
Tài liệu hướng dẫn được tối ưu hóa cho hệ điều hành **Ubuntu 22.04 LTS**.

*   **Hệ điều hành:** Ubuntu 22.04 LTS (khuyến nghị) hoặc 20.04.
*   **CPU:** Tối thiểu 4 Cores.
*   **RAM:** Tối thiểu 8GB (Khuyến nghị 16GB cho các thực nghiệm quy mô lớn).
*   **Disk:** 10GB không gian trống.
*   **Môi trường:** Python 3.11, Node.js 20.x.

---

## 2. Quy trình cài đặt chi tiết

Thực hiện các bước thiết lập môi trường trên Terminal:

### Bước 2.1: Cài đặt Python 3.11
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-pip python3.11-venv
```

### Bước 2.2: Cài đặt Node.js 20 (Hardhat Runtime)
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version
```

### Bước 2.3: Thiết lập môi trường ảo và cấu hình
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Bước 2.4: Cài đặt Node.js dependencies và biên dịch Smart Contracts
```bash
npm install
npm run compile
```

### Bước 2.5: Cài đặt Docker (Tùy chọn - Khuyến nghị để triển khai nhanh)
Nếu bạn muốn sử dụng Docker để khởi tạo nhanh môi trường Blockchain, hãy thực hiện các lệnh sau:
```bash
# Cài đặt Docker
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Cấp quyền cho user (Cần logout và login lại sau khi chạy lệnh này)
sudo usermod -aG docker $USER
```

---

## 3. Hướng dẫn vận hành

Hệ thống cung cấp hai phương thức vận hành: Triển khai thủ công hoặc sử dụng Docker.

### Cách 1: Sử dụng Docker Compose (Nhanh chóng)
Phương thức này sẽ tự động khởi tạo Hardhat node, Deploy contracts và Fund accounts trong một bước:
```bash
cd docker
docker compose up -d
```
Sau đó, bạn có thể tiến hành chạy thực nghiệm ngay tại Bước 4.

### Cách 2: Triển khai thủ công (Manual)
Hệ thống yêu cầu mạng Blockchain (Hardhat) hoạt động trước khi thực hiện các tiến trình Federated Learning.

#### Bước 3.1: Khởi động mạng Blockchain nội bộ
Mở Terminal 1:
```bash
npx hardhat node
```

#### Bước 3.2: Triển khai Smart Contracts và Cấp vốn
Mở Terminal 2:
```bash
source .venv/bin/activate
npm run deploy
npm run fund
```

### Bước 3.3: Kiểm tra tính sẵn sàng của hệ thống (Healthcheck)
```bash
npm run healthcheck
pytest tests/unit -v
```

---

## 4. Thực nghiệm và So sánh hệ thống

Hệ thống cung cấp hai phương pháp để phân tích đối chứng nhằm phục vụ mục đích nghiên cứu:

### 4.1. Chạy thực nghiệm đơn lẻ
Sử dụng script `run_experiment.py` cho hệ thống Gốc hoặc `run_experiment_csra.py` cho hệ thống cải tiến:
```bash
# Hệ thống Gốc
python experiments/run_experiment.py --dataset mnist --scenario K3 --config C --alpha 0.5

# Hệ thống Cải tiến (CSRA-Inspired)
python experiments/run_experiment_csra.py --dataset mnist --scenario K3 --config C --alpha 0.5 --with-freeriders
```

### 4.2. Chạy kịch bản tự động
Script `run_all.sh` hỗ trợ nhiều chế độ chạy để kiểm tra nhanh hoặc chạy ma trận đầy đủ trên VM.
```bash
# Xem lệnh, không chạy
bash experiments/run_all.sh --full --dry-run

# Smoke test nhanh, không cần blockchain
bash experiments/run_all.sh --smoke

# Chạy nhanh MNIST trước khi chạy full matrix
bash experiments/run_all.sh --quick --resume

# Chạy đầy đủ 70 runs, mặc định 50 rounds/run
bash experiments/run_all.sh --full --resume
```

---

## 5. Đặc điểm nổi bật của hệ thống CSRA (Cải tiến)

Hệ thống cải tiến tích hợp các cơ chế nâng cao dựa trên khung CSRA để đối phó với các Client tấn công:
*   **CSRA-DCD (Detection):** Thuật toán phát hiện bất thường dựa trên chuẩn L2 của update delta và MAD robust z-score. Các bản cập nhật bất thường sẽ bị loại khỏi aggregation và reward.
*   **CSRA-QMS (Quality Management):** Điểm chất lượng được tính toán kết hợp với hồ sơ trung thực lịch sử $p(H)$, giúp hệ thống ghi nhận đóng góp công bằng hơn.
*   **Bidding Mechanism:** Mỗi Client gửi kèm báo giá (Bid), cho phép máy chủ điều chỉnh chính sách phần thưởng theo chi phí thực tế.

---

## 6. Phân tích kết quả
Dữ liệu nhật ký được lưu trữ tại `results/logs/`. Bạn có thể trích xuất biểu đồ phân tích bằng lệnh:
```bash
python analyze_results.py --report
```
Kết quả biểu đồ sẽ hiển thị sự khác biệt về độ chính xác (Accuracy), tính công bằng (Fairness) và khả năng chống tấn công giữa các phiên bản.

---

## 7. Cấu trúc thư mục mã nguồn
*   `contracts/`: Chứa các Smart Contracts (Solidity) để quản lý phần thưởng và danh tiếng.
*   `fl/`: Chứa mã nguồn logic lõi (Client, Server, Blockchain Bridge, Metrics).
*   `experiments/`: Chứa các script thực hiện thực nghiệm và phân tích dữ liệu.
*   `results/`: Thư mục lưu trữ kết quả đầu ra (Logs, Plots).
*   `tests/`: Hệ thống các bài kiểm tra đơn vị (Unit Tests) và tích hợp.

---
**Tác giả:** Phan Hồng Đạt - 23520266
**Đơn vị:** [Tên trường/viện của bạn]
