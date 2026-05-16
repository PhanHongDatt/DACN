# Hướng Dẫn Chạy Thực Nghiệm Trên Máy Ảo Ubuntu

Tài liệu này hướng dẫn từng bước chạy full experiment matrix trên VM Ubuntu 24/7. Nếu là lần đầu cài đặt, làm theo thứ tự từ §1 đến §9. Nếu code đã có trên VM rồi và chỉ muốn chạy tiếp, nhảy thẳng §6.

> **Branch dùng:** `refactor/reward-policies` (schema v2)
> **Tham chiếu kế hoạch:** [`docs/PLAN.md`](PLAN.md)
> **Ước lượng compute:** ~1.5-2 ngày 24/7 với `PARALLEL=4` (VM 8 core).

---

## 1. Yêu Cầu Hệ Thống

| Mục | Khuyến nghị | Tối thiểu |
|---|---|---|
| OS | Ubuntu 22.04 LTS | Ubuntu 20.04 |
| Python | 3.11 | 3.9+ |
| Node.js (chỉ nếu dùng blockchain) | 20.x | 18.x |
| CPU | 8 cores | 4 cores |
| RAM | 16 GB | 8 GB |
| Disk | 15 GB trống | 8 GB |

Kiểm tra nhanh:
```bash
nproc                     # số CPU cores
free -h                   # RAM
df -h ~                   # disk còn lại
python3 --version
```

---

## 2. Đưa Code Lên VM

### Phương án A — Git clone (khuyến nghị)

Trên Windows host, push branch refactor lên remote trước:

```bash
# Trên Windows (D:\DACN)
git push -u origin refactor/reward-policies
```

Trên VM:
```bash
cd ~
git clone <your-github-url> DACN
cd DACN
git checkout refactor/reward-policies
git log --oneline -5    # verify đúng commit cuối
```

### Phương án B — SCP từ Windows

```powershell
# Trên Windows PowerShell
scp -r D:\DACN user@vm-ip:~/DACN
```

Loại trừ các thư mục lớn không cần (cache, data, node_modules):
```powershell
robocopy D:\DACN \\temp\DACN /E /XD node_modules cache data results\logs_legacy .git\objects
scp -r \temp\DACN user@vm-ip:~/DACN
```

### Phương án C — Rsync (nếu có)

```bash
# Trên Windows (Git Bash) hoặc WSL
rsync -avz --exclude='node_modules' --exclude='cache' --exclude='data' \
      --exclude='results/logs_legacy' --exclude='.venv' \
      /d/DACN/ user@vm-ip:~/DACN/
```

---

## 3. Cài Python + Dependencies

```bash
cd ~/DACN

# Cài Python 3.11 + venv
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential

# Tạo virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Cài packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Lưu ý:** Pipeline mới dùng `simulation_local.py`, **không cần Ray**. requirements.txt có thể có `flwr[simulation]` — không sao, vẫn cài được.

Verify môi trường:
```bash
python scripts/check_python.py
```

Output cần thấy `All checks passed. Ready.` Nếu lỗi `Missing: xxx`, cài thêm package đó.

---

## 4. (Tùy Chọn) Cài Hardhat Cho Blockchain Audit

Chỉ làm bước này nếu định chạy với `--with-blockchain` (demo audit layer). Cho full experiment matrix thì **bỏ qua**.

```bash
# Cài Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version           # phải v20.x

# Cài Hardhat + dependencies
cd ~/DACN
npm install
npm run compile          # compile smart contracts
```

---

## 5. Smoke Test (5-10 Phút)

Chạy thử 6 cells × 3 rounds × 5 clients để verify pipeline hoạt động:

```bash
source .venv/bin/activate   # nếu chưa
PARALLEL=2 NUM_THREADS=2 \
  bash experiments/run_all.sh --smoke
```

Kỳ vọng thấy:
```
[OK]   smoke[seed=42]/M1 FedAvg+EqualSplit
[OK]   smoke[seed=42]/M2 FedAvg+DataSize
...
[OK]   smoke[seed=42]/M6 CSRA-DCD+CSRAReward
```

Kiểm tra CSVs đã sinh:
```bash
ls -lh results/logs/*.csv | tail -10
```

---

## 6. Quick Test (45 Phút) — Khuyến Nghị Chạy Trước Full

Quick test chạy 12 runs (6 cells × 2 scenarios K1+K3@0.1) để confirm setup ổn định trước full matrix:

```bash
PARALLEL=4 NUM_THREADS=2 \
  bash experiments/run_all.sh --quick
```

Sau khi xong, verify:
```bash
ls results/logs/*.csv | wc -l         # phải = 12
python -c "from analysis.loader import load_all_logs; from pathlib import Path; df = load_all_logs(Path('results/logs')); print(df.shape); print(df.groupby('method')['global_accuracy'].last())"
```

---

## 7. Full Matrix (~1.5-2 Ngày 24/7)

### Bước 7.1: Khởi động tmux (giữ session khi disconnect SSH)

```bash
# Cài tmux nếu chưa có
sudo apt install -y tmux

# Tạo session
tmux new -s fl-exp
```

### Bước 7.2: Chạy full matrix

Trong session tmux:
```bash
cd ~/DACN
source .venv/bin/activate

PARALLEL=4 NUM_THREADS=2 \
  bash experiments/run_all.sh --full --no-cifar --resume \
  2>&1 | tee results/full_run.log
```

Giải thích flags:
- `PARALLEL=4`: 4 cells chạy đồng thời (cho VM 8 core)
- `NUM_THREADS=2`: mỗi cell dùng 2 thread → tổng 8 = full 8 core
- `--full`: full ablation matrix
- `--no-cifar`: bỏ CIFAR-10 (chạy riêng trên Windows GPU sau nếu cần)
- `--resume`: tự skip cells đã có CSV (an toàn nếu phải interrupt)
- `2>&1 | tee`: lưu stdout+stderr vào file để xem lại

### Bước 7.3: Detach để không bị mất khi SSH disconnect

Trong tmux: nhấn **`Ctrl+B`** rồi **`D`** (detach). Session vẫn chạy ở background.

### Bước 7.4: Re-attach sau

```bash
tmux attach -t fl-exp
```

Hoặc xem trạng thái:
```bash
tmux ls
```

---

## 8. Theo Dõi Tiến Độ

Mở terminal khác (hoặc tmux pane khác — `Ctrl+B "` chia pane):

### 8.1. Đếm runs đã xong

```bash
watch -n 60 'ls ~/DACN/results/logs/*.csv 2>/dev/null | wc -l'
# Hiện thực-time, refresh mỗi 60s. Mục tiêu: 354 cho full --no-cifar
```

### 8.2. Xem progress của run đang chạy

```bash
cd ~/DACN
tail -f $(ls -t results/logs/*.log | head -1)
```

### 8.3. Xem run gần nhất accuracy

```bash
cd ~/DACN
for f in $(ls -t results/logs/*.log | head -5); do
  last=$(grep "global accuracy" "$f" | tail -1)
  echo "$(basename "$f"): $last"
done
```

### 8.4. Tính thời gian còn lại

```bash
cd ~/DACN
done=$(ls results/logs/*.csv | wc -l)
total=354    # full --no-cifar
echo "Done: $done / $total ($((done*100/total))%)"
```

---

## 9. Xử Lý Sự Cố

### 9.1. Hết RAM / OOM Killer

VM bị kill process? Giảm parallelism:
```bash
PARALLEL=2 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume
```

Hoặc giảm batch size:
```bash
# Edit run_all.sh, hoặc:
BATCH_SIZE=16 PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume
```

### 9.2. Hết disk

```bash
# Check
df -h ~

# Clean cache
rm -rf ~/DACN/cache/
rm -rf ~/DACN/__pycache__ ~/DACN/**/__pycache__

# Backup logs cũ rồi xoá khỏi VM
tar czf ~/logs_backup_$(date +%Y%m%d).tar.gz ~/DACN/results/logs/
```

### 9.3. Bị interrupt giữa chừng

Resume sẽ tự skip CSV đã có:
```bash
PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume
```

Nếu CSV bị corrupt (interrupt giữa write), xoá file đó trước:
```bash
# Tìm CSV file size bất thường (< 1KB là nghi ngờ)
find results/logs -name "*.csv" -size -1k -ls
# Xoá file nghi ngờ
find results/logs -name "*.csv" -size -1k -delete
```

### 9.4. Process bị treo, không tiến triển

```bash
# Kiểm tra process
ps aux | grep run_experiment

# Kill toàn bộ
pkill -f run_experiment

# Restart
tmux attach -t fl-exp
# Nhấn Ctrl+C nếu cần, rồi gõ:
PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume
```

### 9.5. Lỗi `ModuleNotFoundError`

Quên activate venv:
```bash
cd ~/DACN
source .venv/bin/activate
which python    # phải trỏ tới ~/DACN/.venv/bin/python
```

### 9.6. CSV không parse được trong analysis

Filename phải đúng schema v2. Kiểm tra:
```bash
ls results/logs/*.csv | head -3
# Phải match: <ds>_<sc>[_da<n>]_<agg>_<reward>_b<n>g<n>d<n>_s<n>_<atk>_<ts>.csv
# Ví dụ:  mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_143022.csv
```

Nếu logs cũ schema v1 còn sót:
```bash
# Move qua thư mục khác để không lẫn
mkdir -p results/logs_v1_old
mv results/logs/*_a*_*.csv results/logs_v1_old/ 2>/dev/null
```

---

## 10. Sau Khi Chạy Xong

### 10.1. Backup logs ra ngoài VM

```bash
# Trên Windows
scp -r user@vm-ip:~/DACN/results/logs ./results_from_vm
```

Hoặc nén trước rồi scp:
```bash
# Trên VM
cd ~/DACN
tar czf ~/full_results.tar.gz results/logs/
# Trên Windows
scp user@vm-ip:~/full_results.tar.gz D:\DACN\
```

### 10.2. Chạy analysis pipeline

Trên VM hoặc Windows:
```bash
python analyze_results.py --report --latex
```

(Lưu ý: analysis pipeline chưa được refactor cho schema v2 — sẽ làm ở W5).

### 10.3. (Tùy chọn) Chạy CIFAR-10 trên Windows GPU

Trên Windows host (có GPU + CUDA):
```bash
# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Chạy CIFAR-10 clean matrix
ROUNDS=30 LOCAL_EPOCHS=1 PARALLEL=1 \
  bash experiments/run_all.sh --full --cifar-only --resume
```

(`--cifar-only` chưa support — sẽ thêm sau nếu cần. Tạm thời chạy thủ công per cell.)

---

## 11. Cheat Sheet — Lệnh Hay Dùng

```bash
# Activate
source ~/DACN/.venv/bin/activate

# Smoke test
PARALLEL=2 NUM_THREADS=2 bash experiments/run_all.sh --smoke

# Quick test (45 min)
PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --quick

# Full matrix (~2 days)
tmux new -s fl-exp
PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume \
  2>&1 | tee results/full_run.log
# Ctrl+B D để detach

# Watch progress
tmux attach -t fl-exp
# Hoặc:
watch -n 60 'ls ~/DACN/results/logs/*.csv | wc -l'

# Dry-run (xem commands không chạy)
bash experiments/run_all.sh --full --no-cifar --dry-run | head -20

# Resume sau interrupt
PARALLEL=4 NUM_THREADS=2 bash experiments/run_all.sh --full --no-cifar --resume

# Test pipeline có blockchain (cần Hardhat)
bash scripts/run_pipeline.sh --smoke --with-blockchain
```

---

## 12. Liên Hệ Khi Có Vấn Đề

Nếu gặp lỗi không xử lý được:
1. Save log đầy đủ: `cp results/logs/*.log /tmp/`
2. Snapshot trạng thái: `git status; git log --oneline -3`
3. Liên hệ kèm:
   - Output của `python scripts/check_python.py`
   - Lỗi cụ thể từ log
   - Lệnh đã chạy

---

**Chúc bạn chạy thực nghiệm trơn tru!** 🚀
