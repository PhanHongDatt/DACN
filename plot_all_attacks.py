import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

attacks = {
    'clean': ('FedAvg (Thuần/Sạch)', 'green', 'o', '-'),
    'free_rider': ('FedAvg (Free Rider)', 'blue', 'x', '--'),
    'stealth_free_rider': ('FedAvg (Stealth Free Rider)', 'cyan', 's', '--'),
    'lazy': ('FedAvg (Lazy)', 'orange', '^', '--'),
    'label_noise': ('FedAvg (Label Noise)', 'red', 'v', '--'),
    'sign_flip': ('FedAvg (Sign Flip)', 'purple', 'd', '--')
}

plt.figure(figsize=(12, 7))

for attack, (label_name, color, marker, ls) in attacks.items():
    pattern = f'results/logs/mnist_K2_fedavg_equal_*_{attack}_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f"Khong tim thay log cho kich ban: {attack}")
        continue
    
    # Lấy file mới nhất
    files.sort(key=os.path.getmtime)
    file_path = files[-1]
    
    df = pd.read_csv(file_path)
    acc = df.groupby('round')['global_accuracy'].first()
    
    plt.plot(acc.index, acc.values, label=label_name, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=6)

plt.title('So sánh Độ chính xác toàn cục (Global Accuracy): FedAvg Sạch vs Các loại Tấn công', fontsize=15, pad=15)
plt.xlabel('Vòng (Round)', fontsize=13)
plt.ylabel('Độ chính xác (Accuracy)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11, loc='lower right')
plt.tight_layout()

out_path = 'results/plots/fedavg_clean_vs_all_attacks.png'
os.makedirs('results/plots', exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"Da luu bieu do tai: {out_path}")
