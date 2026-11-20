import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

attacks = {
    'free_rider': ('FedAvg (Free Rider)', 'blue', 'x', '--'),
    'stealth_free_rider': ('FedAvg (Stealth Free Rider)', 'cyan', 's', '--'),
    'lazy': ('FedAvg (Lazy)', 'orange', '^', '--'),
    'label_noise': ('FedAvg (Label Noise)', 'red', 'v', '--'),
    'sign_flip': ('FedAvg (Sign Flip)', 'purple', 'd', '--')
}

# Load clean baseline
clean_files = glob.glob('results/logs/mnist_K2_fedavg_equal_*_clean_*.csv')
if not clean_files:
    print("Khong tim thay file clean!")
    exit(1)
clean_files.sort(key=os.path.getmtime)
df_clean = pd.read_csv(clean_files[-1])
acc_clean = df_clean.groupby('round')['global_accuracy'].first()

os.makedirs('results/plots', exist_ok=True)

# Generate individual plots
for attack, (label_name, color, marker, ls) in attacks.items():
    pattern = f'results/logs/mnist_K2_fedavg_equal_*_{attack}_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f"Khong tim thay log cho kich ban: {attack}")
        continue
    
    files.sort(key=os.path.getmtime)
    df_attack = pd.read_csv(files[-1])
    acc_attack = df_attack.groupby('round')['global_accuracy'].first()
    
    plt.figure(figsize=(8, 5))
    plt.plot(acc_clean.index, acc_clean.values, label='FedAvg (Thuần/Sạch)', color='green', marker='o', linewidth=2, markersize=6)
    plt.plot(acc_attack.index, acc_attack.values, label=label_name, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=6)
    
    plt.title(f'Độ chính xác: FedAvg Sạch vs {attack.replace("_", " ").title()}', fontsize=14, pad=15)
    plt.xlabel('Vòng (Round)', fontsize=12)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    
    out_path = f'results/plots/fedavg_clean_vs_{attack}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Da luu: {out_path}")

print("Hoan thanh ve cac bieu do rieng le!")
