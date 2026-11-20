import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find clean log
clean_files = glob.glob('results/logs/mnist_K2_fedavg_equal_*_clean_*.csv')
# Find label noise log
noise_files = glob.glob('results/logs/mnist_K2_fedavg_equal_*_label_noise_*.csv')

if not clean_files or not noise_files:
    print("Khong tim thay du file log de ve bieu do!")
    exit(1)

clean_file = clean_files[-1]
noise_file = noise_files[-1]

df_clean = pd.read_csv(clean_file)
df_noise = pd.read_csv(noise_file)

# Extract global accuracy per round
# The CSV has multiple rows per round (one for each client). We can group by round and take the first accuracy.
acc_clean = df_clean.groupby('round')['global_accuracy'].first()
acc_noise = df_noise.groupby('round')['global_accuracy'].first()

plt.figure(figsize=(10, 6))
plt.plot(acc_clean.index, acc_clean.values, label='FedAvg (Thuần/Sạch)', marker='o', linewidth=2, color='green')
plt.plot(acc_noise.index, acc_noise.values, label='FedAvg (Bị lật nhãn - Label Noise)', marker='x', linewidth=2, color='red', linestyle='--')

plt.title('So sánh Độ chính xác toàn cục (Global Accuracy): FedAvg Sạch vs FedAvg Lật Nhãn', fontsize=14, pad=15)
plt.xlabel('Vòng (Round)', fontsize=12)
plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

out_path = 'results/plots/fedavg_clean_vs_labelnoise.png'
os.makedirs('results/plots', exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"Da luu bieu do tai: {out_path}")
