import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Tạo thư mục
os.makedirs('report/img1', exist_ok=True)

# Lấy 24 file mới nhất
files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]

# Phân tích filename và nạp vào dict
data_dict = {}
for f in last_24:
    basename = os.path.basename(f)
    dataset = "fashion_mnist" if "fashion_mnist" in basename else "mnist"
    method = "CSRA" if "csra_dcd" in basename else "FedAvg"
    attack = "clean"
    for a in ["free_rider", "stealth_free_rider", "lazy", "label_noise", "sign_flip"]:
        if a in basename:
            attack = a
            break
    try:
        df = pd.read_csv(f)
        if df.empty: continue
        data_dict[(dataset, method, attack)] = df
    except Exception as e:
        print(f"Lỗi: {e}")

attacks = ["free_rider", "stealth_free_rider", "lazy", "label_noise", "sign_flip"]
datasets = ["mnist", "fashion_mnist"]

# 1. Vẽ 10 Accuracy Plots & 10 Reputation Plots
reward_leakage = {ds: {'FedAvg': [], 'CSRA': []} for ds in datasets}

for ds in datasets:
    # Lấy FedAvg Sạch
    df_clean = data_dict.get((ds, "FedAvg", "clean"))
    if df_clean is not None:
        acc_clean = df_clean.groupby('round')['global_accuracy'].first()
        
    for atk in attacks:
        df_fedavg = data_dict.get((ds, "FedAvg", atk))
        df_csra = data_dict.get((ds, "CSRA", atk))
        
        # === 1.1 Biểu đồ Accuracy ===
        plt.figure(figsize=(8, 5))
        if df_clean is not None:
            plt.plot(acc_clean.index, acc_clean.values, label='FedAvg (Clean Baseline)', color='green', marker='o', linewidth=2)
            
        if df_fedavg is not None:
            acc_fedavg = df_fedavg.groupby('round')['global_accuracy'].first()
            plt.plot(acc_fedavg.index, acc_fedavg.values, label=f'FedAvg ({atk})', color='red', marker='x', linestyle='--', linewidth=2)
            
        if df_csra is not None:
            acc_csra = df_csra.groupby('round')['global_accuracy'].first()
            plt.plot(acc_csra.index, acc_csra.values, label=f'CSRA ({atk})', color='blue', marker='s', linewidth=2)
            
        plt.title(f'Accuracy: FedAvg vs CSRA under {atk.title()} ({ds})', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'report/img1/acc_{ds}_{atk}.png', dpi=300)
        plt.close()
        
        # === 1.2 Biểu đồ Reputation (CSRA) ===
        if df_csra is not None and 'reputation' in df_csra.columns and 'client_type' in df_csra.columns:
            plt.figure(figsize=(8, 5))
            honest_rep = df_csra[df_csra['client_type'] == 'honest'].groupby('round')['reputation'].mean()
            malicious_rep = df_csra[df_csra['client_type'] != 'honest'].groupby('round')['reputation'].mean()
            
            plt.plot(honest_rep.index, honest_rep.values, label='Honest (Mean Rep)', color='green', marker='o', linewidth=2)
            if not malicious_rep.empty:
                plt.plot(malicious_rep.index, malicious_rep.values, label='Malicious (Mean Rep)', color='red', marker='x', linestyle='--', linewidth=2)
            
            plt.title(f'Reputation Dynamics: CSRA under {atk.title()} ({ds})', fontsize=14)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Reputation Score', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'report/img1/rep_{ds}_{atk}.png', dpi=300)
            plt.close()
            
        # === 1.3 Tính toán Reward Leakage ===
        # Reward leakage = sum(reward of malicious) / sum(reward of all) * 100
        for m_name, df_m in [("FedAvg", df_fedavg), ("CSRA", df_csra)]:
            leakage = 0.0
            if df_m is not None and 'reward' in df_m.columns and 'client_type' in df_m.columns:
                total_reward = df_m['reward'].sum()
                malicious_reward = df_m[df_m['client_type'] != 'honest']['reward'].sum()
                if total_reward > 0:
                    leakage = (malicious_reward / total_reward) * 100.0
            reward_leakage[ds][m_name].append(leakage)

# 2. Vẽ 2 biểu đồ Reward Leakage (Grouped Bar Chart)
x = np.arange(len(attacks))
width = 0.35

for ds in datasets:
    plt.figure(figsize=(10, 6))
    bars_fedavg = plt.bar(x - width/2, reward_leakage[ds]['FedAvg'], width, label='FedAvg', color='red', alpha=0.7)
    bars_csra = plt.bar(x + width/2, reward_leakage[ds]['CSRA'], width, label='CSRA', color='blue', alpha=0.7)
    
    plt.ylabel('Reward Leakage (%)', fontsize=12)
    plt.title(f'Reward Leakage Comparison ({ds})', fontsize=14)
    plt.xticks(x, [a.replace('_', '\n').title() for a in attacks], fontsize=11)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gắn label giá trị lên bar
    for bars in [bars_fedavg, bars_csra]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center', fontsize=10)
            
    plt.tight_layout()
    plt.savefig(f'report/img1/leakage_{ds}.png', dpi=300)
    plt.close()

print("Hoan thanh ve toan bo 22 bieu do vao d:/DACN/report/img1")
