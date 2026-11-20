import pandas as pd
import glob
import os
import numpy as np

files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]

def compute_jain_index(rewards):
    rewards = np.array(rewards)
    if len(rewards) == 0 or np.sum(rewards) == 0:
        return 0.0
    return (np.sum(rewards)**2) / (len(rewards) * np.sum(rewards**2))

results = []

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
        if df.empty:
            continue
            
        last_round = df['round'].max()
        df_last = df[df['round'] == last_round]
        
        final_acc = df_last['global_accuracy'].iloc[0] if 'global_accuracy' in df_last.columns else 0.0
        
        honest_df = df_last[df_last['client_type'] == 'honest']
        malicious_df = df_last[df_last['client_type'] != 'honest']
        
        jain_honest = compute_jain_index(honest_df['reward'].values) if not honest_df.empty and 'reward' in honest_df.columns else 0.0
        
        tpr = 0.0
        fpr = 0.0
        # If method is FedAvg, there's no is_valid, everything is valid (True)
        if 'is_valid' in df_last.columns:
            if not malicious_df.empty:
                tpr = 100.0 * (malicious_df['is_valid'] == False).sum() / len(malicious_df)
            if not honest_df.empty:
                fpr = 100.0 * (honest_df['is_valid'] == False).sum() / len(honest_df)
                
        malicious_reward = malicious_df['reward'].sum() if not malicious_df.empty and 'reward' in malicious_df.columns else 0.0
        
        results.append({
            'Dataset': dataset,
            'Attack': attack,
            'Method': method,
            'Accuracy': final_acc,
            'Jain_Honest': jain_honest,
            'TPR': tpr,
            'FPR': fpr,
            'Mal_Reward': malicious_reward
        })
    except Exception as e:
        print(f"Lỗi đọc {basename}: {e}")

df_res = pd.DataFrame(results)

# Manual markdown table string builder
markdown = "| Dataset | Attack | Method | Final Acc | Jain (Honest) | TPR (%) | FPR (%) | Malicious Reward |\n"
markdown += "|---------|--------|--------|-----------|---------------|---------|---------|------------------|\n"

for index, row in df_res.iterrows():
    markdown += f"| {row['Dataset']} | {row['Attack']} | {row['Method']} | {row['Accuracy']:.4f} | {row['Jain_Honest']:.4f} | {row['TPR']:.1f} | {row['FPR']:.1f} | {row['Mal_Reward']:.4f} |\n"

with open('results/csv/analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown)

print("Tao report thanh cong tai results/csv/analysis_report.md")
