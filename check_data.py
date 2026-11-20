import pandas as pd
import glob
import os
from datetime import datetime, date

# Lấy các file CSV sinh ra trong ngày hôm nay
files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)

today_files = [f for f in files if datetime.fromtimestamp(os.path.getmtime(f)).date() == date.today()]

if not today_files:
    print("Không tìm thấy file nào sinh ra trong hôm nay!")
    exit(0)

print(f"Tổng số file sinh ra hôm nay: {len(today_files)}")
print("-" * 80)
print(f"{'Dataset':<15} | {'Attack':<18} | {'Method':<12} | {'Rows':<5} | {'Final Acc':<10} | {'Blocked? (Reward=0)':<20}")
print("-" * 80)

for f in today_files:
    # Phân tích tên file: vd: fashion_mnist_K2_csra_dcd_csra_b50g30d20_s42_sign_flip_20260620_012616.csv
    basename = os.path.basename(f)
    parts = basename.split('_')
    
    # Heuristic để lấy thông tin
    if "fashion_mnist" in basename:
        dataset = "fashion_mnist"
    else:
        dataset = "mnist"
        
    if "csra_dcd" in basename:
        method = "CSRA"
    else:
        method = "FedAvg"
        
    attack = "clean"
    for a in ["free_rider", "stealth_free_rider", "lazy", "label_noise", "sign_flip"]:
        if a in basename:
            attack = a
            break
            
    try:
        df = pd.read_csv(f)
        rows = len(df)
        final_acc = df['global_accuracy'].iloc[-1] if not df.empty and 'global_accuracy' in df.columns else "N/A"
        
        # Kiểm tra xem có block client độc hại không (ở round cuối)
        if 'reward' in df.columns and 'client_type' in df.columns:
            last_round = df['round'].max()
            df_last = df[df['round'] == last_round]
            malicious = df_last[df_last['client_type'] != 'honest']
            if not malicious.empty:
                reward_blocked = (malicious['reward'] == 0).all()
                blocked_status = "Yes" if reward_blocked else "No (Rewarded!)"
            else:
                blocked_status = "N/A (All Honest)"
        else:
            blocked_status = "Unknown"
            
        print(f"{dataset:<15} | {attack:<18} | {method:<12} | {rows:<5} | {final_acc:<10.4f} | {blocked_status:<20}")
    except Exception as e:
        print(f"Error reading {basename}: {e}")

