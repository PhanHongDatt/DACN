import pandas as pd
import glob
import os
import numpy as np

files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]

def compute_jain(rewards):
    r = np.array(rewards, dtype=float)
    if len(r)==0 or np.sum(r)==0: return 0.0
    return (np.sum(r)**2)/(len(r)*np.sum(r**2))

def compute_gini(rewards):
    r = np.array(sorted(rewards), dtype=float)
    n = len(r)
    if n==0 or np.sum(r)==0: return 0.0
    index = np.arange(1, n+1)
    return (2*np.sum(index*r)/(n*np.sum(r))) - (n+1)/n

for f in last_24:
    bn = os.path.basename(f)
    ds = "fashion_mnist" if "fashion_mnist" in bn else "mnist"
    method = "CSRA" if "csra_dcd" in bn else "FedAvg"
    attack = "clean"
    for a in ["free_rider","stealth_free_rider","lazy","label_noise","sign_flip"]:
        if a in bn:
            attack = a
            break
    try:
        df = pd.read_csv(f)
        last_r = df['round'].max()
        df_l = df[df['round']==last_r]
        acc = df_l['global_accuracy'].iloc[0]
        honest = df_l[df_l['client_type']=='honest']
        mal = df_l[df_l['client_type']!='honest']
        
        total_rew = df_l['reward'].sum() if 'reward' in df_l.columns else 0
        mal_rew = mal['reward'].sum() if not mal.empty and 'reward' in mal.columns else 0
        leakage = (mal_rew/total_rew*100) if total_rew>0 else 0
        
        jain = compute_jain(honest['reward'].values) if not honest.empty and 'reward' in honest.columns else 0
        
        # Reputation
        hon_rep = honest['reputation'].mean() if not honest.empty and 'reputation' in honest.columns else 0
        mal_rep = mal['reputation'].mean() if not mal.empty and 'reputation' in mal.columns else 0
        
        # TPR/FPR
        tpr, fpr = 0, 0
        if 'is_valid' in df_l.columns:
            if not mal.empty: tpr = 100*(mal['is_valid']==False).sum()/len(mal)
            if not honest.empty: fpr = 100*(honest['is_valid']==False).sum()/len(honest)
        
        # Data sizes
        n_honest = len(honest)
        n_mal = len(mal)
        
        # Blockchain tx info
        has_tx = 'tx_hash' in df_l.columns
        
        print(f"{ds:15} | {method:6} | {attack:20} | acc={acc:.4f} | leak={leakage:5.1f}% | jain={jain:.4f} | tpr={tpr:5.1f} | fpr={fpr:5.1f} | hon_rep={hon_rep:.2f} | mal_rep={mal_rep:.2f} | n_h={n_honest} n_m={n_mal} | rows={len(df)} | rounds={last_r}")
    except Exception as e:
        print(f"ERR {bn}: {e}")
