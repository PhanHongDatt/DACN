import pandas as pd
import glob
import os
import numpy as np

files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]

def jain(r):
    r=np.array(r,dtype=float)
    if len(r)==0 or np.sum(r)==0: return 0.0
    return (np.sum(r)**2)/(len(r)*np.sum(r**2))

print("="*160)
print(f"{'Dataset':15} | {'Method':6} | {'Attack':20} | {'Acc':>7} | {'Leak%':>6} | {'Jain_H':>6} | {'TPR%':>6} | {'FPR%':>6} | {'H_Rep':>6} | {'M_Rep':>6} | {'H_Rew':>7} | {'M_Rew':>7} | {'Blk%':>5}")
print("="*160)

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

        # Reward columns: reward_eth, reward_blocked
        h_rew = honest['reward_eth'].sum() if 'reward_eth' in honest.columns else 0
        m_rew = mal['reward_eth'].sum() if 'reward_eth' in mal.columns and not mal.empty else 0
        total = h_rew + m_rew
        leak = (m_rew/total*100) if total>0 else 0
        j = jain(honest['reward_eth'].values) if 'reward_eth' in honest.columns else 0

        # Detection: is_anomaly
        tpr, fpr = 0, 0
        if 'is_anomaly' in df_l.columns:
            if not mal.empty: tpr = 100*(mal['is_anomaly']==1).sum()/len(mal)
            if not honest.empty: fpr = 100*(honest['is_anomaly']==1).sum()/len(honest)

        # Reputation
        h_rep = honest['reputation'].mean() if 'reputation' in honest.columns else 0
        m_rep = mal['reputation'].mean() if 'reputation' in mal.columns and not mal.empty else 0

        # reward_blocked rate for malicious
        blk = 0
        if 'reward_blocked' in df_l.columns and not mal.empty:
            blk = 100*(mal['reward_blocked']==1).sum()/len(mal)

        print(f"{ds:15} | {method:6} | {attack:20} | {acc:7.4f} | {leak:5.1f}% | {j:6.4f} | {tpr:5.1f}% | {fpr:5.1f}% | {h_rep:6.2f} | {m_rep:6.2f} | {h_rew:7.4f} | {m_rew:7.4f} | {blk:4.0f}%")
    except Exception as e:
        print(f"ERR {bn}: {e}")
