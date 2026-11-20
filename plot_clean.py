import pandas as pd
import matplotlib.pyplot as plt
import glob, os

files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]

data_dict = {}
for f in last_24:
    bn = os.path.basename(f)
    ds = 'fashion_mnist' if 'fashion_mnist' in bn else 'mnist'
    method = 'CSRA' if 'csra_dcd' in bn else 'FedAvg'
    attack = 'clean'
    for a in ['free_rider','stealth_free_rider','lazy','label_noise','sign_flip']:
        if a in bn:
            attack = a
            break
    df = pd.read_csv(f)
    if not df.empty:
        data_dict[(ds, method, attack)] = df

for ds in ['mnist', 'fashion_mnist']:
    df_fedavg = data_dict.get((ds, 'FedAvg', 'clean'))
    df_csra = data_dict.get((ds, 'CSRA', 'clean'))
    
    plt.figure(figsize=(8, 5))
    if df_fedavg is not None:
        acc = df_fedavg.groupby('round')['global_accuracy'].first()
        plt.plot(acc.index, acc.values, label='FedAvg (Clean)', color='green', marker='o', linewidth=2)
    if df_csra is not None:
        acc = df_csra.groupby('round')['global_accuracy'].first()
        plt.plot(acc.index, acc.values, label='CSRA (Clean)', color='blue', marker='s', linewidth=2)
    
    plt.title(f'Accuracy: FedAvg vs CSRA - Clean Baseline ({ds})', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'report/img1/acc_{ds}_clean.png', dpi=300)
    plt.close()
    print(f'Saved acc_{ds}_clean.png')

print('Done')
