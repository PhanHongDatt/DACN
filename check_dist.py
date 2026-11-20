import pandas as pd, glob, os
files = glob.glob('results/logs/*.csv')
files.sort(key=os.path.getmtime)
last_24 = files[-24:]
df = pd.read_csv(last_24[0])
r1 = df[df['round']==1]
for _, row in r1.iterrows():
    print(f'client {int(row.client_id):2d} | type={row.client_type:10s} | data_size={int(row.data_size):5d}')
