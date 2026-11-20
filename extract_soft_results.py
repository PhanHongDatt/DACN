import pandas as pd
import json

df = pd.read_csv('results/summary_metrics.csv')
attacks = ['lazy']

results = []
for idx, r in df[df['attack_type'] == 'lazy'].iterrows():
    results.append({
        'dataset': r['dataset'],
        'scenario': r['scenario'],
        'method': r['method'],
        'leakage': r.get('reward_leakage', 0),
        'reward_blocked': r.get('attack_reward_block_rate', 0),
    })

print(json.dumps(results[:10], indent=2))
