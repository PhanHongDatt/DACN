"""Comprehensive analysis script for all experiment CSVs."""
import sys, io, csv, os, glob, re, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import defaultdict

LOG_DIR = "results/logs"

FNAME_RE = re.compile(
    r"^(?P<dataset>fashion_mnist|cifar10|mnist)_(?P<scenario>K\d+)"
    r"(?:_da(?P<dirichlet>\d{3}))?_(?P<agg>fedavg|trimmed|csra_dcd)"
    r"_(?P<reward>equal|data|quality|csra)"
    r"_b(?P<beta>\d{2})g(?P<gamma>\d{2})d(?P<delta>\d{2})"
    r"_s(?P<seed>\d+)_(?P<attack>[a-z_]+?)_(?P<ts>\d{8}_\d{6})\.csv$"
)

ATTACK_TYPES = ['free_rider', 'stealth_free_rider', 'lazy', 'label_noise', 'sign_flip']
METHODS = ['fedavg+equal', 'fedavg+data', 'fedavg+quality', 'fedavg+csra', 'csra_dcd+equal', 'csra_dcd+csra']
METHOD_LABEL = {
    'fedavg+equal': 'M1: FedAvg+Equal',
    'fedavg+data': 'M2: FedAvg+Data',
    'fedavg+quality': 'M3: FedAvg+Quality',
    'fedavg+csra': 'M4: FedAvg+CSRA',
    'csra_dcd+equal': 'M5: CSRA-DCD+Equal',
    'csra_dcd+csra': 'M6: CSRA-DCD+CSRA',
}

runs = []
parse_fail = 0
for f in sorted(glob.glob(os.path.join(LOG_DIR, "*.csv"))):
    fname = os.path.basename(f)
    m = FNAME_RE.match(fname)
    if not m:
        parse_fail += 1
        continue
    attack = m['attack']
    if attack not in ATTACK_TYPES and attack != 'clean':
        parse_fail += 1
        continue
    with open(f, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        continue
    max_round = max(int(r['round']) for r in rows)
    client_rewards = defaultdict(float)
    client_types = {}
    for row in rows:
        cid = int(row['client_id'])
        client_rewards[cid] += float(row.get('reward_eth', 0) or 0)
        client_types[cid] = row.get('client_type', 'honest')
    total_rewards = list(client_rewards.values())
    if len(total_rewards) > 1 and sum(total_rewards) > 0:
        s = sum(total_rewards)
        ss = sum(x**2 for x in total_rewards)
        jain = s**2 / (len(total_rewards) * ss)
    else:
        jain = 0.0
    sorted_r = sorted(total_rewards)
    n = len(sorted_r)
    c = 0
    cumsum = []
    for v in sorted_r:
        c += v
        cumsum.append(c)
    if cumsum and cumsum[-1] > 0:
        gini = (2 * sum((i+1)*v for i, v in enumerate(sorted_r)) - (n+1)*cumsum[-1]) / (n * cumsum[-1])
    else:
        gini = 0.0
    accs = [float(r.get('global_accuracy', 0) or 0) for r in rows if r.get('global_accuracy')]
    final_acc = accs[-1] if accs else 0.0
    peak_acc = max(accs) if accs else 0.0
    malicious_total = sum(v for cid, v in client_rewards.items() if client_types.get(cid, 'honest') != 'honest')
    all_total = sum(client_rewards.values())
    leakage = malicious_total / all_total if all_total > 0 else 0.0
    honest_flagged = sum(1 for r in rows if r.get('is_anomaly', '0') == '1' and r.get('client_type', '') == 'honest')
    honest_total = sum(1 for r in rows if r.get('client_type', '') == 'honest')
    malicious_flagged = sum(1 for r in rows if r.get('is_anomaly', '0') == '1' and r.get('client_type', '') != 'honest')
    malicious_total_rows = sum(1 for r in rows if r.get('client_type', '') != 'honest')
    fpr = honest_flagged / honest_total if honest_total > 0 else 0.0
    tpr = malicious_flagged / malicious_total_rows if malicious_total_rows > 0 else 0.0
    reward_blocked = sum(1 for r in rows if r.get('reward_blocked', '0') == '1')
    runs.append({
        'file': fname, 'dataset': m['dataset'], 'scenario': m['scenario'],
        'dirichlet': m['dirichlet'] if m['dirichlet'] else '',
        'agg': m['agg'], 'reward': m['reward'],
        'beta': float(m['beta']) / 100, 'seed': m['seed'], 'attack': attack,
        'max_round': max_round, 'n_rows': len(rows),
        'final_acc': final_acc, 'peak_acc': peak_acc,
        'jain': jain, 'gini': gini, 'leakage': leakage,
        'fpr': fpr, 'tpr': tpr, 'reward_blocked': reward_blocked,
    })

print(f"Loaded {len(runs)} runs ({parse_fail} failed)")

# ======== OUTPUT ========
print("\n" + "="*80)
print("PHAN TICH TOAN DIEN KET QUA THUC NGHIEM")
print("="*80)

# 1. Clean
print("\n### 1. CLEAN RUNS ###\n")
print(f"{'Method':<30} {'FinalAcc':>10} {'PeakAcc':>10} {'Jain':>8} {'Gini':>8} {'Runs':>5}")
print("-"*75)
for method in METHODS:
    agg, rew = method.split('+')
    clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean']
    if not clean:
        continue
    print(f"{METHOD_LABEL[method]:<30} {statistics.mean([r['final_acc'] for r in clean]):>10.4f} {statistics.mean([r['peak_acc'] for r in clean]):>10.4f} {statistics.mean([r['jain'] for r in clean]):>8.4f} {statistics.mean([r['gini'] for r in clean]):>8.4f} {len(clean):>5}")

# 2. Per-scenario
print("\n### 2. CLEAN - THEO SCENARIO ###\n")
for scenario_label, filt in [
    ('K1 (IID)', lambda r: r['scenario'] == 'K1'),
    ('K2 (Weak Non-IID)', lambda r: r['scenario'] == 'K2'),
    ('K3 alpha=0.5', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '050'),
    ('K3 alpha=0.1', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '010'),
]:
    print(f"--- {scenario_label} ---")
    print(f"  {'Method':<30} {'Acc':>8} {'Jain':>8} {'Gini':>8}")
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean' and filt(r)]
        if not subset:
            continue
        print(f"  {METHOD_LABEL[method]:<30} {statistics.mean([r['final_acc'] for r in subset]):>8.4f} {statistics.mean([r['jain'] for r in subset]):>8.4f} {statistics.mean([r['gini'] for r in subset]):>8.4f}")
    print()

# 3. Attack leakage
print("\n### 3. REWARD LEAKAGE THEO ATTACK ###\n")
for atk in ATTACK_TYPES:
    print(f"--- {atk} ---")
    print(f"  {'Method':<30} {'Leakage':>10} {'FPR':>8} {'TPR':>8}")
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == atk]
        if not subset:
            continue
        print(f"  {METHOD_LABEL[method]:<30} {statistics.mean([r['leakage'] for r in subset]):>10.4f} {statistics.mean([r['fpr'] for r in subset]):>8.4f} {statistics.mean([r['tpr'] for r in subset]):>8.4f}")
    print()

# 4. Detection
print("\n### 4. CSRA-DCD DETECTION ###\n")
print(f"  {'Attack':<25} {'TPR':>8} {'FPR':>8} {'Leakage':>10}")
for atk in ATTACK_TYPES:
    subset = [r for r in runs if r['agg'] == 'csra_dcd' and r['attack'] == atk]
    if not subset:
        continue
    print(f"  {atk:<25} {statistics.mean([r['tpr'] for r in subset]):>8.4f} {statistics.mean([r['fpr'] for r in subset]):>8.4f} {statistics.mean([r['leakage'] for r in subset]):>10.4f}")

# 5. Per-dataset
print("\n### 5. THEO DATASET ###\n")
for ds in ['mnist', 'fashion_mnist']:
    print(f"--- {ds} ---")
    print(f"  {'Method':<30} {'CleanAcc':>10} {'AtkLeak':>10}")
    for method in METHODS:
        agg, rew = method.split('+')
        clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean' and r['dataset'] == ds]
        atk = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] != 'clean' and r['dataset'] == ds]
        ca = statistics.mean([r['final_acc'] for r in clean]) if clean else 0
        al = statistics.mean([r['leakage'] for r in atk]) if atk else 0
        print(f"  {METHOD_LABEL[method]:<30} {ca:>10.4f} {al:>10.4f}")
    print()

# 6. Summary table for report
print("\n### 6. BANG TONG HOP (M6 vs M1) ###\n")
for method in ['fedavg+equal', 'csra_dcd+csra']:
    agg, rew = method.split('+')
    clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean']
    atk = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] != 'clean']
    print(f"{METHOD_LABEL[method]}:")
    if clean:
        print(f"  Clean: Acc={statistics.mean([r['final_acc'] for r in clean]):.4f}, Jain={statistics.mean([r['jain'] for r in clean]):.4f}, Gini={statistics.mean([r['gini'] for r in clean]):.4f}")
    if atk:
        print(f"  Attack: Leakage={statistics.mean([r['leakage'] for r in atk]):.4f}, FPR={statistics.mean([r['fpr'] for r in atk]):.4f}, TPR={statistics.mean([r['tpr'] for r in atk]):.4f}")
    print()
