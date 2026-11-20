"""Phân tích điểm nổi bật: chống client độc hại & phân phối phần thưởng."""
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
METHOD_SHORT = {
    'fedavg+equal': 'M1', 'fedavg+data': 'M2', 'fedavg+quality': 'M3',
    'fedavg+csra': 'M4', 'csra_dcd+equal': 'M5', 'csra_dcd+csra': 'M6',
}

runs = []
for f in sorted(glob.glob(os.path.join(LOG_DIR, "*.csv"))):
    fname = os.path.basename(f)
    m = FNAME_RE.match(fname)
    if not m:
        continue
    attack = m['attack']
    if attack not in ATTACK_TYPES and attack != 'clean':
        continue
    with open(f, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        continue

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

    honest_rewards = [v for cid, v in client_rewards.items() if client_types.get(cid) == 'honest']
    malicious_rewards = [v for cid, v in client_rewards.items() if client_types.get(cid) != 'honest']
    honest_total = sum(honest_rewards)
    malicious_total = sum(malicious_rewards)
    all_total = honest_total + malicious_total
    leakage = malicious_total / all_total if all_total > 0 else 0.0

    accs = [float(r.get('global_accuracy', 0) or 0) for r in rows if r.get('global_accuracy')]
    final_acc = accs[-1] if accs else 0.0

    honest_flagged = sum(1 for r in rows if r.get('is_anomaly', '0') == '1' and r.get('client_type', '') == 'honest')
    honest_total_rows = sum(1 for r in rows if r.get('client_type', '') == 'honest')
    malicious_flagged = sum(1 for r in rows if r.get('is_anomaly', '0') == '1' and r.get('client_type', '') != 'honest')
    malicious_total_rows = sum(1 for r in rows if r.get('client_type', '') != 'honest')
    fpr = honest_flagged / honest_total_rows if honest_total_rows > 0 else 0.0
    tpr = malicious_flagged / malicious_total_rows if malicious_total_rows > 0 else 0.0

    reward_blocked_honest = sum(1 for r in rows if r.get('reward_blocked', '0') == '1' and r.get('client_type', '') == 'honest')
    reward_blocked_malicious = sum(1 for r in rows if r.get('reward_blocked', '0') == '1' and r.get('client_type', '') != 'honest')

    runs.append({
        'dataset': m['dataset'], 'scenario': m['scenario'],
        'dirichlet': m['dirichlet'] if m['dirichlet'] else '',
        'agg': m['agg'], 'reward': m['reward'],
        'seed': m['seed'], 'attack': attack,
        'final_acc': final_acc, 'jain': jain, 'leakage': leakage,
        'fpr': fpr, 'tpr': tpr,
        'honest_avg': statistics.mean(honest_rewards) if honest_rewards else 0,
        'malicious_avg': statistics.mean(malicious_rewards) if malicious_rewards else 0,
        'reward_blocked_honest': reward_blocked_honest,
        'reward_blocked_malicious': reward_blocked_malicious,
    })

print(f"Loaded {len(runs)} runs")

print("\n" + "="*80)
print("  PHAN TICH DIEM NOI BAT")
print("="*80)

# 1. Leakage table
print("\n### 1. REWARD LEAKAGE THEO LOAI TAN CONG ###\n")
header = f"{'Method':<8}"
for atk in ATTACK_TYPES:
    header += f" {atk[:10]:>10}"
header += f" {'TB':>8}"
print(header)
print("-" * 75)

for method in METHODS:
    agg, rew = method.split('+')
    row = f"{METHOD_SHORT[method]:<8}"
    for atk in ATTACK_TYPES:
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == atk]
        if subset:
            row += f" {statistics.mean([r['leakage'] for r in subset]):>10.4f}"
        else:
            row += f" {'N/A':>10}"
    atk_runs = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] != 'clean']
    if atk_runs:
        row += f" {statistics.mean([r['leakage'] for r in atk_runs]):>8.4f}"
    print(row)

# 2. Honest vs Malicious reward
print("\n\n### 2. REWARD TRUNG BINH: Honest vs Malicious ###\n")
for atk in ATTACK_TYPES:
    print(f"--- {atk} ---")
    print(f"  {'Method':<8} {'Honest':>10} {'Malicious':>10} {'Ratio':>8} {'Blocked':>8}")
    print(f"  {'-'*48}")
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == atk]
        if not subset:
            continue
        ha = statistics.mean([r['honest_avg'] for r in subset])
        ma = statistics.mean([r['malicious_avg'] for r in subset])
        ratio = ha / ma if ma > 0.001 else float('inf')
        blocked = statistics.mean([r['reward_blocked_malicious'] for r in subset])
        print(f"  {METHOD_SHORT[method]:<8} {ha:>10.6f} {ma:>10.6f} {ratio:>8.2f} {blocked:>8.1f}")
    print()

# 3. CSRA-DCD detection
print("\n### 3. CSRA-DCD: PHAT HIEN VA BLOCK ###\n")
print(f"  {'Attack':<20} {'TPR':>8} {'FPR':>8} {'MalBlock':>10} {'HonBlock':>10} {'Leakage':>10}")
print(f"  {'-'*68}")
for atk in ATTACK_TYPES:
    subset = [r for r in runs if r['agg'] == 'csra_dcd' and r['attack'] == atk]
    if not subset:
        continue
    print(f"  {atk:<20} "
          f"{statistics.mean([r['tpr'] for r in subset]):>8.4f} "
          f"{statistics.mean([r['fpr'] for r in subset]):>8.4f} "
          f"{statistics.mean([r['reward_blocked_malicious'] for r in subset]):>10.1f} "
          f"{statistics.mean([r['reward_blocked_honest'] for r in subset]):>10.1f} "
          f"{statistics.mean([r['leakage'] for r in subset]):>10.4f}")

# 4. Jain by scenario
print("\n\n### 4. JAIN INDEX THEO KICH BAN (CLEAN) ###\n")
for scenario_label, filt in [
    ('K1 (IID)', lambda r: r['scenario'] == 'K1'),
    ('K2 (Weak Non-IID)', lambda r: r['scenario'] == 'K2'),
    ('K3 alpha=0.5', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '050'),
    ('K3 alpha=0.1', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '010'),
]:
    print(f"  {scenario_label}:")
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean' and filt(r)]
        if not subset:
            continue
        j = statistics.mean([r['jain'] for r in subset])
        print(f"    {METHOD_SHORT[method]}: Jain={j:.4f}")
    print()

# 5. Key findings
print("\n### 5. PHAT HIEN CHINH ###\n")

m1_leak = statistics.mean([r['leakage'] for r in runs if r['agg'] == 'fedavg' and r['reward'] == 'equal' and r['attack'] != 'clean'])
m6_leak = statistics.mean([r['leakage'] for r in runs if r['agg'] == 'csra_dcd' and r['reward'] == 'csra' and r['attack'] != 'clean'])
reduction = (1 - m6_leak/m1_leak) * 100

m1_jain = statistics.mean([r['jain'] for r in runs if r['agg'] == 'fedavg' and r['reward'] == 'equal' and r['attack'] == 'clean'])
m6_jain = statistics.mean([r['jain'] for r in runs if r['agg'] == 'csra_dcd' and r['reward'] == 'csra' and r['attack'] == 'clean'])

m1_acc = statistics.mean([r['final_acc'] for r in runs if r['agg'] == 'fedavg' and r['reward'] == 'equal' and r['attack'] == 'clean'])
m6_acc = statistics.mean([r['final_acc'] for r in runs if r['agg'] == 'csra_dcd' and r['reward'] == 'csra' and r['attack'] == 'clean'])

print(f"Leakage: M1={m1_leak:.4f} -> M6={m6_leak:.4f} (giam {reduction:.1f}%)")
print(f"Jain:    M1={m1_jain:.4f} -> M6={m6_jain:.4f}")
print(f"Acc:     M1={m1_acc:.4f} -> M6={m6_acc:.4f}")

for atk in ATTACK_TYPES:
    subset = [r for r in runs if r['agg'] == 'csra_dcd' and r['attack'] == atk]
    if subset:
        tpr = statistics.mean([r['tpr'] for r in subset])
        lk = statistics.mean([r['leakage'] for r in subset])
        print(f"  {atk}: TPR={tpr:.4f}, Leakage={lk:.4f}")
