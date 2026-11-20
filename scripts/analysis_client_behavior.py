"""Phân tích biểu hiện của từng loại client độc hại so với client bình thường."""
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

# Load all attack runs with per-client data
all_data = []
for f in sorted(glob.glob(os.path.join(LOG_DIR, "*.csv"))):
    fname = os.path.basename(f)
    m = FNAME_RE.match(fname)
    if not m:
        continue
    attack = m['attack']
    if attack not in ATTACK_TYPES:
        continue
    with open(f, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    for row in rows:
        all_data.append({
            'attack': attack,
            'agg': m['agg'],
            'client_type': row.get('client_type', 'honest'),
            'quality_score': float(row.get('quality_score', 0) or 0),
            'data_size': int(row.get('data_size', 0) or 0),
            'reported_data_size': int(row.get('reported_data_size', 0) or 0),
            'reward_eth': float(row.get('reward_eth', 0) or 0),
            'anomaly_score': float(row.get('anomaly_score', 0) or 0) if row.get('anomaly_score') else 0,
            'raw_update_norm': float(row.get('raw_update_norm', 0) or 0) if row.get('raw_update_norm') else 0,
            'cosine_to_reference': float(row.get('cosine_to_reference', 0) or 0) if row.get('cosine_to_reference') else 0,
            'authenticity_score': float(row.get('authenticity_score', 0) or 0) if row.get('authenticity_score') else 0,
            'is_anomaly': row.get('is_anomaly', '0') == '1',
            'reward_blocked': row.get('reward_blocked', '0') == '1',
            'robust_z': float(row.get('robust_z', 0) or 0) if row.get('robust_z') else 0,
            'data_size_mismatch': row.get('data_size_mismatch', '0') == '1',
            'suspicion_score': float(row.get('suspicion_score', 0) or 0) if row.get('suspicion_score') else 0,
        })

print(f"Loaded {len(all_data)} data points")

# ===== ANALYSIS =====
print("\n" + "="*90)
print("  PHAN TICH BIEN HIEN CUA TUNG LOAI CLIENT DOC HAI")
print("="*90)

for atk in ATTACK_TYPES:
    honest = [d for d in all_data if d['attack'] == atk and d['client_type'] == 'honest']
    malicious = [d for d in all_data if d['attack'] == atk and d['client_type'] == atk]

    if not honest or not malicious:
        continue

    print(f"\n\n{'='*90}")
    print(f"  {atk.upper().replace('_', ' ')}")
    print(f"{'='*90}")

    # Compare each metric
    metrics = [
        ('quality_score', 'Quality Score (Δloss)', True),
        ('data_size', 'Data Size', True),
        ('reward_eth', 'Reward (ETH)', True),
        ('raw_update_norm', 'Update Norm (||Δ||₂)', True),
        ('cosine_to_reference', 'Cosine to Reference', True),
        ('authenticity_score', 'Authenticity (STD)', True),
        ('robust_z', 'Robust Z-Score', False),
        ('suspicion_score', 'Suspicion Score', False),
    ]

    print(f"\n  {'Metric':<30} {'Honest TB':>12} {'Malicious TB':>14} {'Chenh lech':>12} {'Danh gia':>15}")
    print(f"  {'-'*85}")

    for key, label, higher_is_better in metrics:
        h_vals = [d[key] for d in honest if d[key] != 0 or key in ['reward_eth']]
        m_vals = [d[key] for d in malicious if d[key] != 0 or key in ['reward_eth']]

        if not h_vals or not m_vals:
            continue

        h_mean = statistics.mean(h_vals)
        m_mean = statistics.mean(m_vals)

        if h_mean != 0:
            diff_pct = ((m_mean - h_mean) / abs(h_mean)) * 100
        else:
            diff_pct = 0

        # Determine if this difference helps detection
        if abs(diff_pct) < 5:
            eval_str = "~Giong nhau"
        elif key == 'quality_score' and m_mean < h_mean:
            eval_str = "Mal thap hon"
        elif key == 'data_size' and m_mean < h_mean:
            eval_str = "Mal it data hon"
        elif key == 'raw_update_norm' and m_mean > h_mean * 1.5:
            eval_str = "Mal lon hon (co the detect)"
        elif key == 'cosine_to_reference' and m_mean < h_mean - 0.2:
            eval_str = "Mal khac huong (detect)"
        elif key == 'authenticity_score' and m_mean < h_mean * 0.5:
            eval_str = "Mal thap hon (detect)"
        elif key == 'robust_z' and m_mean > 3.0:
            eval_str = "Vuot nguong (bi flag)"
        else:
            eval_str = "Kha biet duoc"

        print(f"  {label:<30} {h_mean:>12.6f} {m_mean:>14.6f} {diff_pct:>+11.1f}% {eval_str:>15}")

    # Detection stats
    h_anomaly = sum(1 for d in honest if d['is_anomaly']) / len(honest) * 100
    m_anomaly = sum(1 for d in malicious if d['is_anomaly']) / len(malicious) * 100
    h_blocked = sum(1 for d in honest if d['reward_blocked']) / len(honest) * 100
    m_blocked = sum(1 for d in malicious if d['reward_blocked']) / len(malicious) * 100

    print(f"\n  Detection:")
    print(f"    Honest bi flag:    {h_anomaly:.1f}% anomaly, {h_blocked:.1f}% reward blocked")
    print(f"    Malicious bi flag: {m_anomaly:.1f}% anomaly, {m_blocked:.1f}% reward blocked")

    # Data mismatch
    h_mismatch = sum(1 for d in honest if d['data_size_mismatch']) / len(honest) * 100
    m_mismatch = sum(1 for d in malicious if d['data_size_mismatch']) / len(malicious) * 100
    print(f"    Data mismatch:     Honest={h_mismatch:.1f}%, Malicious={m_mismatch:.1f}%")

# ===== SUMMARY TABLE =====
print("\n\n" + "="*90)
print("  BANG TONG HOP SO SANH HONEST VS MALICIOUS")
print("="*90)

print(f"\n  {'Attack':<20} {'Quality':>10} {'DataSize':>10} {'UpdateNorm':>12} {'Cosine':>10} {'Authenticity':>14}")
print(f"  {'-'*78}")

for atk in ATTACK_TYPES:
    honest = [d for d in all_data if d['attack'] == atk and d['client_type'] == 'honest']
    malicious = [d for d in all_data if d['attack'] == atk and d['client_type'] == atk]
    if not honest or not malicious:
        continue

    hq = statistics.mean([d['quality_score'] for d in honest])
    mq = statistics.mean([d['quality_score'] for d in malicious])
    hd = statistics.mean([d['data_size'] for d in honest])
    md = statistics.mean([d['data_size'] for d in malicious])
    hn = statistics.mean([d['raw_update_norm'] for d in honest if d['raw_update_norm'] > 0])
    mn = statistics.mean([d['raw_update_norm'] for d in malicious if d['raw_update_norm'] > 0])
    hc = statistics.mean([d['cosine_to_reference'] for d in honest if d['cosine_to_reference'] != 0])
    mc = statistics.mean([d['cosine_to_reference'] for d in malicious if d['cosine_to_reference'] != 0])
    ha = statistics.mean([d['authenticity_score'] for d in honest if d['authenticity_score'] > 0])
    ma = statistics.mean([d['authenticity_score'] for d in malicious if d['authenticity_score'] > 0])

    print(f"  {atk:<20}")
    print(f"    {'Honest':<18} {hq:>10.4f} {hd:>10.0f} {hn:>12.4f} {hc:>10.4f} {ha:>14.4f}")
    print(f"    {'Malicious':<18} {mq:>10.4f} {md:>10.0f} {mn:>12.4f} {mc:>10.4f} {ma:>14.4f}")
    # Ratio
    if hq > 0: print(f"    {'Ratio M/H':<18} {mq/hq:>10.2f}x {md/hd if hd > 0 else 0:>10.2f}x {mn/hn if hn > 0 else 0:>12.2f}x {mc/hc if hc > 0 else 0:>10.2f}x {ma/ha if ha > 0 else 0:>14.2f}x")
    print()
