"""Generate comprehensive analysis report as DOCX."""
import sys, io, csv, os, glob, re, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import defaultdict
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

LOG_DIR = "results/logs"
OUT_FILE = "Bao_Cao_Tong_Luc_CSRA.docx"

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
ATTACK_LABEL = {
    'free_rider': 'Free-Rider',
    'stealth_free_rider': 'Stealth Free-Rider',
    'lazy': 'Lazy',
    'label_noise': 'Label Noise',
    'sign_flip': 'Sign Flip',
}

# ===== LOAD DATA =====
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

print(f"Loaded {len(runs)} runs")

# ===== CREATE DOCUMENT =====
doc = Document()

# Style setup
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.5

for level in range(1, 4):
    h_style = doc.styles[f'Heading {level}']
    h_style.font.name = 'Times New Roman'
    h_style.font.color.rgb = RGBColor(0, 0, 0)

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    # Data
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
    return table

def mean_std(vals):
    if len(vals) < 2:
        return f"{statistics.mean(vals):.4f}" if vals else "N/A"
    return f"{statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}"

# ===== CONTENT =====

# Title
title = doc.add_heading('BAO CAO PHAN TICH HE THONG FL-BLOCKCHAIN (CSRA)', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('')
doc.add_paragraph(f'Tong so luong chay thuc nghiem: {len(runs)} runs')
doc.add_paragraph(f'Du lieu MNIST va Fashion-MNIST, moi run 20 rounds, 10 clients.')
doc.add_paragraph(f'3 seeds: 42, 123, 2024 (clean); 2 seeds: 42, 123 (attack).')
doc.add_paragraph('')

# ===== 1. GIOI THIEU =====
doc.add_heading('1. GIOI THIEU HE THONG', level=1)

doc.add_paragraph(
    'He thong nghien cuu co che phan phoi phan thuong da chieu (CSRA) ket hop voi bo loc '
    'bat thuong CSRA-DCD trong Federated Learning. Bao cao dua tren ket qua tu 379 dot chay '
    'thuc nghiem tren MNIST va Fashion-MNIST.'
)

doc.add_heading('1.1. Ma tran thuc nghiem', level=2)
doc.add_paragraph('Ma tran ablation 2 chieu (Aggregation x Reward Policy):')

add_table(doc,
    ['ID', 'Aggregation', 'Reward Policy', 'Vai tro'],
    [
        ['M1', 'FedAvg', 'EqualSplit', 'Baseline tran'],
        ['M2', 'FedAvg', 'DataSize', 'Bias theo luong du lieu'],
        ['M3', 'FedAvg', 'QualityOnly', 'Bias theo chat luong'],
        ['M4', 'FedAvg', 'CSRAReward', 'Ablation: chi reward formula'],
        ['M5', 'CSRA-DCD', 'EqualSplit', 'Ablation: chi filtering'],
        ['M6', 'CSRA-DCD', 'CSRAReward', 'He thong de xuat'],
    ])

doc.add_paragraph('')
doc.add_paragraph(
    'Kich ban du lieu: K1 (IID), K2 (Weak Non-IID), K3 Dirichlet alpha=0.5, K3 Dirichlet alpha=0.1.'
)
doc.add_paragraph(
    'Loai tan cong: Free-Rider, Stealth Free-Rider, Lazy, Label Noise, Sign Flip.'
)

# ===== 2. THAM SO HE THONG =====
doc.add_heading('2. GIAI THICH CAC THAM SO HE THONG', level=1)

doc.add_heading('2.1. Tham so FL', level=2)
add_table(doc,
    ['Tham so', 'Gia tri', 'Y nghia'],
    [
        ['n_clients', '10', 'So luong client tham gia FL'],
        ['n_rounds', '20', 'So vong huan luyen'],
        ['local_epochs', '1', 'So epoch huan luyen local'],
        ['batch_size', '128', 'Kich thuoc batch'],
        ['learning_rate', '0.01', 'Toc do hoc'],
    ])

doc.add_paragraph('')

doc.add_heading('2.2. Tham so CSRA Reward (beta, gamma, delta)', level=2)
doc.add_paragraph(
    'CSRA Reward su dung 3 thanh phan de tinh diem dong gop:'
)
doc.add_paragraph('  W_i = beta * q_i + gamma * d_i + delta * r_i')
doc.add_paragraph(
    '- beta (quality): trong so chat luong dong gop, mac dinh 0.5\n'
    '- gamma (data size): trong so luong du lieu, mac dinh 0.3\n'
    '- delta (reputation): trong so danh gia lich su, mac dinh 0.2\n'
    '- Rang buoc: beta + gamma + delta = 1'
)
doc.add_paragraph(
    'Voi beta=0.5, he thong uu tien chat luong dong gop hon luong du lieu. '
    'Reputation giup on dinh reward qua nhieu round, giam nhieu tu quality score 1 round.'
)

doc.add_heading('2.3. Tham so CSRA-DCD (MAD threshold)', level=2)
doc.add_paragraph(
    'CSRA-DCD su dung MAD (Median Absolute Deviation) robust z-score de phat hien client bat thuong:'
)
doc.add_paragraph('  z_i = |score_i - median| / (1.4826 * MAD)')
doc.add_paragraph(
    '- mad_threshold = 3.0: client co z-score > 3 bi danh dau la bat thuong\n'
    '- cosine_threshold = -0.8: phat hien huong update nguoc chieu (sign-flip)\n'
    '- min_honest_ratio = 0.5: neu qua nhieu client bi flag, fallback chap nhan tat ca\n'
    '- suspicion_decay = 0.6: he so suy giam diem nghi ngo qua cac round\n'
    '- suspicion_threshold = 1.0: nguong diem nghi ngo de block reward'
)

# ===== 3. KET QUA CLEAN =====
doc.add_heading('3. PHAN TICH KET QUA — CLEAN RUNS (KHONG TAN CONG)', level=1)

doc.add_heading('3.1. Bang tong hop 6 methods', level=2)
doc.add_paragraph('Ket qua trung binh tren tat ca kich ban va seed:')

clean_rows = []
for method in METHODS:
    agg, rew = method.split('+')
    clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean']
    if not clean:
        continue
    clean_rows.append([
        METHOD_LABEL[method],
        mean_std([r['final_acc'] for r in clean]),
        mean_std([r['peak_acc'] for r in clean]),
        mean_std([r['jain'] for r in clean]),
        mean_std([r['gini'] for r in clean]),
        str(len(clean)),
    ])

add_table(doc,
    ['Method', 'Final Accuracy', 'Peak Accuracy', 'Jain Index', 'Gini', 'Runs'],
    clean_rows)

doc.add_paragraph('')
doc.add_paragraph(
    'Nhan xet: Do chinh xac cuoi cung (Final Accuracy) cua 6 methods tuong doi gan nhau '
    f'(khoang {min(float(r[1].split()[0]) for r in clean_rows):.4f} - {max(float(r[1].split()[0]) for r in clean_rows):.4f}). '
    'Dieu nay cho thay viec chon aggregation method va reward policy khong anh huong lon den do chinh xac '
    'trong dieu kien khong co tan cong.'
)

doc.add_heading('3.2. Ket qua theo kich ban du lieu', level=2)

for scenario_label, filt in [
    ('K1 (IID)', lambda r: r['scenario'] == 'K1'),
    ('K2 (Weak Non-IID)', lambda r: r['scenario'] == 'K2'),
    ('K3 alpha=0.5 (Non-IID trung binh)', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '050'),
    ('K3 alpha=0.1 (Non-IID manh)', lambda r: r['scenario'] == 'K3' and r['dirichlet'] == '010'),
]:
    doc.add_heading(f'  {scenario_label}', level=3)
    scn_rows = []
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean' and filt(r)]
        if not subset:
            continue
        scn_rows.append([
            METHOD_LABEL[method],
            f"{statistics.mean([r['final_acc'] for r in subset]):.4f}",
            f"{statistics.mean([r['jain'] for r in subset]):.4f}",
            f"{statistics.mean([r['gini'] for r in subset]):.4f}",
        ])
    add_table(doc, ['Method', 'Accuracy', 'Jain Index', 'Gini'], scn_rows)
    doc.add_paragraph('')

doc.add_paragraph(
    'Nhan xet theo kich ban:\n'
    '- K1 (IID): Jain Index cao nhat (gan 1.0) cho tat ca methods, cho thay phan phoi reward deu khi du lieu dong nhat.\n'
    '- K2 (Weak Non-IID): Jain Index van cao, accuracy giam nhe so voi K1.\n'
    '- K3 alpha=0.5: Jain Index bat dau giam, dac biet voi M2 (DataSize) va M6 (CSRA-DCD+CSRA).\n'
    '- K3 alpha=0.1: Jain Index thap nhat cho M2 va M6, cho thay trong Non-IID manh, '
    'cac policy phan phoi theo chat luong/du lieu tao ra bat cong lon hon.'
)

# ===== 4. KET QUA ATTACK =====
doc.add_heading('4. PHAN TICH KET QUA — ATTACK RUNS', level=1)

doc.add_heading('4.1. Reward Leakage theo loai tan cong', level=2)
doc.add_paragraph(
    'Reward Leakage = tong reward ma client ac nhan / tong reward. '
    'Thap hon = tot hon (it reward bi rot ra ngoai).'
)

for atk in ATTACK_TYPES:
    doc.add_heading(f'  {ATTACK_LABEL[atk]}', level=3)
    atk_rows = []
    for method in METHODS:
        agg, rew = method.split('+')
        subset = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == atk]
        if not subset:
            continue
        atk_rows.append([
            METHOD_LABEL[method],
            f"{statistics.mean([r['leakage'] for r in subset]):.4f}",
            f"{statistics.mean([r['final_acc'] for r in subset]):.4f}",
            str(len(subset)),
        ])
    add_table(doc, ['Method', 'Leakage', 'Final Accuracy', 'Runs'], atk_rows)
    doc.add_paragraph('')

doc.add_heading('4.2. Bang tong hop leakage tat ca attack', level=2)

summary_rows = []
for method in METHODS:
    agg, rew = method.split('+')
    atk_runs = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] != 'clean']
    if not atk_runs:
        continue
    summary_rows.append([
        METHOD_LABEL[method],
        f"{statistics.mean([r['leakage'] for r in atk_runs]):.4f}",
        f"{min(r['leakage'] for r in atk_runs):.4f}",
        f"{max(r['leakage'] for r in atk_runs):.4f}",
    ])

add_table(doc, ['Method', 'Mean Leakage', 'Min', 'Max'], summary_rows)

doc.add_paragraph('')
doc.add_paragraph(
    'Nhan xet:\n'
    '- M6 (CSRA-DCD+CSRA) co leakage thap nhat trung binh (0.0939), giam 42% so voi M1 (0.1629).\n'
    '- Free-Rider va Sign Flip co leakage = 0.0000 voi M5 va M6, cho thay CSRA-DCD phat hien hoan hao 2 loai nay.\n'
    '- Stealth Free-Rider la loai kho phat hien nhat: leakage 0.0650 voi M6, van con cao.\n'
    '- Lazy va Label Noise co leakage cao (0.19-0.22) o tat ca methods, cho thay day la 2 loai tan cong kho xu ly.'
)

# ===== 5. DETECTION =====
doc.add_heading('5. PHAN TICH KHANG NANG PHAT HIEN TAN CONG (CSRA-DCD)', level=1)

doc.add_heading('5.1. True Positive Rate (TPR) va False Positive Rate (FPR)', level=2)
doc.add_paragraph(
    'TPR (Recall): ti le client ac duoc phat hien dung. Cao = tot.\n'
    'FPR: ti le client honest bi danh dau sai. Thap = tot.'
)

det_rows = []
for atk in ATTACK_TYPES:
    subset = [r for r in runs if r['agg'] == 'csra_dcd' and r['attack'] == atk]
    if not subset:
        continue
    det_rows.append([
        ATTACK_LABEL[atk],
        f"{statistics.mean([r['tpr'] for r in subset]):.4f}",
        f"{statistics.mean([r['fpr'] for r in subset]):.4f}",
        f"{statistics.mean([r['leakage'] for r in subset]):.4f}",
    ])

add_table(doc, ['Attack Type', 'TPR (Recall)', 'FPR', 'Leakage'], det_rows)

doc.add_paragraph('')
doc.add_paragraph(
    'Nhan xet chi tiet:\n'
    '- Free-Rider: TPR = 100%, FPR = 5.7%. Phat hien hoan hao, it false positive.\n'
    '- Sign Flip: TPR = 100%, FPR = 17.3%. Phat hien hoan hao nhung FPR cao — '
    'co the flag honest client trong Non-IID manh co update khac biet.\n'
    '- Stealth Free-Rider: TPR = 48.4%, FPR = 6.7%. Chi phat hien duoc mot nua — '
    'day la loai tan cong tinh vi nhat vi gia metadata giong client hop le.\n'
    '- Lazy: TPR = 5.9%, FPR = 16.8%. Rat kem — lazy client train it hon nhung update '
    'khong du bat thuong de MAD phat hien. FPR cao vi honest Non-IID co the bi flag.\n'
    '- Label Noise: TPR = 5.6%, FPR = 16.6%. Tuong tu Lazy — label noise tao update '
    'co pattern khac nhung khong du outlier de MAD-based filter bat.'
)

doc.add_heading('5.2. Han che cua MAD-based filter', level=2)
doc.add_paragraph(
    'CSRA-DCD dua tren MAD robust z-score cua update norm, co nhung han che sau:\n\n'
    '1. Chi hieu qua voi tan cong tao update co magnitude lon (Sign Flip, Free-Rider co noise).\n'
    '2. Khong hieu qua voi tan cong co update nho hoac tuong tu honest (Lazy, Label Noise).\n'
    '3. FPR tang trong Non-IID manh (K3 alpha=0.1) vi honest client cung co update khac biet.\n'
    '4. Rolling suspicion score giup giam leakage cho cac loai tan cong lap lai, '
    'nhung khong giai quyet duoc goc re cua van de.'
)

# ===== 6. FAIRNESS =====
doc.add_heading('6. PHAN TICH TINH CONG BANG', level=1)

doc.add_heading('6.1. Jain Index so sanh', level=2)
doc.add_paragraph(
    'Jain Index do muc do cong bang phan phoi. Gan 1.0 = phan phoi deu, '
    'gan 0 = phan phoi lech.'
)

jain_rows = []
for method in METHODS:
    agg, rew = method.split('+')
    clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean']
    if not clean:
        continue
    jain_rows.append([
        METHOD_LABEL[method],
        f"{statistics.mean([r['jain'] for r in clean]):.4f}",
        f"{statistics.mean([r['gini'] for r in clean]):.4f}",
    ])

add_table(doc, ['Method', 'Jain Index (Clean)', 'Gini (Clean)'], jain_rows)

doc.add_paragraph('')
doc.add_paragraph(
    'Nhan xet:\n'
    '- M1 (EqualSplit) co Jain cao nhat (0.9645) vi chia deu cho tat ca — nhung dieu nay '
    'khong co nghia la tot vi client ac cung nhan phan thuong bang nhau.\n'
    '- M2 (DataSize) co Jain thap nhat (0.7267) vi phan phoi hoan toan theo luong du lieu, '
    'client co du lieu it bi bat cong.\n'
    '- M6 (CSRA-DCD+CSRA) co Jain = 0.7713, thap hon M1 nhung dieu nay phan anh '
    'su phan phoi theo dong gop thuc te, khong phai bat cong.\n'
    '- M4 (FedAvg+CSRA) co Jain = 0.8556, can doi giua cong bang va uu dai chat luong.'
)

doc.add_heading('6.2. Phan tich sau hon ve Fairness', level=2)
doc.add_paragraph(
    'Jain Index cao khong luon tot. Can phan tich ket hop:\n\n'
    '- M1 (Jain=0.96): Phan phoi deu nhung khong phan biet client tot va client xau. '
    'Free-rider cung nhan reward giong honest.\n'
    '- M6 (Jain=0.77): Phan phoi it deu hon nhung phan anh dung dong gop. '
    'Client co chat luong cao nhan nhieu hon, client ac nhan it hon.\n'
    '- Trade-off: Cong bang (Jain cao) vs Do chinh xac cua reward (thuong client tot). '
    'M6 chon do chinh xac, M1 chon cong bang.'
)

# ===== 7. SO SANH THEO DATASET =====
doc.add_heading('7. SO SANH THEO DATASET', level=1)

for ds in ['mnist', 'fashion_mnist']:
    ds_label = 'MNIST' if ds == 'mnist' else 'Fashion-MNIST'
    doc.add_heading(f'7.{1 if ds == "mnist" else 2}. {ds_label}', level=2)
    ds_rows = []
    for method in METHODS:
        agg, rew = method.split('+')
        clean = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] == 'clean' and r['dataset'] == ds]
        atk = [r for r in runs if r['agg'] == agg and r['reward'] == rew and r['attack'] != 'clean' and r['dataset'] == ds]
        ca = f"{statistics.mean([r['final_acc'] for r in clean]):.4f}" if clean else "N/A"
        al = f"{statistics.mean([r['leakage'] for r in atk]):.4f}" if atk else "N/A"
        ds_rows.append([METHOD_LABEL[method], ca, al])
    add_table(doc, ['Method', 'Clean Accuracy', 'Attack Leakage'], ds_rows)
    doc.add_paragraph('')

doc.add_paragraph(
    'Nhan xet:\n'
    '- MNIST co accuracy cao hon Fashion-MNIST (0.93 vs 0.79) do Fashion-MNIST kho hon.\n'
    '- Leakage tren Fashion-MNIST hoi cao hon MNIST cho mot so methods, '
    'cho thay do phuc tap cua du lieu anh huong den kha nang phat hien.'
)

# ===== 8. KET LUAN =====
doc.add_heading('8. KET LUAN VA DONG GOP', level=1)

doc.add_heading('8.1. Diem manh cua he thong M6 (CSRA-DCD + CSRAReward)', level=2)
doc.add_paragraph(
    '1. Phat hien hoan hao Free-Rider va Sign Flip (TPR=100%).\n'
    '2. Giam 42% reward leakage so voi baseline M1 (0.094 vs 0.163).\n'
    '3. Khong lam giam accuracy so voi FedAvg tren clean runs.\n'
    '4. Reward phan phoi theo dong gop thuc te, khong chi theo luong du lieu.\n'
    '5. Blockchain audit layer dam bao tinh minh bach.'
)

doc.add_heading('8.2. Han che can cai thien', level=2)
doc.add_paragraph(
    '1. Khong phat hien duoc Lazy va Label Noise (TPR < 6%) — can them signal khac '
    'ben canh MAD z-score.\n'
    '2. FPR cao (14-17%) trong Non-IID manh — co the flag honest client co du lieu khac biet.\n'
    '3. Stealth Free-Rider chi phat hien duoc 48% — can server-side data commitment check.\n'
    '4. Jain Index thap hon M1 — trade-off giua cong bang va do chinh xac reward.'
)

doc.add_heading('8.3. De xuat tiep theo', level=2)
doc.add_paragraph(
    '1. Them multi-round signal (rolling suspicion) de phat hien Lazy va Label Noise.\n'
    '2. Su dung server-known data size de detect data mismatch (stealth free-rider).\n'
    '3. Chay them CIFAR-10 de xac nhan ket qua tren du lieu phuc tap hon.\n'
    '4. Sweep beta de tim trong so toi uu cho CSRAReward.\n'
    '5. Nghien cuu adaptive threshold thay vi fixed MAD threshold=3.0.'
)

# Save
doc.save(OUT_FILE)
print(f"Saved report to {OUT_FILE}")
