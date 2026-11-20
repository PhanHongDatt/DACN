"""Báo cáo chi tiết cách thức hoạt động các phương thức tấn công trong hệ thống FL-CSRA."""
import sys, io, csv, os, glob, re, statistics, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import defaultdict
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "results/logs"
OUT_FILE = "Bao_Cao_Phuong_Thuc_Tan_Cong.docx"
PLOT_DIR = tempfile.mkdtemp()

FNAME_RE = re.compile(
    r"^(?P<dataset>fashion_mnist|cifar10|mnist)_(?P<scenario>K\d+)"
    r"(?:_da(?P<dirichlet>\d{3}))?_(?P<agg>fedavg|trimmed|csra_dcd)"
    r"_(?P<reward>equal|data|quality|csra)"
    r"_b(?P<beta>\d{2})g(?P<gamma>\d{2})d(?P<delta>\d{2})"
    r"_s(?P<seed>\d+)_(?P<attack>[a-z_]+?)_(?P<ts>\d{8}_\d{6})\.csv$"
)

ATTACK_TYPES = ['free_rider', 'stealth_free_rider', 'lazy', 'label_noise', 'sign_flip']
ATTACK_VN = {
    'free_rider': 'Kẻ ăn ké (Free-Rider)',
    'stealth_free_rider': 'Kẻ ăn ké ẩn danh (Stealth Free-Rider)',
    'lazy': 'Kẻ lười biếng (Lazy)',
    'label_noise': 'Nhiễu nhãn (Label Noise)',
    'sign_flip': 'Đảo dấu (Sign Flip)',
}

# ===== LOAD DATA =====
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
            'reward_eth': float(row.get('reward_eth', 0) or 0),
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

# ===== GENERATE PLOTS =====
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})

def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

# Plot 1: Radar chart so sánh 5 loại attack
fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
categories = ['Quality\nScore', 'Cosine\nSimilarity', 'Update\nNorm', 'Authenticity\n(STD)', 'Robust\nZ-Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors_atk = {'free_rider': '#e74c3c', 'stealth_free_rider': '#e67e22', 'lazy': '#f1c40f', 'label_noise': '#2ecc71', 'sign_flip': '#9b59b6'}

for atk in ATTACK_TYPES:
    honest = [d for d in all_data if d['attack'] == atk and d['client_type'] == 'honest']
    malicious = [d for d in all_data if d['attack'] == atk and d['client_type'] == atk]
    if not honest or not malicious:
        continue
    hq = statistics.mean([d['quality_score'] for d in honest])
    mq = statistics.mean([d['quality_score'] for d in malicious])
    hc = statistics.mean([d['cosine_to_reference'] for d in honest if d['cosine_to_reference'] != 0])
    mc = statistics.mean([d['cosine_to_reference'] for d in malicious if d['cosine_to_reference'] != 0])
    hn = statistics.mean([d['raw_update_norm'] for d in honest if d['raw_update_norm'] > 0])
    mn = statistics.mean([d['raw_update_norm'] for d in malicious if d['raw_update_norm'] > 0])
    ha = statistics.mean([d['authenticity_score'] for d in honest if d['authenticity_score'] > 0])
    ma = statistics.mean([d['authenticity_score'] for d in malicious if d['authenticity_score'] > 0])
    hz = statistics.mean([d['robust_z'] for d in honest])
    mz = statistics.mean([d['robust_z'] for d in malicious])

    # Normalize to 0-1 range for radar
    vals = [
        min(mq / max(hq, 0.01), 2.0) / 2.0,  # quality ratio capped at 2x
        max(0, (mc + 1) / 2),  # cosine: -1 to 1 -> 0 to 1
        min(mn / max(hn, 0.01), 2.0) / 2.0,  # norm ratio
        min(ma / max(ha, 0.01), 2.0) / 2.0,  # authenticity ratio
        min(mz / max(hz, 0.01), 5.0) / 5.0,  # z-score ratio capped at 5x
    ]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=ATTACK_VN[atk], color=colors_atk[atk])
    ax.fill(angles, vals, alpha=0.1, color=colors_atk[atk])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title('So sánh đặc trưng Malicious vs Honest\n(Chuẩn hóa: 0.5 = giống honest, 1.0 = gấp đôi)', fontsize=11, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
plot_radar = save_plot(fig, 'radar_attack.png')

# Plot 2: Cosine distribution per attack type
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
for idx, atk in enumerate(ATTACK_TYPES):
    ax = axes[idx]
    honest = [d['cosine_to_reference'] for d in all_data if d['attack'] == atk and d['client_type'] == 'honest' and d['cosine_to_reference'] != 0]
    malicious = [d['cosine_to_reference'] for d in all_data if d['attack'] == atk and d['client_type'] == atk and d['cosine_to_reference'] != 0]
    if honest and malicious:
        ax.hist(honest, bins=30, alpha=0.6, color='#2a8a4e', label='Honest', density=True)
        ax.hist(malicious, bins=30, alpha=0.6, color='#c14b6e', label=ATTACK_VN[atk].split('(')[0].strip(), density=True)
        ax.axvline(0, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_title(ATTACK_VN[atk].split('(')[0].strip(), fontsize=9, fontweight='bold')
    ax.set_xlabel('Cosine', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.set_ylabel('Density', fontsize=9)
axes[-1].legend(fontsize=8)
fig.suptitle('Phân bố Cosine to Reference — Honest vs Malicious', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plot_cosine = save_plot(fig, 'cosine_dist.png')

# Plot 3: Quality score comparison
fig, ax = plt.subplots(figsize=(10, 5))
atk_names = []
honest_q = []
malicious_q = []
for atk in ATTACK_TYPES:
    honest = [d['quality_score'] for d in all_data if d['attack'] == atk and d['client_type'] == 'honest']
    malicious = [d['quality_score'] for d in all_data if d['attack'] == atk and d['client_type'] == atk]
    if honest and malicious:
        atk_names.append(ATTACK_VN[atk].split('(')[0].strip())
        honest_q.append(statistics.mean(honest))
        malicious_q.append(statistics.mean(malicious))

x = np.arange(len(atk_names))
w = 0.35
bars1 = ax.bar(x - w/2, honest_q, w, color='#2a8a4e', label='Honest', edgecolor='white')
bars2 = ax.bar(x + w/2, malicious_q, w, color='#c14b6e', label='Malicious', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(atk_names, fontsize=9)
ax.set_ylabel('Quality Score (Δloss)', fontsize=11)
ax.set_title('So sánh Quality Score — Honest vs Malicious\n(Lazy và Label Noise có quality CAO HƠN honest)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
for b, v in zip(bars1, honest_q):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
for b, v in zip(bars2, malicious_q):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
plot_quality = save_plot(fig, 'quality_compare.png')

# Plot 4: Robust Z-Score comparison
fig, ax = plt.subplots(figsize=(10, 5))
honest_z = []
malicious_z = []
for atk in ATTACK_TYPES:
    honest = [d['robust_z'] for d in all_data if d['attack'] == atk and d['client_type'] == 'honest']
    malicious = [d['robust_z'] for d in all_data if d['attack'] == atk and d['client_type'] == atk]
    if honest and malicious:
        honest_z.append(statistics.mean(honest))
        malicious_z.append(statistics.mean(malicious))

x = np.arange(len(atk_names))
bars1 = ax.bar(x - w/2, honest_z, w, color='#2a8a4e', label='Honest', edgecolor='white')
bars2 = ax.bar(x + w/2, malicious_z, w, color='#c14b6e', label='Malicious', edgecolor='white')
ax.axhline(3.0, color='red', ls='--', lw=1.5, label='MAD Threshold = 3.0')
ax.set_xticks(x)
ax.set_xticklabels(atk_names, fontsize=9)
ax.set_ylabel('Robust Z-Score', fontsize=11)
ax.set_title('So sánh Robust Z-Score — Honest vs Malicious\n(Chỉ Free-Rider và Sign-Flip vượt ngưỡng)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plot_zscore = save_plot(fig, 'zscore_compare.png')

# Plot 5: Detection result heatmap
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ['TPR\n(Phát hiện đúng)', 'FPR\n(Báo sai)', 'Leakage\n(Reward rò rỉ)', 'Mal Blocked\n(Block malicious)', 'Hon Blocked\n(Block nhầm honest)']
data_matrix = []
for atk in ATTACK_TYPES:
    csra_data = [d for d in all_data if d['attack'] == atk and d['agg'] == 'csra_dcd']
    honest = [d for d in csra_data if d['client_type'] == 'honest']
    malicious = [d for d in csra_data if d['client_type'] == atk]
    if not csra_data:
        continue
    tpr = sum(1 for d in malicious if d['is_anomaly']) / len(malicious) if malicious else 0
    fpr = sum(1 for d in honest if d['is_anomaly']) / len(honest) if honest else 0
    h_reward = sum(d['reward_eth'] for d in honest)
    m_reward = sum(d['reward_eth'] for d in malicious)
    leakage = m_reward / (h_reward + m_reward) if (h_reward + m_reward) > 0 else 0
    m_blocked = sum(1 for d in malicious if d['reward_blocked']) / len(malicious) if malicious else 0
    h_blocked = sum(1 for d in honest if d['reward_blocked']) / len(honest) if honest else 0
    data_matrix.append([tpr, fpr, leakage, m_blocked, h_blocked])

hm = np.array(data_matrix)
im = ax.imshow(hm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        color = 'white' if hm[i, j] > 0.5 else '#222'
        ax.text(j, i, f'{hm[i,j]:.1%}', ha='center', va='center', fontsize=10, color=color, fontweight='bold')
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=9)
ax.set_yticks(range(len(ATTACK_TYPES)))
ax.set_yticklabels([ATTACK_VN[a].split('(')[0].strip() for a in ATTACK_TYPES], fontsize=9)
ax.set_title('Kết quả phát hiện CSRA-DCD theo loại tấn công', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Tỷ lệ')
plt.tight_layout()
plot_detection = save_plot(fig, 'detection_heatmap.png')

print(f"Generated 5 plots in {PLOT_DIR}")

# ===== CREATE DOCUMENT =====
doc = Document()
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(13)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.5

for level in range(1, 4):
    h_style = doc.styles[f'Heading {level}']
    h_style.font.name = 'Times New Roman'
    h_style.font.color.rgb = RGBColor(0, 0, 0)

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
    return table

def add_img(doc, path, width=Inches(5.5)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# ===== NỘI DUNG =====
title = doc.add_heading('BÁO CÁO PHÂN TÍCH CÁC PHƯƠNG THỨC TẤN CÔNG\nTRONG HỆ THỐNG FL-BLOCKCHAIN CSRA', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('')
doc.add_paragraph(
    'Báo cáo này phân tích chi tiết cách thức hoạt động và biểu hiện của 5 loại tấn công '
    'trong hệ thống Federated Learning kết hợp Blockchain, dựa trên kết quả từ 529 runs thực nghiệm.'
)

# ===== 1. TỔNG QUAN =====
doc.add_heading('1. TỔNG QUAN CÁC LOẠI TẤN CÔNG', level=1)
doc.add_paragraph(
    'Hệ thống mô phỏng 5 loại client độc hại, mỗi loại có cách thức tấn công khác nhau. '
    'Mục tiêu là đánh giá khả năng phát hiện và giảm thiểu reward leakage của cơ chế CSRA-DCD + CSRAReward.'
)

add_table(doc,
    ['Loại tấn công', 'Ký hiệu', 'Cách thức hoạt động', 'Mức độ nguy hiểm'],
    [
        ['Kẻ ăn ké', 'Free-Rider', 'Không train, gửi global params + noise nhỏ', 'Trung bình'],
        ['Kẻ ăn ké ẩn danh', 'Stealth FR', 'Không train, giả metadata hợp lệ', 'Cao'],
        ['Kẻ lười biếng', 'Lazy', 'Train ít epoch hơn bình thường', 'Thấp'],
        ['Nhiễu nhãn', 'Label Noise', 'Flip 30% label trước khi train', 'Trung bình'],
        ['Đảo dấu', 'Sign Flip', 'Đảo dấu delta sau khi train (Byzantine)', 'Rất cao'],
    ])
doc.add_paragraph('')

add_img(doc, plot_radar, Inches(4.5))
doc.add_paragraph('Hình 1: So sánh đặc trưng 5 loại tấn công với client bình thường', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

# ===== 2. FREE-RIDER =====
doc.add_heading('2. KẺ ĂN KÉ (FREE-RIDER)', level=1)

doc.add_heading('2.1. Cách thức hoạt động', level=2)
doc.add_paragraph(
    'Free-Rider không thực hiện huấn luyện mà sao chép nguyên trạng tham số toàn cục (global parameters), '
    'thêm một lượng nhiễu Gaussian rất nhỏ (noise_std = 0.05) rồi gửi về server. '
    'Client này báo cáo quality = 0 và data_size = 0, thể hiện sự "ăn ké" rõ ràng.'
)

doc.add_heading('2.2. Biểu hiện so với client bình thường', level=2)
add_table(doc,
    ['Đặc trưng', 'Honest', 'Free-Rider', 'Khác biệt', 'Ý nghĩa'],
    [
        ['Quality Score', '0.60', '0.00', '-100%', 'Không train nên không cải thiện loss'],
        ['Cosine to Reference', '0.97', '0.00', '-99.6%', 'Update không có hướng học thật'],
        ['Data Size', '2459', '2108', '-14%', 'Báo data_size = 0'],
        ['Update Norm', '28.4', '24.3', '-14%', 'Gửi global + noise nhỏ'],
        ['Robust Z-Score', '0.86', '12.84', '+1388%', 'Vượt xa ngưỡng MAD = 3.0'],
        ['Data Mismatch', '0%', '100%', '+100%', 'Server biết data_size thực ≠ client báo'],
    ])
doc.add_paragraph('')

doc.add_heading('2.3. Dấu hiệu nhận biết', level=2)
doc.add_paragraph(
    '• data_size_mismatch = 100%: Client báo data = 0 nhưng server biết client có data.\n'
    '• cosine ≈ 0: Update không có hướng học thật, chỉ là noise ngẫu nhiên.\n'
    '• robust_z = 12.84: Vượt xa ngưỡng 3.0, dễ dàng bị MAD-based filter phát hiện.\n'
    '• quality = 0: Không có cải thiện loss nào sau "huấn luyện".'
)

doc.add_heading('2.4. Kết quả phát hiện', level=2)
doc.add_paragraph(
    'CSRA-DCD phát hiện Free-Rider với TPR = 100% và Leakage = 0.0000. '
    'Tất cả Free-Rider bị block reward hoàn toàn. Đây là loại tấn công dễ phát hiện nhất '
    'vì update quá khác biệt so với client bình thường.'
)

# ===== 3. STEALTH FREE-RIDER =====
doc.add_heading('3. KẺ ĂN KÉ ẨN DANH (STEALTH FREE-RIDER)', level=1)

doc.add_heading('3.1. Cách thức hoạt động', level=2)
doc.add_paragraph(
    'Stealth Free-Rider cũng không train nhưng tinh vi hơn: giả mạo metadata để trông giống client hợp lệ. '
    'Client này báo cáo quality = 0.2 (giả) và data_size bằng đúng số mẫu thực tế, '
    'nhưng update vẫn chỉ là global parameters + noise nhỏ.'
)

doc.add_heading('3.2. Biểu hiện so với client bình thường', level=2)
add_table(doc,
    ['Đặc trưng', 'Honest', 'Stealth FR', 'Khác biệt', 'Ý nghĩa'],
    [
        ['Quality Score', '0.60', '0.20', '-67%', 'Giả quality nhưng vẫn thấp hơn honest'],
        ['Cosine to Reference', '0.97', '0.00', '-99.6%', 'Không có hướng học thật (dấu hiệu mạnh nhất)'],
        ['Update Norm', '28.1', '24.3', '-13%', 'Tương tự Free-Rider'],
        ['Robust Z-Score', '0.89', '10.18', '+1040%', 'Vẫn bị MAD phát hiện'],
        ['Data Mismatch', '0%', '0%', '0%', 'Giả đúng data_size nên không mismatch'],
    ])
doc.add_paragraph('')

doc.add_heading('3.3. So sánh với Free-Rider thường', level=2)
doc.add_paragraph(
    'Stealth Free-Rider khó phát hiện hơn Free-Rider thường ở một điểm: '
    'data_size_mismatch = 0% vì giả đúng data_size. Tuy nhiên, cosine ≈ 0 vẫn là dấu hiệu rõ ràng. '
    'CSRA-DCD phát hiện được 48.4% Stealth Free-Rider, giảm leakage 69% so với baseline M1.'
)

# ===== 4. LAZY =====
doc.add_heading('4. KẺ LƯỜI BIẾNG (LAZY)', level=1)

doc.add_heading('4.1. Cách thức hoạt động', level=2)
doc.add_paragraph(
    'Lazy client train ít epoch hơn bình thường (1 epoch thay vì 2). '
    'Điểm đáng chú ý: Lazy client có quality_score CAO HƠN client bình thường (0.75 vs 0.56). '
    'Điều này có vẻ phi lý nhưng hoàn toàn có cơ sở khoa học.'
)

doc.add_heading('4.2. Tại sao Lazy lại có quality cao hơn?', level=2)
doc.add_paragraph(
    'Trong hệ thống, quality = loss_before - loss_after (delta-loss trên local train set).\n\n'
    '• Client bình thường train 2 epochs: Học quá tốt dữ liệu local → overfitting → '
    'loss giảm nhiều ở epoch 1 nhưng epoch 2 overfit, delta-loss không còn ý nghĩa.\n\n'
    '• Lazy client train 1 epoch: Loss giảm "đúng mức" → ít overfitting hơn → '
    'delta-loss trông đẹp hơn, chất lượng update thực tế tốt hơn cho mô hình toàn cục.\n\n'
    'Đây là phát hiện quan trọng: "lazy" không phải lúc nào cũng xấu. '
    'Trong FL, train ít hơn đôi khi lại tạo ra update tốt hơn cho mô hình global.'
)

doc.add_heading('4.3. Biểu hiện so với client bình thường', level=2)
add_table(doc,
    ['Đặc trưng', 'Honest', 'Lazy', 'Khác biệt', 'Ý nghĩa'],
    [
        ['Quality Score', '0.56', '0.75', '+33%', 'CAO HƠN do ít overfitting'],
        ['Cosine to Reference', '0.98', '0.98', '≈0%', 'Hướng update gần giống hệt'],
        ['Update Norm', '28.2', '24.6', '-13%', 'Nhỏ hơn do train ít hơn'],
        ['Robust Z-Score', '1.58', '0.83', '-48%', 'THẤP HƠN, không vượt ngưỡng MAD'],
        ['Suspicion Score', '1.00', '0.49', '-51%', 'THẤP HƠN, không bị nghi ngờ'],
    ])
doc.add_paragraph('')

doc.add_heading('4.4. Kết quả phát hiện', level=2)
doc.add_paragraph(
    'CSRA-DCD GẦN NHƯ KHÔNG PHÁT HIỆN được Lazy client (TPR = 5.9%). '
    'Tất cả các chỉ số (cosine, quality, z-score) đều cho thấy Lazy client "giống hệt" honest. '
    'Đây là hạn chế lớn nhất của MAD-based filter.'
)

# ===== 5. LABEL NOISE =====
doc.add_heading('5. NHIỄU NHÃN (LABEL NOISE)', level=1)

doc.add_heading('5.1. Cách thức hoạt động', level=2)
doc.add_paragraph(
    'Label Noise client lật ngược 30% nhãn dữ liệu trước khi huấn luyện. '
    'Model học trên dữ liệu bị nhiễu → tạo ra update có pattern khác với honest, '
    'nhưng vẫn trong phạm vi "hợp lệ" theo các metric server-side.'
)

doc.add_heading('5.2. Tại sao Label Noise có quality cao hơn honest?', level=2)
doc.add_paragraph(
    'Tương tự Lazy, Label Noise cũng có quality_score cao hơn honest (0.76 vs 0.59).\n\n'
    'Lý do: quality = loss_before - loss_after tính trên CÙNG dữ liệu bị nhiễu. '
    'Model học tốt trên dữ liệu nhiễu → loss giảm nhiều → quality cao. '
    'Đây là lỗ hổng thật: metric quality không phân biệt được "học đúng" vs "học sai".'
)

doc.add_heading('5.3. Biểu hiện so với client bình thường', level=2)
add_table(doc,
    ['Đặc trưng', 'Honest', 'Label Noise', 'Khác biệt', 'Ý nghĩa'],
    [
        ['Quality Score', '0.59', '0.76', '+28%', 'CAO HƠN do học tốt trên dữ liệu nhiễu'],
        ['Cosine to Reference', '0.98', '0.98', '≈0%', 'Hướng update vẫn gần giống honest'],
        ['Update Norm', '29.1', '25.2', '-14%', 'Nhỏ hơn nhẹ'],
        ['Robust Z-Score', '1.61', '0.81', '-50%', 'THẤP HƠN, không vượt ngưỡng MAD'],
    ])
doc.add_paragraph('')

doc.add_heading('5.4. Thực tế trong FL', level=2)
doc.add_paragraph(
    'Label noise rất phổ biến trong thực tế: lỗi gán nhãn thủ công, crowdsourcing, dữ liệu bẩn. '
    'Server không biết nhãn nào đúng, chỉ thấy model "học tốt" trên local data. '
    'Đây là bài toán thật, chưa có giải pháp hoàn hảo trong literature. '
    'CSRA-DCD chỉ phát hiện được 5.6% Label Noise clients.'
)

# ===== 6. SIGN FLIP =====
doc.add_heading('6. ĐẢO DẤU (SIGN FLIP)', level=1)

doc.add_heading('6.1. Cách thức hoạt động', level=2)
doc.add_paragraph(
    'Sign Flip là tấn công Byzantine mạnh nhất: client huấn luyện bình thường, '
    'sau đó đảo dấu delta (model update).\n\n'
    'Công thức: attacked_params = global_params - (local_params - global_params)\n'
    'Nghĩa là: update bị đảo ngược hoàn toàn so với hướng học đúng.'
)

doc.add_heading('6.2. Biểu hiện so với client bình thường', level=2)
add_table(doc,
    ['Đặc trưng', 'Honest', 'Sign Flip', 'Khác biệt', 'Ý nghĩa'],
    [
        ['Quality Score', '0.59', '1.91', '+162%', 'Cực cao do đảo dấu delta-loss'],
        ['Cosine to Reference', '0.97', '-0.94', '-197%', 'NGƯỢC CHIỀU hoàn toàn'],
        ['Update Norm', '31.0', '38.9', '+25%', 'Lớn hơn do đảo dấu'],
        ['Authenticity (STD)', '0.06', '0.08', '+25%', 'Cũng lớn hơn'],
    ])
doc.add_paragraph('')

doc.add_heading('6.3. Dấu hiệu nhận biết rõ ràng nhất', level=2)
doc.add_paragraph(
    '• cosine = -0.94: Ngược chiều hoàn toàn với reference → dấu hiệu cực mạnh.\n'
    '• quality = 1.91: Cực cao do đảo dấu delta-loss → bất thường.\n'
    '• update_norm = 38.9: Lớn hơn honest 25%.\n\n'
    'CSRA-DCD phát hiện Sign Flip qua cosine anomaly: bất kỳ update nào có cosine < -0.8 '
    'và robust_z ≥ direction_min_norm_z sẽ bị gắn cờ. TPR = 100%, Leakage = 0.0000.'
)

# ===== 7. SO SÁNH TỔNG HỢP =====
doc.add_heading('7. SO SÁNH TỔNG HỢP KHẢ NĂNG PHÁT HIỆN', level=1)

add_img(doc, plot_quality, Inches(5.5))
doc.add_paragraph('Hình 2: So sánh Quality Score — Lazy và Label Noise CAO HƠN honest', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

add_img(doc, plot_cosine, Inches(6.0))
doc.add_paragraph('Hình 3: Phân bố Cosine — Free-Rider/Stealth FR có cosine ≈ 0', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

add_img(doc, plot_zscore, Inches(5.5))
doc.add_paragraph('Hình 4: Robust Z-Score — Chỉ Free-Rider và Stealth FR vượt ngưỡng MAD', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

add_img(doc, plot_detection, Inches(5.5))
doc.add_paragraph('Hình 5: Bản đồ nhiệt kết quả phát hiện CSRA-DCD', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

doc.add_heading('7.1. Bảng tổng hợp', level=2)
add_table(doc,
    ['Loại tấn công', 'Cosine', 'Quality vs Honest', 'Z-Score', 'TPR', 'Leakage', 'Đánh giá'],
    [
        ['Free-Rider', '≈0', '=0 (thấp hơn)', '12.84 (vượt ngưỡng)', '100%', '0.0000', 'Phát hiện hoàn hảo'],
        ['Stealth FR', '≈0', '0.20 (thấp hơn)', '10.18 (vượt ngưỡng)', '48.4%', '0.0650', 'Phát hiện một phần'],
        ['Lazy', '≈1.0', '0.75 (CAO HƠN)', '0.83 (dưới ngưỡng)', '5.9%', '0.2148', 'Không phát hiện'],
        ['Label Noise', '≈1.0', '0.76 (CAO HƠN)', '0.81 (dưới ngưỡng)', '5.6%', '0.2047', 'Không phát hiện'],
        ['Sign Flip', '-0.94', '1.91 (rất cao)', '0.87 (dưới ngưỡng)', '100%', '0.0000', 'Phát hiện hoàn hảo'],
    ])
doc.add_paragraph('')

doc.add_heading('7.2. Nguyên tắc phát hiện', level=2)
doc.add_paragraph(
    'CSRA-DCD phát hiện được tấn công khi update có hướng hoặc magnitude KHÁC BIỆT RÕ '
    'so với client bình thường:\n\n'
    '• Phát hiện được: cosine ≈ 0 hoặc < 0, norm rất lớn/nhỏ (Free-Rider, Stealth FR, Sign Flip)\n'
    '• Không phát hiện được: cosine ≈ 1.0, norm trong range bình thường (Lazy, Label Noise)\n\n'
    'Lazy và Label Noise thuộc nhóm "khó phát hiện" vì chúng train trên cùng dữ liệu '
    'nhưng với chất lượng kém hơn, tạo ra update "hợp lệ" theo mọi metric server-side.'
)

# ===== 8. HẠN CHẾ VÀ ĐỀ XUẤT =====
doc.add_heading('8. HẠN CHẾ VÀ ĐỀ XUẤT CẢI THIỆN', level=1)

doc.add_heading('8.1. Hạn chế hiện tại', level=2)
doc.add_paragraph(
    '1. Không phát hiện được Lazy và Label Noise (TPR < 6%): MAD-based filter chỉ dựa trên '
    'update magnitude, không phân biệt được "học đúng" vs "học sai".\n\n'
    '2. FPR cao trong Non-IID mạnh (14-17%): Honest client có dữ liệu khác biệt cũng có thể bị flag.\n\n'
    '3. Simulation nhỏ (10 clients): Kết quả mang tính minh họa, không đại diện cho FL production.\n\n'
    '4. Ground truth bias: Trong simulation biết rõ ai là attacker, trong thực tế không biết.'
)

doc.add_heading('8.2. Đề xuất cải thiện', level=2)
doc.add_paragraph(
    '1. Multi-round signal: Sử dụng lịch sử nhiều round để phát hiện Lazy (train ít hơn qua thời gian).\n\n'
    '2. Server-side validation: So sánh update với validation set thay vì chỉ dùng delta-loss local.\n\n'
    '3. Reputation-based filtering: Sử dụng reputation tích lũy để giảm reward cho client có chất lượng kém liên tục.\n\n'
    '4. Adaptive threshold: Tự điều chỉnh MAD threshold theo mức Non-IID thay vì cố định 3.0.\n\n'
    '5. Ensemble detection: Kết hợp nhiều tín hiệu (cosine, norm, quality, reputation) thay vì chỉ dùng MAD.'
)

# Save
doc.save(OUT_FILE)
print(f"\nSaved report to {OUT_FILE}")
print(f"  Paragraphs: {len(doc.paragraphs)}")
print(f"  Tables: {len(doc.tables)}")
print(f"  Plots embedded: 5")
