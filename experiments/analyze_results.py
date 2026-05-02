"""
analyze_results.py — Đọc toàn bộ CSV logs và tính metrics cho 3 nhóm.
Xuất kết quả ra results/summary_*.csv và vẽ plots.

Sử dụng: python experiments/analyze_results.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fl.metrics import (
    jain_index, gini_coefficient, contribution_reward_correlation,
    fairness_gap, free_rider_detection_rate, reward_ratio, economic_incentive_index
)

LOG_DIR    = Path("./results/logs")
PLOT_DIR   = Path("./results/plots")
RESULT_DIR = Path("./results")
PLOT_DIR.mkdir(exist_ok=True)

# ── Load all logs ────────────────────────────────────────────
dfs = []
for f in LOG_DIR.glob("*.csv"):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"Skip {f}: {e}")

if not dfs:
    print("No log files found in results/logs/")
    sys.exit(0)

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(dfs)} files")

# ── Group 2: Fairness metrics ─────────────────────────────────
g2_records = []
for (dataset, scenario, config, alpha, rnd), grp in df.groupby(["dataset","scenario","config","alpha","round"]):
    honest = grp[grp.is_honest == 1]
    if len(honest) == 0: continue
    rewards = honest.reward_eth.values
    contribs = honest.quality_score.values + honest.data_size.values / (honest.data_size.values.mean() + 1e-10)
    g2_records.append({
        "dataset": dataset, "scenario": scenario, "config": config, "alpha": alpha, "round": rnd,
        "jain":  jain_index(rewards),
        "gini":  gini_coefficient(rewards),
        "crc":   contribution_reward_correlation(contribs, rewards),
        "fg":    fairness_gap(rewards, contribs),
        "n_honest": len(honest)
    })

g2 = pd.DataFrame(g2_records)
g2_agg = g2.groupby(["dataset","scenario","config","alpha"])[["jain","gini","crc","fg"]].mean().reset_index()
g2_agg.to_csv(RESULT_DIR / "summary_group2_fairness.csv", index=False)
print(f"\n[Group 2] Saved summary_group2_fairness.csv ({len(g2_agg)} rows)")

# ── Group 3: Free-rider metrics ───────────────────────────────
freerider_files = df[df.client_type == "free_rider"]
if len(freerider_files) > 0:
    g3_records = []
    for (dataset, scenario, config, alpha), grp in df.groupby(["dataset","scenario","config","alpha"]):
        honest_r = grp[grp.client_type == "honest"].reward_eth.values
        fr_r     = grp[grp.client_type == "free_rider"].reward_eth.values
        lazy_r   = grp[grp.client_type == "lazy"].reward_eth.values
        if len(honest_r) == 0 or len(fr_r) == 0: continue

        detected_fr = grp[(grp.client_type == "free_rider") & (grp.is_honest == 0)].client_id.unique().tolist()
        actual_fr   = grp[grp.client_type == "free_rider"].client_id.unique().tolist()

        g3_records.append({
            "dataset": dataset, "scenario": scenario, "config": config, "alpha": alpha,
            "fdr": free_rider_detection_rate(detected_fr, actual_fr),
            "rr":  reward_ratio(honest_r, fr_r),
            "eii": economic_incentive_index(
                honest_r.mean(), lazy_r.mean() if len(lazy_r) > 0 else 0,
                grp[grp.client_type=="honest"].data_size.mean(),
                grp[grp.client_type=="lazy"].data_size.mean() if len(lazy_r) > 0 else 1
            )
        })
    g3 = pd.DataFrame(g3_records)
    g3.to_csv(RESULT_DIR / "summary_group3_freerider.csv", index=False)
    print(f"[Group 3] Saved summary_group3_freerider.csv ({len(g3)} rows)")

# ── Plot: Fairness Gap B vs C ─────────────────────────────────
for ds in df.dataset.unique():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sc in zip(axes, ["K1", "K2", "K3"]):
        sub = g2_agg[(g2_agg.dataset == ds) & (g2_agg.scenario == sc)]
        for cfg, color in [("B","#E07070"), ("C","#4A90D9")]:
            s = sub[sub.config == cfg].sort_values("alpha")
            if len(s) > 0:
                ax.bar([str(a) for a in s.alpha], s.fg, color=color, alpha=0.75, label=f"Config {cfg}")
        ax.set_title(f"{ds} — {sc}")
        ax.set_xlabel("α")
        ax.set_ylabel("Fairness Gap (↓ better)")
        ax.legend()
        ax.set_ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"fairness_gap_{ds}.png", dpi=150)
    plt.close()
    print(f"[Plot] fairness_gap_{ds}.png saved")

print("\nAnalysis complete. Results in ./results/")
