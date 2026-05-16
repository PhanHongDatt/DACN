"""
loader.py — Load và parse toàn bộ CSV logs (schema v2).

Schema v2 filename:
  <dataset>_<scenario>[_da<3digit>]_<agg>_<reward>_b<2d>g<2d>d<2d>_s<seed>_<attack>_<ts>.csv

Ví dụ:
  mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_143022.csv
  cifar10_K3_da010_csra_dcd_csra_b50g30d20_s2024_free_rider_20260521_091533.csv

Reference: docs/PLAN.md §8.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Regex parse filename — anchored, allow underscores trong agg/attack/reward names
_FNAME_RE = re.compile(
    r"^(?P<dataset>fashion_mnist|cifar10|mnist)"
    r"_(?P<scenario>K\d+)"
    r"(?:_da(?P<dirichlet_raw>\d{3}))?"
    r"_(?P<aggregation_method>fedavg|trimmed|csra_dcd)"
    r"_(?P<reward_policy>equal|data|quality|csra)"
    r"_b(?P<beta_raw>\d{2})g(?P<gamma_raw>\d{2})d(?P<delta_raw>\d{2})"
    r"_s(?P<seed>\d+)"
    r"_(?P<attack_type>clean|free_rider|lazy|label_noise|sign_flip)"
    r"_(?P<timestamp>\d{8}_\d{6})\.csv$"
)

_SCENARIO_LABEL = {"K1": "IID", "K2": "Weak Non-IID", "K3": "Dirichlet Non-IID"}
_AGG_LABEL = {
    "fedavg": "FedAvg",
    "trimmed": "TrimmedMean",
    "csra_dcd": "CSRA-DCD",
}
_REWARD_LABEL = {
    "equal": "EqualSplit",
    "data": "DataSize",
    "quality": "QualityOnly",
    "csra": "CSRAReward",
}
_SCENARIO_ORDER = {"K1": 1, "K2": 2, "K3": 3}
_MALICIOUS_CLIENT_TYPES = {"free_rider", "lazy", "label_noise", "sign_flip", "malicious"}


def _parse_filename(fname: str) -> dict | None:
    """Parse filename theo schema v2. Trả về None nếu không match."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None

    beta = int(m.group("beta_raw")) / 100.0
    gamma = int(m.group("gamma_raw")) / 100.0
    delta = int(m.group("delta_raw")) / 100.0

    dirichlet_raw = m.group("dirichlet_raw")
    dirichlet_alpha = (
        int(dirichlet_raw) / 100.0 if dirichlet_raw is not None
        else (0.1 if m.group("scenario") == "K3" else 0.0)
    )

    return {
        "dataset": m.group("dataset"),
        "scenario": m.group("scenario"),
        "scenario_label": _SCENARIO_LABEL.get(m.group("scenario"), m.group("scenario")),
        "aggregation_method": m.group("aggregation_method"),
        "aggregation_label": _AGG_LABEL.get(m.group("aggregation_method"), m.group("aggregation_method")),
        "reward_policy": m.group("reward_policy"),
        "reward_label": _REWARD_LABEL.get(m.group("reward_policy"), m.group("reward_policy")),
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "seed": int(m.group("seed")),
        "attack_type": m.group("attack_type"),
        "dirichlet_alpha": dirichlet_alpha,
        "timestamp": m.group("timestamp"),
        "run_id": fname[: -len(".csv")],
    }


def load_all_logs(log_dir: Path) -> pd.DataFrame | None:
    """
    Scan log_dir recursively for CSV files matching schema v2, merge với CSV
    content và trả về DataFrame đã chuẩn hoá.

    Columns đầu ra:
      - Filename metadata (overrides CSV nếu trùng): dataset, scenario,
        scenario_label, aggregation_method, aggregation_label, reward_policy,
        reward_label, beta, gamma, delta, seed, attack_type, dirichlet_alpha,
        timestamp, run_id
      - Per-row CSV: round_num, client_id, client_type, quality_score, data_size,
        w_new, reputation, reward_eth, is_honest, anomaly_score, robust_z,
        is_anomaly, detection_reason, global_accuracy
      - Computed: scenario_variant, is_malicious, has_attack, attack_label,
        run_rounds_observed, run_max_round, run_rows_observed

    Returns None nếu không có file nào hợp lệ.
    """
    log_dir = Path(log_dir)
    csv_files = sorted(log_dir.rglob("*.csv"))

    if not csv_files:
        log.warning("No CSV files found in %s", log_dir)
        return None

    frames = []
    skipped = 0

    for fpath in csv_files:
        meta = _parse_filename(fpath.name)
        if meta is None:
            log.debug("Skipping (filename mismatch): %s", fpath.name)
            skipped += 1
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            log.warning("Cannot read %s: %s", fpath.name, exc)
            skipped += 1
            continue

        if df.empty:
            log.debug("Empty file: %s", fpath.name)
            skipped += 1
            continue

        # Override CSV với filename metadata (authoritative)
        for k, v in meta.items():
            df[k] = v

        frames.append(df)

    if not frames:
        log.error("All %d CSV files were skipped.", len(csv_files))
        return None

    log.info("Loaded %d files (%d skipped). Merging…", len(frames), skipped)
    combined = pd.concat(frames, ignore_index=True)

    # ── Column renames cho consistency ──────────────────────────────────────
    if "round" in combined.columns and "round_num" not in combined.columns:
        combined.rename(columns={"round": "round_num"}, inplace=True)
    if "quality_score" in combined.columns and "quality" not in combined.columns:
        combined.rename(columns={"quality_score": "quality"}, inplace=True)

    # ── Type coercions ──────────────────────────────────────────────────────
    int_cols = ["round_num", "client_id", "data_size", "seed"]
    for col in int_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    float_cols = [
        "quality", "w_new", "reputation", "reward_eth", "global_accuracy",
        "beta", "gamma", "delta", "anomaly_score", "robust_z",
    ]
    for col in float_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    if "is_honest" in combined.columns:
        combined["is_honest"] = (
            pd.to_numeric(combined["is_honest"], errors="coerce")
            .fillna(0).astype(bool)
        )
    if "is_anomaly" in combined.columns:
        combined["is_anomaly"] = (
            pd.to_numeric(combined["is_anomaly"], errors="coerce")
            .fillna(0).astype(bool)
        )

    # ── Scenario variant (giữ K3@α=0.1 và K3@α=0.5 tách biệt) ──────────────
    if "scenario" in combined.columns and "dirichlet_alpha" in combined.columns:
        def _variant(row):
            if row["scenario"] == "K3":
                return f"K3 (dirichlet={float(row['dirichlet_alpha']):g})"
            return str(row["scenario"])
        combined["scenario_variant"] = combined.apply(_variant, axis=1)
    elif "scenario" in combined.columns:
        combined["scenario_variant"] = combined["scenario"].astype(str)

    # ── Method label tổng hợp ───────────────────────────────────────────────
    if "aggregation_method" in combined.columns and "reward_policy" in combined.columns:
        combined["method"] = (
            combined["aggregation_method"].astype(str) + "+" +
            combined["reward_policy"].astype(str)
        )
        combined["method_label"] = (
            combined["aggregation_label"].astype(str) + " + " +
            combined["reward_label"].astype(str)
        )

    # ── Malicious / attack flags ────────────────────────────────────────────
    if "client_type" in combined.columns:
        combined["client_type"] = combined["client_type"].fillna("unknown").astype(str)
        combined["is_malicious"] = combined["client_type"].isin(_MALICIOUS_CLIENT_TYPES)
    else:
        combined["is_malicious"] = False

    if "attack_type" in combined.columns:
        combined["has_attack"] = combined["attack_type"].astype(str) != "clean"
        combined["attack_label"] = combined["has_attack"].map({True: "attack", False: "clean"})
    else:
        combined["has_attack"] = False
        combined["attack_label"] = "clean"

    # ── Run-level metadata ──────────────────────────────────────────────────
    run_meta = combined.groupby("run_id").agg(
        run_rounds_observed=("round_num", "nunique"),
        run_max_round=("round_num", "max"),
        run_rows_observed=("run_id", "size"),
    )
    combined = combined.merge(run_meta, left_on="run_id", right_index=True, how="left")

    # ── Sort consistently ───────────────────────────────────────────────────
    sort_cols = [
        "dataset", "scenario", "dirichlet_alpha", "attack_type",
        "aggregation_method", "reward_policy", "seed", "round_num", "client_id",
    ]
    combined["_scenario_sort"] = combined["scenario"].map(_SCENARIO_ORDER).fillna(99)
    present = [c for c in sort_cols if c in combined.columns]
    combined.sort_values(["_scenario_sort"] + present, inplace=True)
    combined.drop(columns=["_scenario_sort"], errors="ignore", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
