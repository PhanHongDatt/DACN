"""
loader.py — Load và parse toàn bộ CSV logs từ thư mục results/logs.

Filename format: <dataset>_<scenario>_<config>_<alpha>_<timestamp>.csv
  dataset  : mnist | fashion_mnist | cifar10
  scenario : K1 (IID) | K3 (Non-IID)
  config   : A (Traditional FL) | B (Blockchain baseline) | C (CSRA reward)
  alpha    : a00=0.0 | a03=0.3 | a05=0.5 | a07=0.7 | a10=1.0
"""
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Regex: supports fashion_mnist (underscore in dataset name)
_FNAME_RE = re.compile(
    r"^(?P<dataset>fashion_mnist|cifar10|mnist)"
    r"_(?P<scenario>K\d+)"
    r"_(?P<config>[A-Za-z0-9][A-Za-z0-9-]*)"
    r"_a(?P<alpha_raw>\d{2})"
    r"(?:_da(?P<dirichlet_raw>\d{3}))?"
    r"_(?P<timestamp>\d{8}_\d{6})\.csv$"
)

_ALPHA_MAP = {"00": 0.0, "03": 0.3, "05": 0.5, "07": 0.7, "10": 1.0}

_SCENARIO_LABEL = {"K1": "IID", "K2": "Weak Non-IID", "K3": "Dirichlet Non-IID"}
_CONFIG_LABEL   = {
    "A": "Traditional FL",
    "B": "Blockchain Baseline",
    "C": "CSRA Reward",
    "C-CSRA": "CSRA-DCD Reward",
    "C-CSRA-Opt": "CSRA-DCD Reward (Optimized)",
    "TrimmedMean": "TrimmedMean Robust FL",
}


def _parse_filename(fname: str) -> dict | None:
    """Return metadata dict from filename, or None if pattern doesn't match."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    alpha_raw = m.group("alpha_raw")
    dirichlet_raw = m.group("dirichlet_raw")
    dirichlet_alpha = (
        float(dirichlet_raw) / 100.0
        if dirichlet_raw is not None
        else (0.1 if m.group("scenario") == "K3" else 0.0)
    )
    return {
        "dataset":         m.group("dataset"),
        "scenario":        m.group("scenario"),
        "scenario_label":  _SCENARIO_LABEL.get(m.group("scenario"), m.group("scenario")),
        "config":          m.group("config"),
        "config_label":    _CONFIG_LABEL.get(m.group("config"), m.group("config")),
        "alpha":           _ALPHA_MAP.get(alpha_raw, float(alpha_raw) / 10),
        "dirichlet_alpha": dirichlet_alpha,
        "timestamp":       m.group("timestamp"),
        "run_id":          fname[: -len(".csv")],
    }


def load_all_logs(log_dir: Path) -> pd.DataFrame | None:
    """
    Scan *log_dir* recursively for CSV files, parse metadata from filenames,
    merge with CSV content, and return a single concatenated DataFrame.

    Columns added / overridden from filename:
        dataset, scenario, scenario_label, config, config_label,
        alpha, timestamp, run_id
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
            log.debug("Skipping (name mismatch): %s", fpath.name)
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

        # Preserve runtime alpha before filename metadata overrides it.
        # This keeps old group-by behavior while retaining dynamic-alpha traces.
        if "alpha" in df.columns and "alpha_runtime" not in df.columns:
            df["alpha_runtime"] = pd.to_numeric(df["alpha"], errors="coerce")

        # Attach / override metadata columns from filename (authoritative)
        for k, v in meta.items():
            df[k] = v

        frames.append(df)
        log.debug("Loaded %s (%d rows)", fpath.name, len(df))

    if not frames:
        log.error("All %d CSV files were skipped.", len(csv_files))
        return None

    log.info(
        "Loaded %d files (%d skipped). Merging …", len(frames), skipped
    )
    combined = pd.concat(frames, ignore_index=True)

    # ── Normalise column names ────────────────────────────────────────────────
    # The logger writes 'round' but requirements mention 'round_num'; support both.
    if "round" in combined.columns and "round_num" not in combined.columns:
        combined.rename(columns={"round": "round_num"}, inplace=True)
    if "quality_score" in combined.columns and "quality" not in combined.columns:
        combined.rename(columns={"quality_score": "quality"}, inplace=True)

    # ── Type coercions ────────────────────────────────────────────────────────
    for col in ["round_num", "client_id", "data_size"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    for col in [
        "quality", "w_new", "reputation", "reward_eth", "global_accuracy",
        "alpha", "alpha_runtime", "anomaly_score", "robust_z",
    ]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "is_honest" in combined.columns:
        combined["is_honest"] = combined["is_honest"].astype(bool)
    if "is_anomaly" in combined.columns:
        combined["is_anomaly"] = pd.to_numeric(combined["is_anomaly"], errors="coerce").fillna(0).astype(bool)

    # Detect attack runs (any malicious client present)
    if "client_type" in combined.columns:
        attack_run_ids = combined.loc[
            combined["client_type"].isin(["free_rider", "lazy"]), "run_id"
        ].unique()
        combined["has_attack"] = combined["run_id"].isin(attack_run_ids)

    combined.sort_values(["dataset", "scenario", "config", "alpha", "dirichlet_alpha", "round_num"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
