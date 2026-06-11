"""style.py - Shared matplotlib style constants for schema v2."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Schema v2 method labels. Blockchain is an audit layer, not a comparison axis.
METHOD_LABELS = {
    "fedavg+equal": "M1 FedAvg + EqualSplit",
    "fedavg+data": "M2 FedAvg + DataSize",
    "fedavg+quality": "M3 FedAvg + QualityOnly",
    "fedavg+csra": "M4 FedAvg + CSRAReward",
    "csra_dcd+equal": "M5 CSRA-DCD + EqualSplit",
    "csra_dcd+csra": "M6 CSRA-DCD + CSRAReward",
}

METHOD_COLORS = {
    "fedavg+equal": "#4E79A7",
    "fedavg+data": "#59A14F",
    "fedavg+quality": "#F28E2B",
    "fedavg+csra": "#B07AA1",
    "csra_dcd+equal": "#E15759",
    "csra_dcd+csra": "#2F7F6F",
}

METHOD_ORDER = [
    "fedavg+equal",
    "fedavg+data",
    "fedavg+quality",
    "fedavg+csra",
    "csra_dcd+equal",
    "csra_dcd+csra",
]

# Backward-compatible aliases used by existing plotting code.
CONFIG_COLORS = METHOD_COLORS
CONFIG_LABELS = METHOD_LABELS
CONFIG_ORDER = METHOD_ORDER

# Legacy constants kept only to parse/read old artifacts if needed.
LEGACY_CONFIG_COLORS = {
    "A": "#E07070",
    "B": "#4A90D9",
    "C": "#5CB85C",
    "C-CSRA": "#2E7D32",
    "C-CSRA-Opt": "#1B5E20",
    "TrimmedMean": "#8E6BBE",
}
LEGACY_CONFIG_LABELS = {
    "A": "Traditional FL",
    "B": "Legacy B",
    "C": "Legacy CSRA Reward",
    "C-CSRA": "Legacy CSRA-DCD Reward",
    "C-CSRA-Opt": "Legacy CSRA-DCD Reward (Optimized)",
    "TrimmedMean": "TrimmedMean Robust FL",
}

CONFIG_COLORS.update(LEGACY_CONFIG_COLORS)
CONFIG_LABELS.update(LEGACY_CONFIG_LABELS)

ALPHA_COLORS = {
    0.0: "#9E9E9E", 0.3: "#42A5F5", 0.5: "#66BB6A",
    0.7: "#FFA726", 1.0: "#EF5350",
}
TYPE_COLORS = {
    "honest": "#4A90D9",
    "free_rider": "#E07070",
    "stealth_free_rider": "#B22222",
    "lazy": "#FFA726",
    "label_noise": "#8E6BBE",
    "sign_flip": "#D95F02",
}


def ordered_configs(configs):
    """Return configs in report-friendly baseline order."""
    present = list(dict.fromkeys(configs))
    known = [cfg for cfg in CONFIG_ORDER if cfg in present]
    extra = sorted(cfg for cfg in present if cfg not in CONFIG_ORDER)
    return known + extra


def apply_style():
    """Apply a clean academic style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
    })


def moving_avg(series, window: int):
    """Return moving average of a pandas Series."""
    return series.rolling(window, min_periods=1, center=True).mean()


def ci95(series_group):
    """Return (mean, lower, upper) 95% CI across a grouped series."""
    mean = series_group.mean()
    std  = series_group.std()
    n    = series_group.count()
    margin = 1.96 * std / np.sqrt(n.clip(lower=1))
    return mean, mean - margin, mean + margin


def save_fig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    import logging
    logging.getLogger(__name__).info("Saved: %s", path)
