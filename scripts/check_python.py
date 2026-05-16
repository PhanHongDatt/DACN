"""
check_python.py — Kiểm tra môi trường Python trước khi chạy.
Chạy: python scripts/check_python.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

issues = []
ok     = []

# 1. Python version
# Note (schema v2): chúng ta dùng simulation_local thay vì Flower's ray-based
# simulation → KHÔNG bắt buộc Python 3.11. 3.9+ là đủ; Python 3.11 vẫn là
# khuyến nghị cho ổn định nhất.
if sys.version_info < (3, 9):
    issues.append(
        f"Python ≥ 3.9 required, got {sys.version_info.major}.{sys.version_info.minor}."
    )
elif sys.version_info < (3, 11) or sys.version_info >= (3, 13):
    ok.append(
        f"Python {sys.version_info.major}.{sys.version_info.minor} "
        f"(khuyến nghị: 3.11)"
    )
else:
    ok.append(f"Python {sys.version_info.major}.{sys.version_info.minor}")

# 2. Required packages (bỏ ray — simulation_local không cần)
packages = {
    "flwr":        "flwr",
    "torch":       "torch",
    "torchvision": "torchvision",
    "web3":        "web3",
    "numpy":       "numpy",
    "pandas":      "pandas",
    "scipy":       "scipy",
    "matplotlib":  "matplotlib",
}
for name, pkg in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        ok.append(f"{name}=={ver}")
    except ImportError:
        issues.append(f"Missing: {name}  (pip install {pkg})")

# 3. Flower API (chỉ cần strategy + types, không cần simulation)
try:
    import flwr as fl
    from flwr.common import FitRes, EvaluateRes, Parameters  # noqa: F401
    from flwr.server.client_proxy import ClientProxy  # noqa: F401
    ok.append("Flower core API available (using simulation_local)")
except Exception as e:
    issues.append(f"Flower API issue: {e}")

# 4. PyTorch sanity
try:
    import torch
    import torch.nn as nn
    m = nn.Linear(10, 5)
    x = torch.randn(3, 10)
    _ = m(x)
    ok.append(f"PyTorch forward pass OK (device={'cuda' if torch.cuda.is_available() else 'cpu'})")
except Exception as e:
    issues.append(f"PyTorch issue: {e}")

# 5. Internal modules (schema v2 — new unified pipeline)
internal = [
    "fl.config", "fl.normalization", "fl.metrics",
    "fl.models", "fl.data_utils", "fl.logger",
    # Schema v2 modules
    "fl.reward_policies", "fl.aggregation_methods",
    "fl.client_attacks", "fl.server_base", "fl.simulation_local",
]
for mod in internal:
    try:
        __import__(mod)
        ok.append(f"{mod} importable")
    except Exception as e:
        issues.append(f"Cannot import {mod}: {e}")

# 6. Normalization sanity test
try:
    import numpy as np
    from fl.normalization import hybrid_normalize, compute_w_new
    v = np.array([6000.0] * 10)
    r = hybrid_normalize(v, mean_val=6000.0)
    assert not any(np.isnan(r)), "NaN in IID case"
    assert not any(np.isinf(r)), "Inf in IID case"

    q = np.array([0.8, 0.3, 0.5, 0.6, 0.7, 0.4, 0.9, 0.0, 0.1, 0.2])
    d = np.array([5000,3000,8000,2000,6000,4000,7000,0,100,500])
    w = compute_w_new(q, d, alpha=0.5, mean_data_size=3660.0)
    assert abs(w.sum() - 1.0) < 1e-5, f"W_new sum={w.sum()}"
    assert all(wi >= 0 for wi in w), "Negative W_new"
    ok.append("Normalization tests passed")
except Exception as e:
    issues.append(f"Normalization test failed: {e}")

# 7. Metrics sanity test
try:
    from fl.metrics import fairness_gap, jain_index, abc_metric
    r = np.array([1.0, 1.0, 1.0])
    c = np.array([1.0, 1.0, 1.0])
    assert fairness_gap(r, c) < 1e-6
    assert abs(jain_index(r) - 1.0) < 1e-6
    ok.append("Metrics tests passed")
except Exception as e:
    issues.append(f"Metrics test failed: {e}")

# ── Summary ──────────────────────────────────────────────────
print("\n=== Python Environment Check ===")
# Chỉ in summary, không in từng OK chi tiết
print(f"  ✓ {len(ok)} checks passed")
if issues:
    for iss in issues:
        print(f"  ✗ {iss}")
    print(f"\n  {len(issues)} issue(s). Fix before running.\n")
    sys.exit(1)
else:
    print("\n  All checks passed. Ready.\n")
