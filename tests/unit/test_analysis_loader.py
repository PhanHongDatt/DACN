"""Tests for analysis/loader.py — schema v2."""
import pandas as pd

from analysis.loader import _parse_filename, load_all_logs


def _write_log(path, client_types, attack_type="clean", agg="csra_dcd", reward="csra"):
    rows = []
    for round_num in [1, 2]:
        for cid, ctype in enumerate(client_types):
            rows.append({
                "run_id": path.stem,
                "dataset": "mnist",
                "scenario": "K3",
                "dirichlet_alpha": 0.1,
                "aggregation_method": agg,
                "reward_policy": reward,
                "beta": 0.5,
                "gamma": 0.3,
                "delta": 0.2,
                "attack_type": attack_type,
                "seed": 42,
                "round": round_num,
                "client_id": cid,
                "client_type": ctype,
                "quality_score": 0.1,
                "data_size": 100,
                "w_new": 0.5,
                "reputation": 1.0,
                "reward_eth": 0.5,
                "is_honest": 1,
                "anomaly_score": 0.0,
                "robust_z": 0.0,
                "is_anomaly": 0,
                "detection_reason": "accepted",
                "global_accuracy": 0.8,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


class TestParseFilename:
    def test_parses_basic_v2_filename(self):
        meta = _parse_filename(
            "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_143022.csv"
        )
        assert meta is not None
        assert meta["dataset"] == "mnist"
        assert meta["scenario"] == "K1"
        assert meta["aggregation_method"] == "fedavg"
        assert meta["reward_policy"] == "equal"
        assert meta["beta"] == 0.0
        assert meta["seed"] == 42
        assert meta["attack_type"] == "clean"

    def test_parses_csra_dcd_with_dirichlet(self):
        meta = _parse_filename(
            "cifar10_K3_da010_csra_dcd_csra_b50g30d20_s2024_free_rider_20260521_091533.csv"
        )
        assert meta is not None
        assert meta["dataset"] == "cifar10"
        assert meta["aggregation_method"] == "csra_dcd"  # gồm underscore
        assert meta["reward_policy"] == "csra"
        assert meta["beta"] == 0.5
        assert meta["gamma"] == 0.3
        assert meta["delta"] == 0.2
        assert meta["dirichlet_alpha"] == 0.1
        assert meta["attack_type"] == "free_rider"

    def test_parses_trimmed_baseline(self):
        meta = _parse_filename(
            "fashion_mnist_K2_trimmed_equal_b00g00d00_s123_clean_20260601_100000.csv"
        )
        assert meta is not None
        assert meta["dataset"] == "fashion_mnist"
        assert meta["aggregation_method"] == "trimmed"
        assert meta["reward_policy"] == "equal"

    def test_rejects_old_schema(self):
        # Old filename format không nên match
        meta = _parse_filename("mnist_K3_C-CSRA_a05_da010_20260514_120000.csv")
        assert meta is None

    def test_rejects_garbage(self):
        assert _parse_filename("random_garbage.csv") is None


class TestLoadAllLogs:
    def test_load_v2_logs(self, tmp_path):
        clean_path = tmp_path / "mnist_K3_da010_csra_dcd_csra_b50g30d20_s42_clean_20260520_120000.csv"
        attack_path = tmp_path / "mnist_K3_da010_csra_dcd_csra_b50g30d20_s42_free_rider_20260520_130000.csv"
        _write_log(clean_path, ["honest", "honest"], attack_type="clean")
        _write_log(attack_path, ["honest", "free_rider"], attack_type="free_rider")

        df = load_all_logs(tmp_path)
        assert df is not None
        assert set(df["attack_label"]) == {"clean", "attack"}
        assert set(df["scenario_variant"]) == {"K3 (dirichlet=0.1)"}
        assert df[df["attack_label"] == "attack"]["is_malicious"].any()
        # method label tổng hợp
        assert "csra_dcd+csra" in set(df["method"])

    def test_method_column_built(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest"], agg="fedavg", reward="equal")
        df = load_all_logs(tmp_path)
        assert df is not None
        assert (df["method"] == "fedavg+equal").all()
        assert (df["method_label"] == "FedAvg + EqualSplit").all()
