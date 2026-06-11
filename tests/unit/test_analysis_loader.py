"""Tests for analysis/loader.py — schema v2."""
import pandas as pd

from analysis.loader import _coerce_bool_series, _parse_filename, load_all_logs


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
                "num_clients": len(client_types),
                "num_rounds": 2,
                "local_epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.01,
                "client_fraction": 1.0,
                "data_split": "K3",
                "data_imbalance": "lognormal",
                "round": round_num,
                "client_id": cid,
                "client_type": ctype,
                "quality_score": 0.1,
                "data_size": 100,
                "reported_data_size": 100,
                "server_known_data_size": 100,
                "w_new": 0.5,
                "reputation": 1.0,
                "reward_eth": 0.5,
                "reward_blocked": 1 if ctype != "honest" else 0,
                "ground_truth_honest": 1 if ctype == "honest" else 0,
                "reward_eligible": 1 if ctype == "honest" else 0,
                "is_honest": 1,
                "anomaly_score": 0.0,
                "robust_z": 0.0,
                "is_anomaly": 0,
                "detection_reason": "accepted",
                "raw_update_norm": 1.0,
                "raw_update_norm_z": 0.0,
                "normalized_update_score": 1.0,
                "authenticity_score": 0.8,
                "alignment_score": 0.7,
                "simplex_weight": 0.25,
                "data_size_mismatch": 0,
                "inefficient_update": 0,
                "reward_component_quality": 0.1,
                "reward_component_data": 0.2,
                "reward_component_reputation": 0.3,
                "global_accuracy": 0.8,
                "global_loss": 0.4,
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

    def test_parses_stealth_free_rider_attack(self):
        meta = _parse_filename(
            "mnist_K3_da010_csra_dcd_csra_b50g30d20_s42_stealth_free_rider_20260611_120000.csv"
        )
        assert meta is not None
        assert meta["attack_type"] == "stealth_free_rider"

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
        assert df["reward_blocked"].dtype == bool
        assert df["ground_truth_honest"].dtype == bool
        assert df["reward_eligible"].dtype == bool
        assert df["data_size_mismatch"].dtype == bool
        assert df["inefficient_update"].dtype == bool
        assert "raw_update_norm" in df.columns
        assert "raw_update_norm_z" in df.columns
        assert df["num_clients"].iloc[0] == 2
        assert df["num_rounds"].iloc[0] == 2
        assert df["learning_rate"].iloc[0] == 0.01
        assert df["client_fraction"].iloc[0] == 1.0
        assert df["global_loss"].iloc[0] == 0.4
        assert df["authenticity_score"].iloc[0] == 0.8
        assert df["alignment_score"].iloc[0] == 0.7
        assert df["simplex_weight"].iloc[0] == 0.25
        # method label tổng hợp
        assert "csra_dcd+csra" in set(df["method"])

    def test_load_old_logs_derives_explicit_honest_and_eligibility_columns(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest", "sign_flip"], agg="fedavg", reward="equal")
        df_raw = pd.read_csv(path)
        df_raw = df_raw.drop(columns=["ground_truth_honest", "reward_eligible"])
        df_raw.loc[df_raw["client_type"] == "sign_flip", "is_honest"] = 0
        df_raw.to_csv(path, index=False)

        df = load_all_logs(tmp_path)

        assert df is not None
        by_type = df.groupby("client_type").first()
        assert bool(by_type.loc["honest", "ground_truth_honest"]) is True
        assert bool(by_type.loc["sign_flip", "ground_truth_honest"]) is False
        assert bool(by_type.loc["honest", "reward_eligible"]) is True
        assert bool(by_type.loc["sign_flip", "reward_eligible"]) is False

    def test_ground_truth_honest_overrides_client_type_for_malicious_flag(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest"], agg="fedavg", reward="equal")
        df_raw = pd.read_csv(path)
        df_raw.loc[0, "client_type"] = "sign_flip"
        df_raw.loc[0, "ground_truth_honest"] = 1
        df_raw.to_csv(path, index=False)

        df = load_all_logs(tmp_path)

        assert df is not None
        assert bool(df.iloc[0]["ground_truth_honest"]) is True
        assert bool(df.iloc[0]["is_malicious"]) is False

    def test_missing_ground_truth_falls_back_to_client_type(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest", "sign_flip"], agg="fedavg", reward="equal")
        df_raw = pd.read_csv(path)
        df_raw.loc[df_raw["client_type"] == "sign_flip", "ground_truth_honest"] = pd.NA
        df_raw.to_csv(path, index=False)

        df = load_all_logs(tmp_path)

        assert df is not None
        by_type = df.groupby("client_type").first()
        assert bool(by_type.loc["sign_flip", "is_malicious"]) is True
        assert bool(by_type.loc["sign_flip", "ground_truth_honest"]) is False

    def test_bool_columns_accept_true_false_strings(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest", "sign_flip"], agg="fedavg", reward="equal")
        df_raw = pd.read_csv(path)
        for col in ["ground_truth_honest", "reward_eligible", "reward_blocked"]:
            df_raw[col] = df_raw[col].astype("object")
        df_raw.loc[df_raw["client_type"] == "honest", "ground_truth_honest"] = "True"
        df_raw.loc[df_raw["client_type"] == "sign_flip", "ground_truth_honest"] = "False"
        df_raw.loc[df_raw["client_type"] == "honest", "reward_eligible"] = "True"
        df_raw.loc[df_raw["client_type"] == "sign_flip", "reward_eligible"] = "False"
        df_raw.loc[df_raw["client_type"] == "sign_flip", "reward_blocked"] = "True"
        df_raw.to_csv(path, index=False)

        df = load_all_logs(tmp_path)

        assert df is not None
        by_type = df.groupby("client_type").first()
        assert bool(by_type.loc["honest", "ground_truth_honest"]) is True
        assert bool(by_type.loc["sign_flip", "ground_truth_honest"]) is False
        assert bool(by_type.loc["honest", "reward_eligible"]) is True
        assert bool(by_type.loc["sign_flip", "reward_eligible"]) is False
        assert bool(by_type.loc["sign_flip", "reward_blocked"]) is True

    def test_method_column_built(self, tmp_path):
        path = tmp_path / "mnist_K1_fedavg_equal_b00g00d00_s42_clean_20260520_120000.csv"
        _write_log(path, ["honest"], agg="fedavg", reward="equal")
        df = load_all_logs(tmp_path)
        assert df is not None
        assert (df["method"] == "fedavg+equal").all()
        assert (df["method_label"] == "FedAvg + EqualSplit").all()


def test_coerce_bool_series_can_preserve_missing_values():
    parsed = _coerce_bool_series(
        pd.Series(["True", "false", "1", "0", pd.NA]),
        preserve_na=True,
    )

    assert parsed.iloc[:4].tolist() == [True, False, True, False]
    assert pd.isna(parsed.iloc[4])
