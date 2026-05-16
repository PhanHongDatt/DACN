from analysis.loader import _parse_filename
from analysis.loader import load_all_logs


def _write_log(path, client_types):
    rows = []
    for round_num in [1, 2]:
        for cid, ctype in enumerate(client_types):
            rows.append({
                "run_id": path.stem,
                "dataset": "mnist",
                "scenario": "K3",
                "config": "C-CSRA-Opt",
                "alpha": 0.5,
                "dirichlet_alpha": 0.1,
                "alpha_runtime": 0.5,
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

    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_trimmed_mean_log_filename():
    meta = _parse_filename("mnist_K3_TrimmedMean_a00_da010_20260514_120000.csv")

    assert meta["dataset"] == "mnist"
    assert meta["scenario"] == "K3"
    assert meta["config"] == "TrimmedMean"
    assert meta["dirichlet_alpha"] == 0.1


def test_load_logs_adds_attack_label_and_scenario_variant(tmp_path):
    clean = tmp_path / "mnist_K3_C-CSRA_a05_da010_20260514_120000.csv"
    attack = tmp_path / "mnist_K3_C-CSRA_a05_da010_20260514_130000.csv"
    _write_log(clean, ["honest", "honest"])
    _write_log(attack, ["honest", "free_rider"])

    df = load_all_logs(tmp_path)

    assert set(df["attack_label"]) == {"clean", "attack"}
    assert set(df["scenario_variant"]) == {"K3 (dirichlet=0.1)"}
    assert df[df["attack_label"] == "attack"]["is_malicious"].any()
