"""Tests for fl.logger — schema v2."""
import csv

from fl.logger import ExperimentLogger, make_run_id


def test_logger_preserves_zero_global_accuracy(tmp_path):
    """global_accuracy=0.0 phải được ghi là "0.0" chứ không phải "" (rỗng)."""
    logger = ExperimentLogger("run", str(tmp_path))
    logger.log_round(
        dataset="mnist",
        scenario="K1",
        aggregation_method="fedavg",
        reward_policy="equal",
        round_num=1,
        client_id=0,
        client_type="honest",
        quality=0.0,
        data_size=10,
        w_new=0.0,
        reputation=0.0,
        reward_eth=0.0,
        is_honest=True,
        ground_truth_honest=True,
        reward_eligible=True,
        global_accuracy=0.0,
    )
    logger.close()

    with open(tmp_path / "run.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["global_accuracy"] == "0.0"


def test_logger_writes_new_columns(tmp_path):
    """Logger phải có đủ cột schema v2 (agg, reward, β γ δ, attack, seed)."""
    logger = ExperimentLogger("run", str(tmp_path))
    logger.log_round(
        dataset="cifar10",
        scenario="K3",
        aggregation_method="csra_dcd",
        reward_policy="csra",
        beta=0.5,
        gamma=0.3,
        delta=0.2,
        mad_threshold=3.0,
        cosine_threshold=-0.8,
        direction_min_norm_z=1.0,
        min_honest_ratio=0.5,
        fallback_hard_z=6.0,
        suspicion_decay=0.6,
        suspicion_threshold=1.0,
        low_quality_z_threshold=2.0,
        low_quality_suspicion=0.5,
        zero_data_suspicion=1.0,
        anomaly_suspicion=0.8,
        authenticity_suspicion=1.2,
        low_authenticity_threshold=1.7,
        high_update_norm_z_threshold=4.5,
        inefficient_update_suspicion=0.9,
        attack_type="free_rider",
        seed=2024,
        persistent_clients=True,
        num_clients=10,
        num_rounds=20,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        client_fraction=1.0,
        data_split="K3",
        data_imbalance="lognormal",
        dirichlet_alpha=0.1,
        round_num=5,
        client_id=7,
        client_type="free_rider",
        quality=0.0,
        data_size=0,
        reported_data_size=0,
        server_known_data_size=128,
        w_new=0.0,
        reputation=0.1,
        reward_eth=0.0,
        reward_blocked=True,
        is_honest=False,
        ground_truth_honest=False,
        reward_eligible=False,
        anomaly_score=12.34,
        robust_z=4.56,
        is_anomaly=True,
        detection_reason="mad",
        direction_anomaly=True,
        cosine_to_reference=-0.75,
        risk_score=1.23,
        anomaly_score_source="server_data_normalized",
        raw_update_norm=24.0,
        raw_update_norm_z=4.75,
        normalized_update_score=12.34,
        data_commitment_anomaly=True,
        data_size_mismatch=True,
        low_quality_outlier=True,
        inefficient_update=True,
        suspicion_signal=1.5,
        suspicion_score=2.25,
        suspicion_quarantine=True,
        suspicion_reason="zero_data+low_quality",
        reward_component_quality=0.1,
        reward_component_data=0.2,
        reward_component_reputation=0.3,
        global_accuracy=0.85,
        global_loss=0.4321,
    )
    logger.close()

    with open(tmp_path / "run.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["aggregation_method"] == "csra_dcd"
    assert row["reward_policy"] == "csra"
    assert row["beta"] == "0.5"
    assert row["gamma"] == "0.3"
    assert row["delta"] == "0.2"
    assert row["mad_threshold"] == "3.0"
    assert row["cosine_threshold"] == "-0.8"
    assert row["direction_min_norm_z"] == "1.0"
    assert row["min_honest_ratio"] == "0.5"
    assert row["fallback_hard_z"] == "6.0"
    assert row["suspicion_decay"] == "0.6"
    assert row["suspicion_threshold"] == "1.0"
    assert row["low_quality_z_threshold"] == "2.0"
    assert row["low_quality_suspicion"] == "0.5"
    assert row["zero_data_suspicion"] == "1.0"
    assert row["anomaly_suspicion"] == "0.8"
    assert row["authenticity_suspicion"] == "1.2"
    assert row["low_authenticity_threshold"] == "1.7"
    assert row["high_update_norm_z_threshold"] == "4.5"
    assert row["inefficient_update_suspicion"] == "0.9"
    assert row["attack_type"] == "free_rider"
    assert row["seed"] == "2024"
    assert row["persistent_clients"] == "1"
    assert row["num_clients"] == "10"
    assert row["num_rounds"] == "20"
    assert row["local_epochs"] == "2"
    assert row["batch_size"] == "32"
    assert row["learning_rate"] == "0.01"
    assert row["client_fraction"] == "1.0"
    assert row["data_split"] == "K3"
    assert row["data_imbalance"] == "lognormal"
    assert row["dirichlet_alpha"] == "0.1"
    assert row["reward_blocked"] == "1"
    assert row["ground_truth_honest"] == "0"
    assert row["reward_eligible"] == "0"
    assert row["is_anomaly"] == "1"
    assert row["detection_reason"] == "mad"
    assert row["direction_anomaly"] == "1"
    assert row["cosine_to_reference"] == "-0.75"
    assert row["risk_score"] == "1.23"
    assert row["anomaly_score_source"] == "server_data_normalized"
    assert row["reported_data_size"] == "0"
    assert row["server_known_data_size"] == "128"
    assert row["raw_update_norm"] == "24.0"
    assert row["raw_update_norm_z"] == "4.75"
    assert row["normalized_update_score"] == "12.34"
    assert row["data_commitment_anomaly"] == "1"
    assert row["data_size_mismatch"] == "1"
    assert row["low_quality_outlier"] == "1"
    assert row["inefficient_update"] == "1"
    assert row["suspicion_signal"] == "1.5"
    assert row["suspicion_score"] == "2.25"
    assert row["suspicion_quarantine"] == "1"
    assert row["suspicion_reason"] == "zero_data+low_quality"
    assert row["reward_component_quality"] == "0.1"
    assert row["reward_component_data"] == "0.2"
    assert row["reward_component_reputation"] == "0.3"
    assert row["global_loss"] == "0.4321"


def test_make_run_id_schema_v2():
    """Filename phải theo format mới với đầy đủ thông tin."""
    rid = make_run_id(
        dataset="mnist",
        scenario="K1",
        aggregation_method="fedavg",
        reward_policy="equal",
        seed=42,
        beta=0.0, gamma=0.0, delta=0.0,
        attack_type="clean",
    )
    # Phải chứa các phần đặc trưng
    assert rid.startswith("mnist_K1_fedavg_equal_b00g00d00_s42_clean_")

    rid2 = make_run_id(
        dataset="cifar10",
        scenario="K3",
        aggregation_method="csra_dcd",
        reward_policy="csra",
        seed=2024,
        beta=0.5, gamma=0.3, delta=0.2,
        attack_type="free_rider",
        dirichlet_alpha=0.1,
    )
    assert rid2.startswith("cifar10_K3_da010_csra_dcd_csra_b50g30d20_s2024_free_rider_")
