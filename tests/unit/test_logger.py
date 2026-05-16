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
        attack_type="free_rider",
        seed=2024,
        dirichlet_alpha=0.1,
        round_num=5,
        client_id=7,
        client_type="free_rider",
        quality=0.0,
        data_size=0,
        w_new=0.0,
        reputation=0.1,
        reward_eth=0.0,
        is_honest=False,
        anomaly_score=12.34,
        robust_z=4.56,
        is_anomaly=True,
        detection_reason="mad",
        global_accuracy=0.85,
    )
    logger.close()

    with open(tmp_path / "run.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["aggregation_method"] == "csra_dcd"
    assert row["reward_policy"] == "csra"
    assert row["beta"] == "0.5"
    assert row["gamma"] == "0.3"
    assert row["delta"] == "0.2"
    assert row["attack_type"] == "free_rider"
    assert row["seed"] == "2024"
    assert row["dirichlet_alpha"] == "0.1"
    assert row["is_anomaly"] == "1"
    assert row["detection_reason"] == "mad"


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
