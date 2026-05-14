import csv

from fl.logger import ExperimentLogger


def test_logger_preserves_zero_global_accuracy(tmp_path):
    logger = ExperimentLogger("run", str(tmp_path))
    logger.log_round(
        dataset="mnist",
        scenario="K1",
        config="A",
        alpha=0.5,
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
