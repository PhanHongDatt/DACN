import pytest

import pandas as pd

from analysis.stats import (
    compute_detection_confusion_metrics,
    compute_summary_metrics,
    paired_wilcoxon_tests,
    wilcoxon_per_run,
)


def _base_rows(include_reward_blocked: bool = True):
    rows = []
    client_types = ["honest", "honest", "sign_flip", "sign_flip"]
    anomalies = [False, False, True, False]
    blocked = [False, True, True, False]
    rewards = [0.4, 0.0, 0.0, 0.2]
    is_honest = [True, False, False, True] if not include_reward_blocked else [True] * 4

    for cid, ctype in enumerate(client_types):
        row = {
            "run_id": "run-1",
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "method": "csra_dcd+csra",
            "method_label": "CSRA-DCD + CSRAReward",
            "aggregation_method": "csra_dcd",
            "reward_policy": "csra",
            "beta": 0.5,
            "gamma": 0.3,
            "delta": 0.2,
            "dirichlet_alpha": 0.1,
            "attack_type": "sign_flip",
            "attack_label": "attack",
            "has_attack": True,
            "round_num": 1,
            "client_id": cid,
            "client_type": ctype,
            "is_malicious": ctype != "honest",
            "quality": 0.1,
            "data_size": 100,
            "reputation": 0.5,
            "reward_eth": rewards[cid],
            "is_honest": is_honest[cid],
            "is_anomaly": anomalies[cid],
            "detection_reason": "accepted",
            "global_accuracy": 0.8,
            "global_loss": 0.4,
        }
        if include_reward_blocked:
            row["reward_blocked"] = blocked[cid]
        rows.append(row)
    return rows


def test_summary_separates_detection_and_quarantine_rates():
    summary = compute_summary_metrics(pd.DataFrame(_base_rows()))
    row = summary.iloc[0]

    assert row["attack_detection_rate"] == pytest.approx(0.5)
    assert row["attack_reward_block_rate"] == pytest.approx(0.5)
    assert row["false_positive_detection_rate"] == pytest.approx(0.0)
    assert row["false_positive_quarantine_rate"] == pytest.approx(0.5)
    assert row["false_positive_rate"] == pytest.approx(0.0)
    assert row["final_loss"] == pytest.approx(0.4)
    assert row["min_loss"] == pytest.approx(0.4)
    assert row["false_positive_quarantine_count"] == 1
    assert row["attack_reward_block_count"] == 1
    assert row["reward_eligible_honest_rows"] == 1
    assert row["honest_reward_starvation_count"] == 0
    assert row["honest_reward_starvation_rate"] == pytest.approx(0.0)


def test_summary_and_confusion_parse_string_boolean_columns():
    rows = _base_rows()
    for row in rows:
        for col in ["is_malicious", "is_honest", "is_anomaly", "reward_blocked"]:
            row[col] = "True" if row[col] else "False"
        row["data_commitment_anomaly"] = "False"
        row["low_quality_outlier"] = "False"
        row["suspicion_quarantine"] = "False"
    rows[2]["data_commitment_anomaly"] = "True"
    rows[1]["suspicion_quarantine"] = "True"

    summary = compute_summary_metrics(pd.DataFrame(rows))
    summary_row = summary.iloc[0]

    assert summary_row["attack_detection_rate"] == pytest.approx(0.5)
    assert summary_row["attack_reward_block_rate"] == pytest.approx(0.5)
    assert summary_row["false_positive_detection_rate"] == pytest.approx(0.0)
    assert summary_row["false_positive_quarantine_rate"] == pytest.approx(0.5)
    assert summary_row["data_commitment_anomaly_count"] == 1
    assert summary_row["suspicion_quarantine_count"] == 1

    confusion = compute_detection_confusion_metrics(pd.DataFrame(rows))
    confusion_row = confusion.iloc[0]

    assert confusion_row["detection_tp"] == 1
    assert confusion_row["detection_fp"] == 0
    assert confusion_row["quarantine_tp"] == 1
    assert confusion_row["quarantine_fp"] == 1


def test_summary_infers_reward_blocked_for_old_logs_from_is_honest():
    summary = compute_summary_metrics(pd.DataFrame(_base_rows(include_reward_blocked=False)))
    row = summary.iloc[0]

    assert row["attack_detection_rate"] == pytest.approx(0.5)
    assert row["attack_reward_block_rate"] == pytest.approx(0.5)
    assert row["false_positive_detection_rate"] == pytest.approx(0.0)
    assert row["false_positive_quarantine_rate"] == pytest.approx(0.5)


def test_summary_prefers_reward_eligible_over_legacy_is_honest():
    rows = _base_rows(include_reward_blocked=False)
    for row in rows:
        row["is_honest"] = True
        row["reward_eligible"] = True
    rows[1]["reward_eligible"] = False

    summary = compute_summary_metrics(pd.DataFrame(rows))
    row = summary.iloc[0]

    assert row["false_positive_quarantine_count"] == 1
    assert row["false_positive_quarantine_rate"] == pytest.approx(0.5)


def test_summary_reward_ratio_uses_ground_truth_honest_not_client_type_string():
    rows = _base_rows()
    rows[0]["client_type"] = "rare_class_honest"
    rows[0]["is_malicious"] = False
    rows[0]["ground_truth_honest"] = "True"

    summary = compute_summary_metrics(pd.DataFrame(rows))
    row = summary.iloc[0]

    assert row["reward_ratio"] == pytest.approx(2.0)


def test_summary_tracks_honest_reward_starvation():
    rows = _base_rows()
    rows[0]["reward_eth"] = 0.0
    rows[0]["reward_blocked"] = False
    rows[1]["reward_eth"] = 0.0
    rows[1]["reward_blocked"] = True

    summary = compute_summary_metrics(pd.DataFrame(rows))
    row = summary.iloc[0]

    assert row["reward_eligible_honest_rows"] == 1
    assert row["honest_reward_starvation_count"] == 1
    assert row["honest_reward_starvation_rate"] == pytest.approx(1.0)


def test_summary_reports_alignment_reward_diagnostics():
    rows = _base_rows()
    for cid, row in enumerate(rows):
        row["quality"] = 1.0 - (cid * 0.1)
        row["cosine_to_reference"] = [0.9, 0.1, -0.5, 0.2][cid]
        row["reward_eth"] = [0.9, 0.1, 0.0, 0.0][cid]
        row["reward_blocked"] = cid >= 2

    summary = compute_summary_metrics(pd.DataFrame(rows))
    row = summary.iloc[0]

    assert row["reward_alignment_corr"] == pytest.approx(1.0)
    assert row["fairness_gap_alignment"] == pytest.approx(0.0)


def test_detection_confusion_metrics_are_per_round_traceable():
    confusion = compute_detection_confusion_metrics(pd.DataFrame(_base_rows()))
    row = confusion.iloc[0]

    assert row["round_num"] == 1
    assert row["n_malicious"] == 2
    assert row["n_honest"] == 2
    assert row["detection_tp"] == 1
    assert row["detection_fn"] == 1
    assert row["detection_fp"] == 0
    assert row["detection_tn"] == 2
    assert row["quarantine_tp"] == 1
    assert row["quarantine_fn"] == 1
    assert row["quarantine_fp"] == 1
    assert row["quarantine_tn"] == 1


def test_detection_confusion_tracks_fallback_modes():
    rows = _base_rows()
    rows[0]["detection_reason"] = "fallback_accept_all"
    rows[1]["detection_reason"] = "fallback_soft_accept"
    rows[2]["detection_reason"] = "norm_mad+fallback_hard_block"

    confusion = compute_detection_confusion_metrics(pd.DataFrame(rows))
    row = confusion.iloc[0]

    assert row["fallback_triggered_count"] == 3
    assert row["fallback_accept_all_count"] == 1
    assert row["fallback_soft_filter_count"] == 2


def test_detection_confusion_tracks_suspicion_and_reward_share():
    rows = _base_rows()
    rows[0]["data_commitment_anomaly"] = True
    rows[1]["low_quality_outlier"] = True
    rows[2]["inefficient_update"] = True
    rows[1]["suspicion_quarantine"] = True

    confusion = compute_detection_confusion_metrics(pd.DataFrame(rows))
    row = confusion.iloc[0]

    assert row["data_commitment_anomaly_count"] == 1
    assert row["low_quality_outlier_count"] == 1
    assert row["inefficient_update_count"] == 1
    assert row["inefficient_update_rate"] == pytest.approx(0.25)
    assert row["suspicion_quarantine_count"] == 1
    assert row["attacker_reward_share"] == pytest.approx(1 / 3)


def test_summary_keeps_authenticity_detector_params_separate():
    rows_a = _base_rows()
    rows_b = _base_rows()
    for row in rows_a:
        row["run_id"] = "run-auth-a"
        row["authenticity_suspicion"] = 1.0
        row["low_authenticity_threshold"] = 1.5
    for row in rows_b:
        row["run_id"] = "run-auth-b"
        row["authenticity_suspicion"] = 1.8
        row["low_authenticity_threshold"] = 2.5

    summary = compute_summary_metrics(pd.DataFrame(rows_a + rows_b))

    assert set(summary["authenticity_suspicion"]) == {1.0, 1.8}
    assert set(summary["low_authenticity_threshold"]) == {1.5, 2.5}
    assert len(summary) == 2


def test_summary_keeps_traceability_metadata_without_splitting_seeds():
    rows_a = _base_rows()
    rows_b = _base_rows()
    for row in rows_a:
        row.update({
            "run_id": "run-seed-42",
            "seed": 42,
            "num_clients": 4,
            "num_rounds": 2,
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "client_fraction": 1.0,
            "data_split": "K3",
            "data_imbalance": "lognormal",
            "persistent_clients": True,
        })
    for row in rows_b:
        row.update({
            "run_id": "run-seed-43",
            "seed": 43,
            "num_clients": 4,
            "num_rounds": 2,
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "client_fraction": 1.0,
            "data_split": "K3",
            "data_imbalance": "lognormal",
            "persistent_clients": True,
        })

    summary = compute_summary_metrics(pd.DataFrame(rows_a + rows_b))
    row = summary.iloc[0]

    assert len(summary) == 1
    assert row["n_runs"] == 2
    assert row["seed_count"] == 2
    assert row["seeds"] == "42,43"
    assert row["run_ids"] == "run-seed-42,run-seed-43"
    assert row["num_clients"] == 4
    assert row["num_rounds"] == 2
    assert row["local_epochs"] == 1
    assert row["batch_size"] == 32
    assert row["learning_rate"] == pytest.approx(0.01)
    assert row["client_fraction"] == pytest.approx(1.0)
    assert row["data_split"] == "K3"
    assert row["data_imbalance"] == "lognormal"
    assert bool(row["persistent_clients"]) is True


def _stat_row(run_id: str, method: str, seed: int, acc: float):
    return {
        "run_id": run_id,
        "dataset": "mnist",
        "scenario": "K3",
        "scenario_variant": "K3 (dirichlet=0.1)",
        "dirichlet_alpha": 0.1,
        "attack_type": "clean",
        "attack_label": "clean",
        "method": method,
        "seed": seed,
        "round_num": 1,
        "global_accuracy": acc,
    }


def test_wilcoxon_pairs_only_same_seed_and_condition():
    df = pd.DataFrame([
        _stat_row("a-s1", "fedavg+equal", 1, 0.50),
        _stat_row("a-s2", "fedavg+equal", 2, 0.60),
        _stat_row("b-s1", "csra_dcd+csra", 1, 0.70),
        _stat_row("b-s2", "csra_dcd+csra", 2, 0.80),
        _stat_row("b-s99", "csra_dcd+csra", 99, 0.99),
    ])

    result = wilcoxon_per_run(df, "fedavg+equal", "csra_dcd+csra")

    assert result["n_pairs"] == 2
    assert result["skipped_unpaired"] == 1
    assert result["skipped_ambiguous"] == 0
    assert "seed" in result["paired_on"]
    assert result["mean_a"] == pytest.approx(0.55)
    assert result["mean_b"] == pytest.approx(0.75)
    assert result["mean_diff"] == pytest.approx(0.20)


def test_wilcoxon_skips_ambiguous_same_seed_pairs():
    df = pd.DataFrame([
        _stat_row("a-s1", "fedavg+equal", 1, 0.50),
        _stat_row("b-s1-a", "csra_dcd+csra", 1, 0.70),
        _stat_row("b-s1-b", "csra_dcd+csra", 1, 0.90),
    ])

    result = wilcoxon_per_run(df, "fedavg+equal", "csra_dcd+csra")

    assert result["n_pairs"] == 0
    assert result["skipped_unpaired"] == 0
    assert result["skipped_ambiguous"] == 1
    assert result["significant"] is False


def test_paired_wilcoxon_tests_runs_all_method_pairs():
    df = pd.DataFrame([
        _stat_row("a-s1", "fedavg+equal", 1, 0.50),
        _stat_row("a-s2", "fedavg+equal", 2, 0.60),
        _stat_row("b-s1", "fedavg+data", 1, 0.55),
        _stat_row("b-s2", "fedavg+data", 2, 0.65),
        _stat_row("c-s1", "csra_dcd+csra", 1, 0.70),
        _stat_row("c-s2", "csra_dcd+csra", 2, 0.80),
    ])

    result = paired_wilcoxon_tests(df)

    assert len(result) == 3
    assert set(result["n_pairs"]) == {2}
    assert set(result["config_a"]).issubset({
        "fedavg+equal", "fedavg+data", "csra_dcd+csra",
    })
    assert result["paired_on"].str.contains("seed").all()
