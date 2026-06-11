import pandas as pd
import pytest

from analysis.report import (
    _attack_table,
    _reward_risk_table,
    generate_markdown_report,
)


def test_attack_table_reports_detection_and_quarantine_rates():
    df = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "honest",
            "is_malicious": False,
            "is_anomaly": False,
            "reward_blocked": True,
            "reward_eth": 0.0,
            "detection_reason": "reward_quarantine_direction",
            "authenticity_suspicion": 1.2,
            "low_authenticity_threshold": 1.7,
        },
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "sign_flip",
            "is_malicious": True,
            "is_anomaly": True,
            "reward_blocked": True,
            "reward_eth": 0.0,
            "detection_reason": "direction_cosine",
            "authenticity_suspicion": 1.2,
            "low_authenticity_threshold": 1.7,
        },
    ])

    table = _attack_table(df)
    row = table.iloc[0]

    assert row["attack_detection_rate"] == pytest.approx(1.0)
    assert row["attack_reward_block_rate"] == pytest.approx(1.0)
    assert row["false_positive_detection_rate"] == pytest.approx(0.0)
    assert row["false_positive_quarantine_rate"] == pytest.approx(1.0)
    assert row["authenticity_suspicion"] == pytest.approx(1.2)
    assert row["low_authenticity_threshold"] == pytest.approx(1.7)


def test_attack_table_parses_string_boolean_columns():
    df = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "honest",
            "is_malicious": "False",
            "is_anomaly": "False",
            "reward_blocked": "True",
            "data_commitment_anomaly": "False",
            "inefficient_update": "False",
            "suspicion_quarantine": "False",
            "reward_eth": 0.0,
            "detection_reason": "reward_quarantine_direction",
        },
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "sign_flip",
            "is_malicious": "True",
            "is_anomaly": "True",
            "reward_blocked": "True",
            "data_commitment_anomaly": "True",
            "inefficient_update": "True",
            "suspicion_quarantine": "True",
            "reward_eth": 0.0,
            "detection_reason": "direction_cosine",
        },
    ])

    table = _attack_table(df)
    row = table.iloc[0]

    assert row["attack_detection_rate"] == pytest.approx(1.0)
    assert row["attack_reward_block_rate"] == pytest.approx(1.0)
    assert row["false_positive_detection_rate"] == pytest.approx(0.0)
    assert row["false_positive_quarantine_rate"] == pytest.approx(1.0)
    assert row["data_commitment_anomaly_rate"] == pytest.approx(0.5)
    assert row["inefficient_update_rate"] == pytest.approx(0.5)
    assert row["suspicion_quarantine_rate"] == pytest.approx(0.5)


def test_attack_table_uses_ground_truth_honest_without_is_malicious():
    df = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "rare_class_honest",
            "ground_truth_honest": "True",
            "is_anomaly": False,
            "reward_eligible": True,
            "reward_eth": 0.4,
            "detection_reason": "accepted",
        },
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "sign_flip",
            "has_attack": True,
            "method": "csra_dcd+csra",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
            "client_type": "sign_flip",
            "ground_truth_honest": "False",
            "is_anomaly": True,
            "reward_eligible": False,
            "reward_eth": 0.0,
            "detection_reason": "direction_cosine",
        },
    ])

    table = _attack_table(df)
    row = table.iloc[0]

    assert row["attack_detection_rate"] == pytest.approx(1.0)
    assert row["attack_reward_block_rate"] == pytest.approx(1.0)
    assert row["false_positive_detection_rate"] == pytest.approx(0.0)


def test_reward_risk_table_reports_honest_reward_starvation():
    summary = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "clean",
            "attack_label": "clean",
            "method": "fedavg+quality",
            "method_label": "FedAvg + QualityOnly",
            "n_runs": 2,
            "reward_eligible_honest_rows": 20,
            "honest_reward_starvation_count": 3,
            "honest_reward_starvation_rate": 0.15,
            "reward_leakage": 0.0,
            "false_positive_quarantine_rate": 0.05,
        }
    ])

    table = _reward_risk_table(summary)
    row = table.iloc[0]

    assert row["method"] == "fedavg+quality"
    assert row["reward_eligible_honest_rows"] == 20
    assert row["honest_reward_starvation_count"] == 3
    assert row["honest_reward_starvation_rate"] == pytest.approx(0.15)


def test_markdown_report_includes_reward_risk_diagnostics(tmp_path):
    df = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario": "K3",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "clean",
            "has_attack": False,
            "method": "fedavg+quality",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
        }
    ])
    summary = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "attack_type": "clean",
            "attack_label": "clean",
            "method": "fedavg+quality",
            "method_label": "FedAvg + QualityOnly",
            "n_runs": 1,
            "reward_eligible_honest_rows": 10,
            "honest_reward_starvation_count": 2,
            "honest_reward_starvation_rate": 0.2,
            "reward_leakage": 0.0,
            "false_positive_quarantine_rate": 0.0,
        }
    ])
    fairness = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario_variant": "K3 (dirichlet=0.1)",
            "method": "fedavg+quality",
            "jain": 0.9,
            "gini": 0.1,
            "fairness_gap": 0.2,
            "fairness_gap_alignment": 0.05,
            "reward_quality_corr": 0.7,
            "reward_alignment_corr": 0.9,
            "reward_variance": 0.01,
        }
    ])
    report_path = tmp_path / "analysis_report.md"

    generate_markdown_report(
        df=df,
        summary=summary,
        fairness=fairness,
        report_path=report_path,
        plot_dir=tmp_path,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Reward Risk Diagnostics" in text
    assert "honest_reward_starvation_rate" in text
    assert "fairness_gap_alignment" in text
    assert "reward_alignment_corr" in text
    assert "0.2000" in text


def test_markdown_report_includes_paired_wilcoxon_table(tmp_path):
    df = pd.DataFrame([
        {
            "dataset": "mnist",
            "scenario": "K1",
            "scenario_variant": "K1",
            "attack_type": "clean",
            "has_attack": False,
            "method": "fedavg+equal",
            "run_id": "run-1",
            "round_num": 1,
            "global_accuracy": 0.8,
        }
    ])
    summary = pd.DataFrame()
    fairness = pd.DataFrame()
    paired = pd.DataFrame([
        {
            "config_a": "fedavg+equal",
            "config_b": "csra_dcd+csra",
            "n_pairs": 3,
            "paired_on": "dataset,scenario,seed",
            "skipped_unpaired": 0,
            "skipped_ambiguous": 0,
            "mean_a": 0.8,
            "mean_b": 0.9,
            "mean_diff": 0.1,
            "p_value": 0.25,
            "statistic": 0.0,
            "significant": False,
        }
    ])
    report_path = tmp_path / "analysis_report.md"

    generate_markdown_report(
        df=df,
        summary=summary,
        fairness=fairness,
        report_path=report_path,
        plot_dir=tmp_path,
        paired_stat_tests=paired,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "### Wilcoxon Signed-Rank (Paired by Seed)" in text
    assert "paired_on" in text
    assert "dataset,scenario,seed" in text
