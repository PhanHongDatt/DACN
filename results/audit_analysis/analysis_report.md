# FL Reward - Schema v2 Experiment Report

**Generated:** 2026-06-11 17:08:39  
**Rows:** 10  
**Runs:** 2  
**Datasets:** mnist  
**Scenarios:** K1  
**Conditions:** stealth_free_rider=2  

## Method Matrix

Blockchain is treated as an audit and reward-distribution layer. It is not used as a separate experimental baseline.

| ID | Method | Role |
| --- | --- | --- |
| M1 | FedAvg + EqualSplit | Minimal baseline |
| M2 | FedAvg + DataSize | Data-quantity reward baseline |
| M3 | FedAvg + QualityOnly | One-round quality reward baseline |
| M4 | FedAvg + CSRAReward | Reward-formula ablation |
| M5 | CSRA-DCD + EqualSplit | Filtering-only ablation |
| M6 | CSRA-DCD + CSRAReward | Proposed full system |

## Clean Accuracy

_No rows._

## Reward Fairness

| dataset | scenario_variant | method | method_label | jain | gini | fairness_gap | reward_variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K1 | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 0.8262 | 0.2238 | 0.0555 | 0.0234 |

## Attack Robustness

| dataset | scenario | attack_type | method | method_label | mad_threshold | cosine_threshold | direction_min_norm_z | min_honest_ratio | fallback_hard_z | suspicion_decay | suspicion_threshold | low_quality_z_threshold | low_quality_suspicion | zero_data_suspicion | anomaly_suspicion | authenticity_suspicion | low_authenticity_threshold | final_accuracy | reward_leakage | attack_detection_rate | attack_reward_block_rate | false_positive_detection_rate | false_positive_quarantine_rate | data_commitment_anomaly_rate | suspicion_quarantine_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K1 | stealth_free_rider | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 3.0000 | -0.8000 | 0.0000 | 0.5000 | 6.0000 | 0.6000 | 1.0000 | 2.0000 | 0.5000 | 1.0000 | 0.8000 | 1.2000 | 1.7000 | 0.9413 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Plots

- `baseline_accuracy_curve_mnist.png`
- `baseline_final_accuracy_mnist.png`
- `fairness_boxplot_mnist.png`
- `fairness_jain_gini_mnist.png`
- `fairness_reward_vs_quality_mnist.png`
- `attack_accuracy_mnist.png`
- `attack_reward_share_mnist.png`
- `convergence_round_mnist.png`
- `convergence_scatter_mnist.png`
- `beta_sensitivity_mnist.png`

## Statistical Tests

_Insufficient runs for pairwise tests._
