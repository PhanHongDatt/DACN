# FL Reward - Schema v2 Experiment Report

**Generated:** 2026-06-12 00:20:40  
**Rows:** 3  
**Runs:** 1  
**Datasets:** mnist  
**Scenarios:** K1  
**Conditions:** label_noise=1  

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

| dataset | scenario_variant | method | method_label | jain | gini | fairness_gap | fairness_gap_alignment | reward_quality_corr | reward_alignment_corr | reward_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K1 | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 0.6923 | 0.3333 | 0.3242 | 0.3326 | 1.0000 | 1.0000 | 0.1111 |

## Reward Risk Diagnostics

Honest reward starvation counts honest participating clients that were not reward-blocked but still received approximately zero reward.

| dataset | scenario_variant | attack_type | method | method_label | mad_threshold | cosine_threshold | direction_min_norm_z | min_honest_ratio | fallback_hard_z | suspicion_decay | suspicion_threshold | low_quality_z_threshold | low_quality_suspicion | zero_data_suspicion | anomaly_suspicion | authenticity_suspicion | low_authenticity_threshold | high_update_norm_z_threshold | inefficient_update_suspicion | n_runs | reward_eligible_honest_rows | honest_reward_starvation_count | honest_reward_starvation_rate | reward_leakage | false_positive_quarantine_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K1 | label_noise | csra_dcd+csra | CSRA-DCD + CSRAReward | 3.0000 | -0.8000 | 0.0000 | 0.5000 | 6.0000 | 0.6000 | 1.0000 | 2.0000 | 0.5000 | 1.0000 | 0.8000 | 1.0000 | 1.5000 | 4.0000 | 1.0000 | 1 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Attack Robustness

| dataset | scenario | attack_type | method | method_label | mad_threshold | cosine_threshold | direction_min_norm_z | min_honest_ratio | fallback_hard_z | suspicion_decay | suspicion_threshold | low_quality_z_threshold | low_quality_suspicion | zero_data_suspicion | anomaly_suspicion | authenticity_suspicion | low_authenticity_threshold | high_update_norm_z_threshold | inefficient_update_suspicion | final_accuracy | reward_leakage | attack_detection_rate | attack_reward_block_rate | false_positive_detection_rate | false_positive_quarantine_rate | data_commitment_anomaly_rate | inefficient_update_rate | suspicion_quarantine_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K1 | label_noise | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 3.0000 | -0.8000 | 0.0000 | 0.5000 | 6.0000 | 0.6000 | 1.0000 | 2.0000 | 0.5000 | 1.0000 | 0.8000 | 1.0000 | 1.5000 | 4.0000 | 1.0000 | 0.9311 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.3333 |

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

### Mann-Whitney U (Unpaired)

_Insufficient runs for pairwise tests._

### Wilcoxon Signed-Rank (Paired by Seed)

_Insufficient same-seed pairs for Wilcoxon tests._
