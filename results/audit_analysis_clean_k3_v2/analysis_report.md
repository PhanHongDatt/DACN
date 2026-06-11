# FL Reward - Schema v2 Experiment Report

**Generated:** 2026-06-12 00:25:07  
**Rows:** 5  
**Runs:** 1  
**Datasets:** mnist  
**Scenarios:** K3 (dirichlet=0.1)  
**Conditions:** clean=1  

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

| dataset | scenario_variant | method | method_label | mad_threshold | cosine_threshold | direction_min_norm_z | min_honest_ratio | fallback_hard_z | suspicion_decay | suspicion_threshold | low_quality_z_threshold | low_quality_suspicion | zero_data_suspicion | anomaly_suspicion | authenticity_suspicion | low_authenticity_threshold | high_update_norm_z_threshold | inefficient_update_suspicion | final_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K3 (dirichlet=0.1) | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 3.0000 | -0.8000 | 0.0000 | 0.5000 | 6.0000 | 0.6000 | 1.0000 | 2.0000 | 0.5000 | 1.0000 | 0.8000 | 1.0000 | 1.5000 | 4.0000 | 1.0000 | 0.5297 |

## Reward Fairness

| dataset | scenario_variant | method | method_label | jain | gini | fairness_gap | fairness_gap_alignment | reward_quality_corr | reward_alignment_corr | reward_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K3 (dirichlet=0.1) | csra_dcd+csra | M6 CSRA-DCD + CSRAReward | 0.8620 | 0.2149 | 0.0554 | 0.0611 | 0.5996 | 0.9262 | 0.0064 |

## Reward Risk Diagnostics

Honest reward starvation counts honest participating clients that were not reward-blocked but still received approximately zero reward.

| dataset | scenario_variant | attack_type | method | method_label | mad_threshold | cosine_threshold | direction_min_norm_z | min_honest_ratio | fallback_hard_z | suspicion_decay | suspicion_threshold | low_quality_z_threshold | low_quality_suspicion | zero_data_suspicion | anomaly_suspicion | authenticity_suspicion | low_authenticity_threshold | high_update_norm_z_threshold | inefficient_update_suspicion | n_runs | reward_eligible_honest_rows | honest_reward_starvation_count | honest_reward_starvation_rate | reward_leakage | false_positive_quarantine_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | K3 (dirichlet=0.1) | clean | csra_dcd+csra | CSRA-DCD + CSRAReward | 3.0000 | -0.8000 | 0.0000 | 0.5000 | 6.0000 | 0.6000 | 1.0000 | 2.0000 | 0.5000 | 1.0000 | 0.8000 | 1.0000 | 1.5000 | 4.0000 | 1.0000 | 1 | 5 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Attack Robustness

_No attack runs found._

## Plots

- `baseline_accuracy_curve_mnist.png`
- `baseline_final_accuracy_mnist.png`
- `fairness_boxplot_mnist.png`
- `fairness_jain_gini_mnist.png`
- `fairness_reward_vs_quality_mnist.png`
- `convergence_round_mnist.png`
- `convergence_scatter_mnist.png`
- `beta_sensitivity_mnist.png`

## Statistical Tests

### Mann-Whitney U (Unpaired)

_Insufficient runs for pairwise tests._

### Wilcoxon Signed-Rank (Paired by Seed)

_Insufficient same-seed pairs for Wilcoxon tests._
