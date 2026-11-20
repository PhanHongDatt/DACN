#!/bin/bash

DATASETS=("mnist" "fashion_mnist")
ATTACKS=("clean" "free_rider" "stealth_free_rider" "lazy" "label_noise" "sign_flip")
SEED=42
ROUNDS=20

echo "B?t d?u ch?y 24 th?c nghi?m v?i Blockchain b?t..."

for d in "${DATASETS[@]}"; do
    for a in "${ATTACKS[@]}"; do
        # Baseline 1: FedAvg + Equal
        echo -e "\n>>> Ðang ch?y: $d | $a | FedAvg + Equal..."
        python -m experiments.run_experiment --dataset $d --scenario K2 --aggregation fedavg --reward-policy equal --attack $a --n-rounds $ROUNDS --seed $SEED

        # Proposed: CSRA-DCD + CSRA
        echo -e "\n>>> Ðang ch?y: $d | $a | CSRA-DCD + CSRA..."
        python -m experiments.run_experiment --dataset $d --scenario K2 --aggregation csra_dcd --reward-policy csra --attack $a --n-rounds $ROUNDS --seed $SEED --mad-threshold 3.0 --cosine-threshold -0.8
    done
done
echo -e "\nHoàn thành 24 k?ch b?n!"
