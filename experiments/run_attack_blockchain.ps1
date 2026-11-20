$datasets = @("mnist", "fashion_mnist")
$attacks = @("clean", "free_rider", "stealth_free_rider", "lazy", "label_noise", "sign_flip")
$seed = 42
$rounds = 20

Write-Host "Bắt đầu chạy 24 thực nghiệm với Blockchain bật (không truyền cờ --no-blockchain)..."

foreach ($d in $datasets) {
    foreach ($a in $attacks) {
        # Baseline 1: FedAvg + Equal
        Write-Host "
>>> Đang chạy: $d | $a | FedAvg + Equal..."
        python -m experiments.run_experiment --dataset $d --scenario K2 --aggregation fedavg --reward-policy equal --attack $a --n-rounds $rounds --seed $seed

        # Proposed: CSRA-DCD + CSRA
        Write-Host "
>>> Đang chạy: $d | $a | CSRA-DCD + CSRA..."
        python -m experiments.run_experiment --dataset $d --scenario K2 --aggregation csra_dcd --reward-policy csra --attack $a --n-rounds $rounds --seed $seed --mad-threshold 3.0 --cosine-threshold -0.8
    }
}
Write-Host "
Hoàn thành 24 kịch bản!"
