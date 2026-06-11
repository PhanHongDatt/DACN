import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl.client import FLClient
from fl.client_attacks import StealthFreeRiderClient
from fl.client_csra import FLClientCSRA
from fl.config import FLConfig
from fl.models import get_model, get_parameters
from experiments.run_experiment import build_attack_kwargs, derive_attack_seed


def _loader(n_samples: int, batch_size: int):
    X = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def test_free_rider_update_participates_but_reports_zero_data_commitment():
    train_loader = _loader(4, 2)
    test_loader = _loader(4, 2)
    client = FLClient(
        0,
        "mnist",
        train_loader,
        test_loader,
        client_type="free_rider",
        fl_cfg=FLConfig(local_epochs=1),
    )

    _, num_examples, metrics = client.fit(get_parameters(get_model("mnist")), {"round": 1})

    assert num_examples == 4
    assert metrics["data_size"] == 0
    assert metrics["quality_score"] == 0.0


def test_csra_free_rider_reports_zero_data_commitment():
    train_loader = _loader(4, 2)
    test_loader = _loader(4, 2)
    client = FLClientCSRA(
        0,
        "mnist",
        train_loader,
        test_loader,
        client_type="free_rider",
        fl_cfg=FLConfig(local_epochs=1),
    )

    _, num_examples, metrics = client.fit(get_parameters(get_model("mnist")), {"round": 1})

    assert num_examples == 4
    assert metrics["data_size"] == 0
    assert metrics["anomaly_score"] > 0
    assert metrics["update_norm"] == metrics["anomaly_score"]
    assert metrics["variance"] > 0


def test_csra_copy_free_rider_reports_zero_delta_norm():
    train_loader = _loader(4, 2)
    test_loader = _loader(4, 2)
    client = FLClientCSRA(
        0,
        "mnist",
        train_loader,
        test_loader,
        client_type="free_rider",
        fl_cfg=FLConfig(local_epochs=1),
        free_rider_mode="copy",
    )

    _, num_examples, metrics = client.fit(get_parameters(get_model("mnist")), {"round": 1})

    assert num_examples == 4
    assert metrics["data_size"] == 0
    assert metrics["anomaly_score"] == 0.0


def test_stealth_free_rider_fakes_commitment_metadata():
    train_loader = _loader(4, 2)
    test_loader = _loader(4, 2)
    client = StealthFreeRiderClient(
        client_id=0,
        dataset_name="mnist",
        train_loader=train_loader,
        test_loader=test_loader,
        fl_cfg=FLConfig(local_epochs=1),
        mode="copy",
        fake_quality=0.25,
    )

    _, num_examples, metrics = client.fit(get_parameters(get_model("mnist")), {"round": 1})

    assert num_examples == 4
    assert metrics["client_type"] == "stealth_free_rider"
    assert metrics["data_size"] == 4
    assert metrics["quality_score"] == 0.25
    assert metrics["anomaly_score"] == 0.0


def test_label_noise_attack_seed_depends_on_experiment_seed_and_client_id():
    seed_a = derive_attack_seed(seed=42, client_id=3)
    seed_b = derive_attack_seed(seed=42, client_id=3)
    seed_c = derive_attack_seed(seed=43, client_id=3)
    seed_d = derive_attack_seed(seed=42, client_id=4)

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert seed_a != seed_d
    assert 0 <= seed_a < 2**32
    assert build_attack_kwargs("label_noise", seed=42, client_id=3) == {
        "rng_seed": seed_a
    }
    assert build_attack_kwargs("sign_flip", seed=42, client_id=3) == {}


def test_honest_client_skips_singleton_training_batch_without_crashing():
    train_loader = _loader(1, 1)
    test_loader = _loader(2, 1)
    client = FLClient(
        0,
        "mnist",
        train_loader,
        test_loader,
        client_type="honest",
        fl_cfg=FLConfig(local_epochs=1),
    )

    _, num_examples, metrics = client.fit(get_parameters(get_model("mnist")), {"round": 1})

    assert num_examples == 1
    assert np.isfinite(metrics["quality_score"])
