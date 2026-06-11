import pandas as pd

from analysis.plots import _honest_participants, _malicious_mask, _ordered_client_types


def test_honest_participants_uses_client_type_not_reward_eligibility():
    df = pd.DataFrame([
        {
            "client_id": 0,
            "client_type": "honest",
            "is_malicious": False,
            "is_honest": False,
            "detection_reason": "reward_quarantine",
        },
        {
            "client_id": 1,
            "client_type": "sign_flip",
            "is_malicious": True,
            "is_honest": True,
            "detection_reason": "accepted",
        },
        {
            "client_id": 2,
            "client_type": "honest",
            "is_malicious": False,
            "is_honest": True,
            "detection_reason": "not_participating",
        },
    ])

    honest = _honest_participants(df)

    assert honest["client_id"].tolist() == [0]


def test_malicious_mask_parses_ground_truth_string_values():
    df = pd.DataFrame([
        {
            "client_id": 0,
            "client_type": "sign_flip",
            "ground_truth_honest": "True",
            "is_malicious": "True",
        },
        {
            "client_id": 1,
            "client_type": "honest",
            "ground_truth_honest": "False",
            "is_malicious": "False",
        },
        {
            "client_id": 2,
            "client_type": "sign_flip",
            "ground_truth_honest": pd.NA,
        },
    ])

    mask = _malicious_mask(df)

    assert mask.tolist() == [False, True, True]


def test_ordered_client_types_includes_all_attack_types():
    types = [
        "sign_flip", "honest", "label_noise", "lazy",
        "stealth_free_rider", "free_rider", "custom_attack",
    ]

    assert _ordered_client_types(types) == [
        "honest",
        "free_rider",
        "stealth_free_rider",
        "lazy",
        "label_noise",
        "sign_flip",
        "custom_attack",
    ]
