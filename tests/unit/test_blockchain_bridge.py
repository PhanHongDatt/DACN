import pytest

from fl.blockchain import BlockchainBridge
from fl.config import BlockchainConfig, ContributionConfig


class _SuccessfulCall:
    def __init__(self, result):
        self._result = result

    def call(self):
        return self._result


class _FailingCall:
    def call(self):
        raise RuntimeError("mock reputation read failed")


class _Functions:
    def __init__(self, call):
        self._call = call

    def getReputation(self, _address):
        return self._call


class _Store:
    def __init__(self, call):
        self.functions = _Functions(call)


def _bridge_with_store(cfg: BlockchainConfig, store: _Store) -> BlockchainBridge:
    bridge = BlockchainBridge.__new__(BlockchainBridge)
    bridge.cfg = cfg
    bridge.contrib_cfg = ContributionConfig()
    bridge.accounts = ["owner", "client-0"]
    bridge.store = store
    return bridge


def test_get_reputation_scales_successful_onchain_value():
    bridge = _bridge_with_store(
        BlockchainConfig(),
        _Store(_SuccessfulCall((750_000, True))),
    )

    reputation, is_honest = bridge.get_reputation(0)

    assert reputation == pytest.approx(0.75)
    assert is_honest is True


def test_get_reputation_fails_closed_by_default():
    bridge = _bridge_with_store(
        BlockchainConfig(),
        _Store(_FailingCall()),
    )

    reputation, is_honest = bridge.get_reputation(0)

    assert reputation == 0.0
    assert is_honest is False


def test_get_reputation_can_fail_open_for_debugging():
    bridge = _bridge_with_store(
        BlockchainConfig(reputation_fail_open=True),
        _Store(_FailingCall()),
    )

    reputation, is_honest = bridge.get_reputation(0)

    assert reputation == 0.0
    assert is_honest is True
