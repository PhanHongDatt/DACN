import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fl.simulation_local import run_simulation_local


class CountingClient:
    def __init__(self, cid: str):
        self.cid = cid
        self.fit_calls = 0
        self.evaluate_calls = 0

    def fit(self, parameters, config):
        self.fit_calls += 1
        return parameters, 1, {"fit_calls": self.fit_calls}

    def evaluate(self, parameters, config):
        self.evaluate_calls += 1
        return 0.0, 1, {"accuracy": 1.0}


class EchoStrategy:
    def __init__(self):
        self.fit_rounds = []
        self.evaluate_rounds = []

    def set_current_global_parameters(self, parameters):
        self.current_parameters = parameters

    def aggregate_fit(self, server_round, results, failures):
        self.fit_rounds.append((server_round, len(results)))
        if not results:
            return None, {}
        return results[0][1].parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        self.evaluate_rounds.append((server_round, len(results)))
        return 0.0, {"accuracy": 1.0}


def test_local_simulation_default_recreates_clients_for_fit_and_evaluate():
    factory_calls = []

    def client_fn(cid: str):
        factory_calls.append(cid)
        return CountingClient(cid)

    strategy = EchoStrategy()
    run_simulation_local(
        strategy=strategy,
        client_fn=client_fn,
        num_clients=2,
        n_rounds=2,
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    assert len(factory_calls) == 8
    assert strategy.fit_rounds == [(1, 2), (2, 2)]
    assert strategy.evaluate_rounds == [(1, 2), (2, 2)]


def test_local_simulation_can_reuse_persistent_clients():
    factory_calls = []
    clients = {}

    def client_fn(cid: str):
        factory_calls.append(cid)
        client = CountingClient(cid)
        clients[cid] = client
        return client

    strategy = EchoStrategy()
    initial = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
    run_simulation_local(
        strategy=strategy,
        client_fn=client_fn,
        num_clients=2,
        n_rounds=2,
        initial_parameters=initial,
        persistent_clients=True,
    )

    assert factory_calls == ["0", "1"]
    assert clients["0"].fit_calls == 2
    assert clients["0"].evaluate_calls == 2
    assert clients["1"].fit_calls == 2
    assert clients["1"].evaluate_calls == 2
    assert parameters_to_ndarrays(initial)[0].shape == (1,)
