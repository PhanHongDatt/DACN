"""
simulation_local.py — Sequential FL simulation không phụ thuộc Ray.

Lý do tồn tại:
  - Python 3.14 hiện chưa có Ray (ray-3.x chỉ hỗ trợ ≤ Python 3.12).
  - Pipeline experiment chạy sequential 10 clients × ~50 rounds, không cần
    parallelism của Ray.
  - Manual loop dễ debug và deterministic hơn — đảm bảo `--seed` reproducible.

API:
  >>> run_simulation_local(
  ...     strategy=strategy,
  ...     clients={cid: numpy_client, ...},
  ...     initial_parameters=init_params,
  ...     n_rounds=50,
  ... )
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

log = logging.getLogger(__name__)


class _LocalClientProxy(ClientProxy):
    """Minimal ClientProxy implementation cho strategy.aggregate_*.

    Strategy chỉ dùng .cid để identify client; các method khác (fit/evaluate)
    sẽ không bao giờ được gọi vì chúng ta đã pre-compute results.
    """

    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout=None, group_id=None):
        raise NotImplementedError("local proxy không hỗ trợ get_properties")

    def get_parameters(self, ins, timeout=None, group_id=None):
        raise NotImplementedError("local proxy không hỗ trợ get_parameters")

    def fit(self, ins, timeout=None, group_id=None):
        raise NotImplementedError("call client directly, not via proxy")

    def evaluate(self, ins, timeout=None, group_id=None):
        raise NotImplementedError("call client directly, not via proxy")

    def reconnect(self, ins, timeout=None, group_id=None):
        raise NotImplementedError("local proxy không hỗ trợ reconnect")


def _wrap_fit_result(
    params_returned,
    num_examples: int,
    metrics: Dict,
) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(params_returned),
        num_examples=int(num_examples),
        metrics=metrics or {},
    )


def _wrap_eval_result(
    loss: float,
    num_examples: int,
    metrics: Dict,
) -> EvaluateRes:
    return EvaluateRes(
        status=Status(code=Code.OK, message=""),
        loss=float(loss),
        num_examples=int(num_examples),
        metrics=metrics or {},
    )


def run_simulation_local(
    strategy,
    client_fn: Callable[[str], fl.client.NumPyClient],
    num_clients: int,
    n_rounds: int,
    initial_parameters: Parameters,
) -> None:
    """
    Sequential simulation loop. Cho mỗi round:
      1. Inject round number vào fit config (giống configure_fit của Flower).
      2. Gọi client.fit() trên TẤT CẢ client (sequential).
      3. strategy.aggregate_fit() → cập nhật parameters.
      4. Gọi client.evaluate() trên TẤT CẢ client.
      5. strategy.aggregate_evaluate() → flush logs với accuracy.

    Args:
        strategy: instance của FLUnifiedStrategy (hoặc bất kỳ Flower strategy nào).
        client_fn: factory `str -> NumPyClient`. Sẽ được gọi mỗi round để khởi tạo
                   client (giống Flower simulation).
        num_clients: số lượng client.
        n_rounds: số rounds.
        initial_parameters: Parameters object khởi đầu (toàn cục).
    """
    parameters = initial_parameters

    # Pre-create proxies (chỉ dùng cid)
    proxies = {cid: _LocalClientProxy(str(cid)) for cid in range(num_clients)}

    log.info(
        f"Starting local sequential simulation — clients={num_clients} rounds={n_rounds}"
    )

    for round_num in range(1, n_rounds + 1):
        log.info(f"════════ Round {round_num}/{n_rounds} ════════")

        # ── Fit phase ────────────────────────────────────────────────────
        params_ndarrays = parameters_to_ndarrays(parameters)
        fit_config = {"round": round_num}

        fit_results: list[tuple[ClientProxy, FitRes]] = []
        for cid in range(num_clients):
            client = client_fn(str(cid))
            try:
                ret = client.fit(params_ndarrays, fit_config)
                if isinstance(ret, tuple) and len(ret) == 3:
                    params_returned, num_examples, metrics = ret
                else:
                    raise ValueError(
                        f"Client {cid} fit() returned unexpected shape: {type(ret)}"
                    )
                fit_results.append((
                    proxies[cid],
                    _wrap_fit_result(params_returned, num_examples, metrics),
                ))
            except Exception as e:
                log.error(f"Client {cid} fit failed: {e}", exc_info=True)
                continue

        new_params, _ = strategy.aggregate_fit(round_num, fit_results, [])
        if new_params is not None:
            parameters = new_params

        # ── Evaluate phase ───────────────────────────────────────────────
        eval_params_ndarrays = parameters_to_ndarrays(parameters)
        eval_config: Dict = {}

        eval_results: list[tuple[ClientProxy, EvaluateRes]] = []
        for cid in range(num_clients):
            client = client_fn(str(cid))
            try:
                ret = client.evaluate(eval_params_ndarrays, eval_config)
                if isinstance(ret, tuple) and len(ret) == 3:
                    loss, num_examples, metrics = ret
                else:
                    raise ValueError(
                        f"Client {cid} evaluate() returned unexpected shape: {type(ret)}"
                    )
                eval_results.append((
                    proxies[cid],
                    _wrap_eval_result(loss, num_examples, metrics),
                ))
            except Exception as e:
                log.error(f"Client {cid} evaluate failed: {e}", exc_info=True)
                continue

        strategy.aggregate_evaluate(round_num, eval_results, [])

    log.info("Local simulation finished successfully.")
