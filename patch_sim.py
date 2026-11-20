# -*- coding: utf-8 -*-
import sys

with open('fl/simulation_local.py', 'r', encoding='utf-8') as f:
    content = f.read()

import_future = 'import concurrent.futures\n'
if import_future not in content:
    content = content.replace('from typing import Callable, Dict', 'from typing import Callable, Dict\nimport concurrent.futures')

old_fit = '''        fit_results: list[tuple[ClientProxy, FitRes]] = []
        for cid in range(num_clients):
            client = _client_for(cid)
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
                continue'''

new_fit = '''        fit_results: list[tuple[ClientProxy, FitRes]] = []
        
        def fit_client(cid):
            client = _client_for(cid)
            try:
                ret = client.fit(params_ndarrays, fit_config)
                if isinstance(ret, tuple) and len(ret) == 3:
                    params_returned, num_examples, metrics = ret
                else:
                    raise ValueError(
                        f"Client {cid} fit() returned unexpected shape: {type(ret)}"
                    )
                return cid, _wrap_fit_result(params_returned, num_examples, metrics)
            except Exception as e:
                log.error(f"Client {cid} fit failed: {e}", exc_info=True)
                return cid, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fit_client, cid) for cid in range(num_clients)]
            for future in concurrent.futures.as_completed(futures):
                cid, res = future.result()
                if res is not None:
                    fit_results.append((proxies[cid], res))
                    
        # Giữ thứ tự deterministic
        fit_results.sort(key=lambda x: int(x[0].cid))'''
        
content = content.replace(old_fit, new_fit)

old_eval = '''        eval_results: list[tuple[ClientProxy, EvaluateRes]] = []
        for cid in range(num_clients):
            client = _client_for(cid)
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
                continue'''

new_eval = '''        eval_results: list[tuple[ClientProxy, EvaluateRes]] = []
        
        def eval_client(cid):
            client = _client_for(cid)
            try:
                ret = client.evaluate(eval_params_ndarrays, eval_config)
                if isinstance(ret, tuple) and len(ret) == 3:
                    loss, num_examples, metrics = ret
                else:
                    raise ValueError(
                        f"Client {cid} evaluate() returned unexpected shape: {type(ret)}"
                    )
                return cid, _wrap_eval_result(loss, num_examples, metrics)
            except Exception as e:
                log.error(f"Client {cid} evaluate failed: {e}", exc_info=True)
                return cid, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(eval_client, cid) for cid in range(num_clients)]
            for future in concurrent.futures.as_completed(futures):
                cid, res = future.result()
                if res is not None:
                    eval_results.append((proxies[cid], res))
                    
        # Giữ thứ tự deterministic
        eval_results.sort(key=lambda x: int(x[0].cid))'''

content = content.replace(old_eval, new_eval)

with open('fl/simulation_local.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully')
