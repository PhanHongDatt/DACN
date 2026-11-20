# -*- coding: utf-8 -*-
import sys

with open('fl/server_base.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace import
old_import = 'from fl.blockchain import BlockchainBridge'
new_import = 'from fl.blockchain import BlockchainBridge\nfrom fl.reputation_simulator import OffChainReputation'
content = content.replace(old_import, new_import, 1)

# Replace init
old_init = 'self.bridge = bridge\n        self.seed = int(seed)'
new_init = 'self.bridge = bridge\n        self.rep_sim = OffChainReputation() if self.bridge is None else None\n        self.seed = int(seed)'
content = content.replace(old_init, new_init, 1)

# Replace else block
old_else = '''        else:
            # Không có blockchain → reputation default = 0.5 cho mọi participant
            # (đủ để CSRA reward formula chạy, tránh bias)
            for cid in range(self.cfg.fl.n_clients):
                reputations_map[cid] = 0.5
                onchain_honesty_map[cid] = True'''

new_else = '''        else:
            # Dùng OffChainReputation simulator
            for i, d in enumerate(client_data):
                is_reward_blocked = d["cid"] in reward_blocked_cids
                alignment_qs = max(0.0, detection_info[d["cid"]].get("cosine_to_reference", 0.0) or 0.0)
                qs = 0.0 if is_reward_blocked else alignment_qs
                ds = 0 if is_reward_blocked else d["data_size"]
                if self.rep_sim is not None:
                    self.rep_sim.record_contribution(
                        d["cid"], qs, ds, self.mean_data_size, server_round
                    )
            
            for cid in range(self.cfg.fl.n_clients):
                if self.rep_sim is not None:
                    rep, is_honest_on_chain = self.rep_sim.get_reputation(cid)
                    reputations_map[cid] = float(rep)
                    onchain_honesty_map[cid] = bool(is_honest_on_chain)
                else:
                    reputations_map[cid] = 0.5
                    onchain_honesty_map[cid] = True'''
content = content.replace(old_else, new_else, 1)

with open('fl/server_base.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully')
