import re

with open('scripts/common.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '            reward_time_normalize=bool(sumo_cfg.get("reward_time_normalize", False)),\n        )'
new = '''            reward_time_normalize=bool(sumo_cfg.get("reward_time_normalize", False)),
            tls_phase_overrides={str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in sumo_cfg.get("tls_phase_overrides", {}).items()},
        )'''

content = content.replace(old, new)

with open('scripts/common.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
