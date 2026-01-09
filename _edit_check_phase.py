
with open('scripts/check_phase_sync.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''    ns_idx = getattr(phases, "ns_green", None)
    ew_idx = getattr(phases, "ew_green", None)
    if ns_idx is None or ew_idx is None:
        return "unknown", {"reason": "phase_indices_missing"}'''

new = '''    try:
        ns_idx, ew_idx = env.get_ns_ew_phase_indices(tls_id)
    except AttributeError:
        ns_idx = getattr(phases, "ns_green", None)
        ew_idx = getattr(phases, "ew_green", None)

    if ns_idx is None or ew_idx is None:
        return "unknown", {"reason": "phase_indices_missing"}'''

content = content.replace(old, new)

with open('scripts/check_phase_sync.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
