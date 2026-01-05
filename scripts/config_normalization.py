from __future__ import annotations

from typing import Any, Dict, List


def normalize_action_table_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize action_table entries to use 'rho_ns'.

    Rules per entry:
    - If 'rho_ns' exists: keep it.
    - Else if 'ns_ratio' exists: map ns_ratio -> rho_ns.
    - If both exist and differ: prefer rho_ns, ignore ns_ratio.
    - If neither exists: raise ValueError.
    """
    action_table = config.get("action_table")
    if not isinstance(action_table, list):
        return config

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(action_table):
        if not isinstance(entry, dict):
            raise ValueError(f"action_table entry at index {idx} must be a mapping/object")
        rho_ns = entry.get("rho_ns")
        ns_ratio = entry.get("ns_ratio")

        new_entry = dict(entry)
        if rho_ns is None and ns_ratio is None:
            raise ValueError("action_table entry must include 'rho_ns' (or legacy 'ns_ratio')")

        if rho_ns is None:
            new_entry["rho_ns"] = ns_ratio
        else:
            new_entry["rho_ns"] = rho_ns
        normalized.append(new_entry)

    config["action_table"] = normalized
    return config
