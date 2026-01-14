from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

RATIO_ABS_TOL = 1e-6
RATIO_REL_TOL = 1e-9

def _validate_action_splits(action_splits: List[Tuple[float, float]], rho_min: float) -> None:
    if len(action_splits) == 0:
        raise ValueError("action_splits must not be empty")
    for idx, (rho_ns_val, rho_ew_val) in enumerate(action_splits):
        if rho_ns_val <= 0.0 or rho_ew_val <= 0.0:
            raise ValueError(f"action_splits[{idx}] values must be >0")
        if rho_ns_val >= 1.0 or rho_ew_val >= 1.0:
            raise ValueError(f"action_splits[{idx}] values must be in (0,1)")
        if not math.isclose((rho_ns_val + rho_ew_val), 1.0, rel_tol=RATIO_REL_TOL, abs_tol=RATIO_ABS_TOL):
            raise ValueError(f"action_splits[{idx}] rho_ns+rho_ew must equal 1.0, got {rho_ns_val + rho_ew_val:.10f}")
        if rho_ns_val < rho_min or rho_ew_val < rho_min:
            raise ValueError(f"action_splits[{idx}] values must be >= rho_min={rho_min}")

def _split_index(rho_ns: float, rho_ew: float, action_splits: List[Tuple[float, float]]) -> int:
    for idx, (target_ns, target_ew) in enumerate(action_splits):
        if math.isclose(rho_ns, target_ns, rel_tol=RATIO_REL_TOL, abs_tol=RATIO_ABS_TOL) and math.isclose(
            rho_ew, target_ew, rel_tol=RATIO_REL_TOL, abs_tol=RATIO_ABS_TOL
        ):
            return idx
    return -1


def validate_action_table(
    action_table_raw: Iterable[Dict[str, Any]],
    action_splits: List[Tuple[float, float]],
    state_dim: int,
    allowed_cycles: List[int],
    rho_min: float,
    g_min_sec: int,
) -> List[Dict[str, Any]]:
    _validate_action_splits(action_splits, rho_min)
    expected_cycles = len(allowed_cycles)
    expected_splits = len(action_splits)
    if state_dim == 12:
        if expected_cycles != 3:
            raise ValueError(f"allowed_cycles_sec must have exactly 3 entries for state_dim=12, got {expected_cycles}")
        if expected_splits != 5:
            raise ValueError(f"action_splits must have exactly 5 entries for state_dim=12, got {expected_splits}")

    processed_action_table: List[Dict[str, Any]] = []
    if isinstance(action_table_raw, list) and len(action_table_raw) > 0:
        entries: List[Tuple[int, float, float, int]] = []
        for idx, item in enumerate(action_table_raw):
            cycle = item.get("cycle_sec")
            rho_ns = item.get("rho_ns", item.get("ns_ratio"))
            rho_ew = item.get("rho_ew")
            if cycle is None or rho_ns is None:
                raise ValueError(f"action_table[{idx}] must include cycle_sec and rho_ns/ns_ratio")
            cycle_val = int(cycle)
            if cycle_val <= 0:
                raise ValueError(f"action_table[{idx}] cycle_sec must be >0")
            rho_ns_val = float(rho_ns)
            if rho_ns_val <= 0.0 or rho_ns_val >= 1.0:
                raise ValueError(f"action_table[{idx}] rho_ns must be in (0,1)")
            if rho_ew is None:
                rho_ew_val = 1.0 - rho_ns_val
            else:
                rho_ew_val = float(rho_ew)
            if rho_ew_val <= 0.0:
                raise ValueError(f"action_table[{idx}] rho_ew must be >0")
            if not math.isclose((rho_ns_val + rho_ew_val), 1.0, rel_tol=RATIO_REL_TOL, abs_tol=RATIO_ABS_TOL):
                raise ValueError(f"action_table[{idx}] rho_ns+rho_ew must equal 1.0, got {rho_ns_val + rho_ew_val:.10f}")
            if cycle_val not in allowed_cycles:
                raise ValueError(f"action_table[{idx}] cycle_sec={cycle_val} not in allowed_cycles_sec={allowed_cycles}")
            if rho_ns_val < rho_min or rho_ew_val < rho_min:
                raise ValueError(f"action_table[{idx}] rho values must be >= rho_min={rho_min}")
            g_ns_check = float(rho_ns_val) * float(cycle_val)
            g_ew_check = float(rho_ew_val) * float(cycle_val)
            if g_ns_check < g_min_sec or g_ew_check < g_min_sec:
                raise ValueError(f"action_table[{idx}] green times must be >= g_min_sec={g_min_sec}")
            split_idx = _split_index(rho_ns_val, rho_ew_val, action_splits)
            if state_dim == 12 and split_idx < 0:
                raise ValueError(f"action_table[{idx}] split must match one of action_splits")
            entries.append((cycle_val, rho_ns_val, rho_ew_val, split_idx))
        if state_dim == 12:
            expected = expected_cycles * expected_splits
            seen = {}
            for cycle_val, rho_ns_val, rho_ew_val, split_idx in entries:
                key = (cycle_val, split_idx)
                if key in seen:
                    raise ValueError(f"Duplicate action_table entry for cycle_sec={cycle_val} split_index={split_idx}")
                seen[key] = (rho_ns_val, rho_ew_val)
            if len(seen) != expected:
                missing = []
                for cycle_val in allowed_cycles:
                    for split_idx in range(expected_splits):
                        if (cycle_val, split_idx) not in seen:
                            missing.append(f"{cycle_val}:{split_idx}")
                raise ValueError(f"action_table must include all cycle/split combinations, missing={missing}")
            for cycle_val in allowed_cycles:
                for split_idx in range(expected_splits):
                    rho_ns_val, rho_ew_val = seen[(cycle_val, split_idx)]
                    processed_action_table.append({"cycle_sec": cycle_val, "rho_ns": rho_ns_val, "rho_ew": rho_ew_val})
        else:
            for cycle_val, rho_ns_val, rho_ew_val, _ in entries:
                processed_action_table.append({"cycle_sec": cycle_val, "rho_ns": rho_ns_val, "rho_ew": rho_ew_val})
    elif state_dim == 12:
        if len(allowed_cycles) == 0:
            raise ValueError("allowed_cycles_sec must not be empty when state_dim=12 and action_table is empty")
        for cycle in allowed_cycles:
            for rho_ns, rho_ew in action_splits:
                if rho_ns < rho_min or rho_ew < rho_min:
                    raise ValueError(f"action_splits contains rho below rho_min={rho_min}")
                g_ns_check = float(rho_ns) * float(cycle)
                g_ew_check = float(rho_ew) * float(cycle)
                if g_ns_check < g_min_sec or g_ew_check < g_min_sec:
                    raise ValueError(f"default action entry cycle {cycle} violates g_min_sec={g_min_sec}")
                processed_action_table.append(
                    {"cycle_sec": int(cycle), "rho_ns": float(rho_ns), "rho_ew": float(rho_ew)}
                )

        seen_actions = set()
        deduplicated = []
        for entry in processed_action_table:
            key = (entry["cycle_sec"], round(entry["rho_ns"], 4), round(entry["rho_ew"], 4))
            if key not in seen_actions:
                seen_actions.add(key)
                deduplicated.append(entry)

        processed_action_table = deduplicated

    return processed_action_table


def validate_scalar_params(
    yellow_sec: int,
    all_red_sec: int,
    rho_min: float,
    g_min_sec: int,
    queue_count_mode: str,
    halt_speed_threshold: float,
    use_enhanced_reward: bool,
    reward_exponent: float,
    enable_spillback_penalty: bool,
    alpha_spillback: float,
    allowed_cycles: List[int],
) -> None:
    mode = str(queue_count_mode).lower()

    if yellow_sec < 0:
        raise ValueError("yellow_sec must be >=0")
    if all_red_sec < 0:
        raise ValueError("all_red_sec must be >=0")
    if rho_min <= 0.0 or rho_min > 0.5:
        raise ValueError("rho_min must be in (0, 0.5]")
    if g_min_sec < 0:
        raise ValueError("g_min_sec must be >=0")
    if mode == "snapshot_last_step":
        raise ValueError(
            "queue_count_mode='snapshot_last_step' is no longer supported.\n"
            "MDP compliance requires 'distinct_cycle' mode.\n"
            "This mode tracks distinct vehicles queued at least once per cycle."
        )
    if mode not in {"distinct_cycle"}:
        raise ValueError(f"queue_count_mode must be 'distinct_cycle', got '{mode}'")
    if halt_speed_threshold < 0.0:
        raise ValueError("halt_speed_threshold must be >=0")
    if use_enhanced_reward and reward_exponent < 1.0:
        raise ValueError("reward_exponent must be >=1 when use_enhanced_reward is True")
    if enable_spillback_penalty and alpha_spillback < 0.0:
        raise ValueError("alpha_spillback must be >=0 when enable_spillback_penalty is True")
    if len(allowed_cycles) == 0 or any(cycle <= 0 for cycle in allowed_cycles):
        raise ValueError("allowed_cycles_sec must contain positive cycle lengths")
