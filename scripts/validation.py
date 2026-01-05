from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _validate_action_splits(action_splits: List[Tuple[float, float]], rho_min: float) -> None:
    for idx, (rho_ns_val, rho_ew_val) in enumerate(action_splits):
        if rho_ns_val <= 0.0 or rho_ew_val <= 0.0:
            raise ValueError(f"action_splits[{idx}] values must be >0")
        if abs((rho_ns_val + rho_ew_val) - 1.0) > 1e-6:
            raise ValueError(f"action_splits[{idx}] rho_ns+rho_ew must equal 1.0")
        if rho_ns_val < rho_min or rho_ew_val < rho_min:
            raise ValueError(f"action_splits[{idx}] values must be >= rho_min={rho_min}")


def validate_action_table(
    action_table_raw: Iterable[Dict[str, Any]],
    action_splits: List[Tuple[float, float]],
    state_dim: int,
    allowed_cycles: List[int],
    rho_min: float,
    g_min_sec: int,
) -> List[Dict[str, Any]]:
    _validate_action_splits(action_splits, rho_min)

    processed_action_table: List[Dict[str, Any]] = []
    if isinstance(action_table_raw, list) and len(action_table_raw) > 0:
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
            if abs((rho_ns_val + rho_ew_val) - 1.0) > 1e-6:
                raise ValueError(f"action_table[{idx}] rho_ns+rho_ew must equal 1.0")
            if cycle_val not in allowed_cycles:
                raise ValueError(f"action_table[{idx}] cycle_sec={cycle_val} not in allowed_cycles_sec={allowed_cycles}")
            if rho_ns_val < rho_min or rho_ew_val < rho_min:
                raise ValueError(f"action_table[{idx}] rho values must be >= rho_min={rho_min}")
            g_ns_check = float(rho_ns_val) * float(cycle_val)
            g_ew_check = float(rho_ew_val) * float(cycle_val)
            if g_ns_check < g_min_sec or g_ew_check < g_min_sec:
                raise ValueError(f"action_table[{idx}] green times must be >= g_min_sec={g_min_sec}")
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
                processed_action_table.append({"cycle_sec": int(cycle), "rho_ns": float(rho_ns), "rho_ew": float(rho_ew)})

    return processed_action_table


def validate_scalar_params(
    yellow_sec: int,
    all_red_sec: int,
    rho_min: float,
    g_min_sec: int,
    lambda_fairness: float,
    fairness_metric: str,
    queue_count_mode: str,
    halt_speed_threshold: float,
    use_enhanced_reward: bool,
    reward_exponent: float,
    enable_anti_flicker: bool,
    kappa: float,
    enable_spillback_penalty: bool,
    beta: float,
    occ_threshold: float,
    allowed_cycles: List[int],
) -> None:
    if yellow_sec < 0:
        raise ValueError("yellow_sec must be >=0")
    if all_red_sec < 0:
        raise ValueError("all_red_sec must be >=0")
    if rho_min <= 0.0 or rho_min > 0.5:
        raise ValueError("rho_min must be in (0, 0.5]")
    if g_min_sec < 0:
        raise ValueError("g_min_sec must be >=0")
    if lambda_fairness < 0.0:
        raise ValueError("lambda_fairness must be >=0")
    if fairness_metric not in {"max", "p95"}:
        raise ValueError("fairness_metric must be max or p95")
    if queue_count_mode not in {"distinct_cycle", "snapshot_last_step"}:
        raise ValueError(
            f"queue_count_mode must be 'distinct_cycle' or 'snapshot_last_step', got '{queue_count_mode}'\n"
            f"Note: 'snapshot_last_step' is deprecated (not MDP-compliant).\n"
            f"Recommended: Use 'distinct_cycle' (MDP requirement)."
        )

    if queue_count_mode == "snapshot_last_step":
        import warnings
        warnings.warn(
            "queue_count_mode='snapshot_last_step' is deprecated.\n"
            "This mode does not comply with MDP specification.\n"
            "Please use 'distinct_cycle' instead.",
            DeprecationWarning,
            stacklevel=2
        )
    if halt_speed_threshold < 0.0:
        raise ValueError("halt_speed_threshold must be >=0")
    if use_enhanced_reward and reward_exponent < 1.0:
        raise ValueError("reward_exponent must be >=1 when use_enhanced_reward is True")
    if enable_anti_flicker and kappa < 0.0:
        raise ValueError("kappa must be >=0 when enable_anti_flicker is True")
    if enable_spillback_penalty:
        if beta < 0.0:
            raise ValueError("beta must be >=0 when enable_spillback_penalty is True")
        if occ_threshold < 0.0 or occ_threshold > 1.0:
            raise ValueError("occ_threshold must be in [0,1] when enable_spillback_penalty is True")
    if len(allowed_cycles) == 0 or any(cycle <= 0 for cycle in allowed_cycles):
        raise ValueError("allowed_cycles_sec must contain positive cycle lengths")
