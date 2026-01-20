#!/usr/bin/env python
"""
Feasibility Gating Script for Demand/Horizon Selection (Snapshot-Based)

Runs MaxPressure and Fixed-Time baselines with SNAPSHOT-BASED metrics collection.
Single long run per (demand, seed, controller), with snapshots at fixed times.

Enhanced per advisor guidance:
- Snapshots at [3600, 4200, 4800, 5400] seconds
- completion_after_drain = arrived_cum(t) / departed_3600
- clear_fraction = (n_present_3600 - n_present(t)) / n_present_3600
- Early stop if network empty after route_end (3600s)

Usage:
    python scripts/feasibility_gating.py --mode demand_sweep --demands 600,800,1000 --seeds 2
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env
from controllers.max_pressure import select_action_from_defs
from controllers.fixed_time import FixedTimeController, FixedTimeControllerConfig


# =============================================================================
# CONFIGURATION
# =============================================================================
ROUTE_END_TIME = 3600  # Route injection ends at this time
DEFAULT_SNAPSHOT_TIMES = [3600, 4200, 4800, 5400]  # Snapshot times in seconds
DEFAULT_HORIZON_END = 5400  # Default max simulation time
BIN_SIZE_SEC = 300  # Fixed bin size for throughput calculation
WARMUP_SEC = 300  # Warmup period (counters still cumulative)

# =============================================================================
# PASS/FAIL THRESHOLDS (locked per advisor guidance)
# =============================================================================
STRICT_THRESHOLDS = {
    'teleport_rate': 0.02,              # ≤ 2%
    'wrong_lane_share': 0.20,           # ≤ 20%
    'completion_after_drain': 0.95,     # ≥ 95%
    'throughput_end_ratio': 0.70,       # ≥ 0.70
}

RELAXED_THRESHOLDS = {
    'teleport_rate': 0.05,              # ≤ 5%
    'wrong_lane_share': 0.30,           # ≤ 30%
    'throughput_end_ratio': 0.50,       # ≥ 0.50
}


def _select_route_for_demand(demand: int, seed: int) -> str:
    """Select a deterministic route from the manifest for a demand/seed combo."""
    manifest_path = project_root / f"networks/variants/train_turn801010/{demand}/manifest_t1000.txt"
    if not manifest_path.exists():
        return ""
    try:
        with open(manifest_path, "r") as f:
            routes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception:
        return ""
    if not routes:
        return ""
    route_file = routes[(int(seed) - 42) % len(routes)]
    full_path = project_root / f"networks/variants/train_turn801010/{demand}" / route_file
    return str(full_path) if full_path.exists() else ""


def _run_gating_task(args_tuple):
    """Helper for multiprocessing pool."""
    cfg, demand, seed, ctrl, horizon, warm = args_tuple
    print(f"[Gating] d={demand} s={seed} c={ctrl} h={horizon}")
    return run_single_episode_quick(
        base_config_path=cfg,
        demand=int(demand),
        seed=int(seed),
        controller=str(ctrl),
        horizon_sec=int(horizon),
        warmup_sec=int(warm),
    )


def run_single_episode_quick(
    base_config_path: str,
    demand: int,
    seed: int,
    controller: str,
    horizon_sec: int = 1500,
    warmup_sec: int = 300,
) -> Dict[str, Any]:
    """
    Run a short gating episode with bounded metrics and collapse evidence.
    """
    from rl.utils import load_yaml_config, set_global_seed
    from scripts.common import build_env

    config = load_yaml_config(base_config_path)
    config = copy.deepcopy(config)
    config.setdefault("env", {}).setdefault("sumo", {})
    config.setdefault("run", {})

    config["run"]["seed"] = int(seed)
    config["env"]["sumo"]["max_sim_seconds"] = int(horizon_sec)
    if warmup_sec is not None:
        config["env"]["sumo"]["warmup_sec"] = int(warmup_sec)

    route_file = _select_route_for_demand(demand, seed)
    if route_file:
        config["env"]["sumo"]["route_file"] = route_file

    set_global_seed(seed)

    env = None
    result: Dict[str, Any] = {}
    try:
        env = build_env(config)
        action_defs = getattr(env, "_action_defs", [])
        fixed_controller = None
        if controller == "fixed":
            try:
                cycle_target = int(config["env"]["sumo"].get("green_cycle_sec", 90))
                fixed_cfg = FixedTimeControllerConfig(target_split=(0.5, 0.5), target_cycle_sec=cycle_target)
                fixed_controller = FixedTimeController(action_space=action_defs, config=fixed_cfg)
            except Exception:
                fixed_controller = None

        state = env.reset()
        total_reward = 0.0
        step_count = 0
        n_present_series: List[int] = []
        max_no_arrival_steps = 0
        last_info: Dict[str, Any] = {}

        while True:
            if isinstance(state, dict):
                allowed = getattr(env, "_current_allowed_actions", None)
                actions: Dict[str, int] = {}
                for tls_id, tls_state in state.items():
                    if controller == "fixed" and fixed_controller is not None:
                        action_id = fixed_controller.act()
                    elif controller == "max_pressure":
                        allowed_ids = None
                        if isinstance(allowed, dict):
                            allowed_ids = allowed.get(tls_id)
                        action_id = select_action_from_defs(
                            state_raw=tls_state,
                            action_defs=action_defs,
                            allowed_action_ids=allowed_ids,
                            default_action_id=7,
                        )
                    else:
                        action_id = select_action_from_defs(
                            state_raw=tls_state,
                            action_defs=action_defs,
                            default_action_id=7,
                        )
                    actions[tls_id] = action_id
            else:
                if controller == "fixed" and fixed_controller is not None:
                    actions = fixed_controller.act()
                elif controller == "max_pressure":
                    actions = select_action_from_defs(state_raw=state, action_defs=action_defs, default_action_id=7)
                else:
                    actions = select_action_from_defs(state_raw=state, action_defs=action_defs, default_action_id=7)

            next_state, rewards, done, info = env.step(actions)
            reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
            total_reward += float(np.mean(reward_values))
            step_count += 1
            state = next_state
            last_info = info if isinstance(info, dict) else {}

            max_no_arrival_steps = max(max_no_arrival_steps, int(getattr(env, "_no_arrival_steps", 0)))

            n_present_val: Optional[int] = None
            if isinstance(info, dict):
                n_present_val = info.get("n_present")
            if n_present_val is None and hasattr(env, "_traci") and env._traci is not None:
                try:
                    n_present_val = int(env._traci.vehicle.getIDCount())
                except Exception:
                    n_present_val = None
            n_present_series.append(int(n_present_val) if n_present_val is not None else 0)

            if bool(done) or step_count >= int(horizon_sec * 2):
                break

        kpi = {}
        if hasattr(env, "episode_kpi"):
            kpi = env.episode_kpi()
        if not kpi and isinstance(last_info, dict):
            kpi = last_info.get("episode_kpi", {})

        step_len = float(config["env"]["sumo"].get("step_length_sec", 1.0))
        no_arrival_sec_max = float(max_no_arrival_steps) * step_len

        slope_window_steps = int(300 / step_len) if step_len > 0 else len(n_present_series)
        slope_series = n_present_series[-slope_window_steps:] if slope_window_steps > 0 else n_present_series
        present_slope = 0.0
        if len(slope_series) >= 2:
            present_slope = (float(slope_series[-1]) - float(slope_series[0])) / max(
                1.0, float(len(slope_series) - 1) * step_len
            )

        completion_rate_raw = float(kpi.get("completion_rate", 0.0))
        teleport_rate_raw = float(kpi.get("teleport_rate", 0.0))
        collapse_flag = int(kpi.get("deadlock_triggered", 0))
        collapse_reason = str(kpi.get("deadlock_reason", ""))

        # Estimate departed count from completion_rate if available
        arrived = int(kpi.get("arrived_vehicles", 0))
        departed_est = int(round(arrived / completion_rate_raw)) if completion_rate_raw > 0 else arrived
        departed_est = max(departed_est, arrived)

        completion_rate_correct = float(np.clip(completion_rate_raw, 0.0, 1.0))
        teleport_rate_correct = float(np.clip(teleport_rate_raw, 0.0, 1.0))
        if completion_rate_raw > 1.0 + 1e-6:
            raise ValueError(f"completion_rate>1 detected (arrived={arrived}, completion_raw={completion_rate_raw})")
        if teleport_rate_raw > 1.0 + 1e-6:
            raise ValueError(f"teleport_rate>1 detected (departed_est={departed_est}, teleport_raw={teleport_rate_raw})")

        result = {
            "controller": controller,
            "demand": int(demand),
            "horizon_sec": int(horizon_sec),
            "warmup_sec": int(warmup_sec),
            "seed": int(seed),
            "route_file": os.path.basename(route_file) if route_file else "",
            "departed_0_injectionEnd": int(departed_est),
            "arrived_0_simEnd": int(arrived),
            "completion_rate": completion_rate_correct,  # legacy alias
            "teleport_rate": teleport_rate_correct,      # legacy alias
            "completion_rate_correct": completion_rate_correct,
            "teleport_rate_correct": teleport_rate_correct,
            "no_arrival_sec_max": float(no_arrival_sec_max),
            "present_slope_last300s": float(present_slope),
            "collapse_flag": int(collapse_flag),
            "collapse_reason": collapse_reason,
            "collapse_flag_rate": float(collapse_flag),
            "n_present_end": int(n_present_series[-1] if n_present_series else 0),
            "avg_wait_time": float(kpi.get("avg_wait_time_corr", kpi.get("avg_wait_time", 0.0))),
            "avg_queue": float(kpi.get("avg_queue", 0.0)),
            "total_reward": float(total_reward),
            "episode_steps": int(step_count),
            "teleport_unique": int(kpi.get("teleport_unique", 0)),
            "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
            "deadlock_no_arrival_sec": float(kpi.get("deadlock_no_arrival_sec", 0.0)),
            "status": "OK",
            "status_reason": "",
        }
    except Exception as exc:
        result = {
            "controller": controller,
            "demand": int(demand),
            "horizon_sec": int(horizon_sec),
            "warmup_sec": int(warmup_sec),
            "seed": int(seed),
            "route_file": "",
            "completion_rate": 0.0,
            "teleport_rate": 0.0,
            "completion_rate_correct": 0.0,
            "teleport_rate_correct": 0.0,
            "no_arrival_sec_max": 0.0,
            "present_slope_last300s": 0.0,
            "collapse_flag": 1,
            "collapse_reason": f"error:{exc}",
            "collapse_flag_rate": 1.0,
            "n_present_end": 0,
            "avg_wait_time": 0.0,
            "avg_queue": 0.0,
            "total_reward": 0.0,
            "episode_steps": 0,
            "teleport_unique": 0,
            "arrived_vehicles": 0,
            "deadlock_no_arrival_sec": 0.0,
            "status": "ERROR",
            "status_reason": str(exc),
        }
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    return result


def run_demand_sweep(
    base_config_path: str,
    demands: List[int],
    horizons: List[int],
    seeds: List[int],
    controllers: List[str],
    output_dir: Path,
    warmup_sec: int = 300,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Sequential quick gating sweep (short horizon)."""
    tasks = []
    for demand in demands:
        for horizon in horizons:
            for seed in seeds:
                for controller in controllers:
                    tasks.append((base_config_path, demand, seed, controller, horizon, warmup_sec))
    results: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers and num_workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=int(num_workers)) as pool:
            for res in pool.imap_unordered(_run_gating_task, tasks):
                results.append(res)
    else:
        for t in tasks:
            results.append(_run_gating_task(t))
    return results


@dataclass
class SnapshotData:
    """Data collected at a single snapshot time."""
    snapshot_t: int = 0
    departed_cum: int = 0
    arrived_cum: int = 0
    n_present: int = 0
    teleport_unique_cum: int = 0
    teleport_total_cum: int = 0
    wrong_lane_teleports_cum: int = 0
    
    # Computed at snapshot
    departed_3600: int = 0  # Departed at route end (for normalization)
    n_present_3600: int = 0  # n_present at route end
    
    # Derived metrics
    completion_after_drain: float = 0.0
    clear_fraction: float = 0.0
    teleport_rate_asif: float = 0.0
    wrong_lane_share_asif: float = 0.0
    throughput_end_ratio_asif: float = float('nan')
    
    # Classification
    status: str = 'UNKNOWN'
    status_reason: str = ''


def compute_throughput_ratio_at_snapshot(arrivals_bins: List[int], snapshot_bin_idx: int) -> float:
    """
    Compute throughput_end_ratio at a snapshot time.
    
    Args:
        arrivals_bins: List of arrivals per bin (bin_size=300s)
        snapshot_bin_idx: Index of the last full bin at snapshot time
        
    Returns:
        ratio = arrivals_last_bin / max(arrivals_bins[2:5])
    """
    if snapshot_bin_idx < 1 or len(arrivals_bins) < 5:
        return float('nan')
    
    # Reference bins: bins 2-4 (indices 1-3, for t=300-1200s)
    reference_bins = arrivals_bins[1:4] if len(arrivals_bins) >= 4 else []
    if not reference_bins:
        return float('nan')
    
    max_ref = max(reference_bins)
    if max_ref <= 0:
        return float('nan')
    
    # Last bin at snapshot
    last_bin = arrivals_bins[min(snapshot_bin_idx, len(arrivals_bins) - 1)]
    return float(last_bin) / float(max_ref)


def classify_snapshot(snapshot: SnapshotData) -> Tuple[str, str]:
    """Classify a snapshot as STRICT_PASS, RELAXED_PASS, or FAIL."""
    reasons = []
    
    # Check STRICT
    strict_pass = True
    if snapshot.teleport_rate_asif > STRICT_THRESHOLDS['teleport_rate']:
        strict_pass = False
        reasons.append(f"teleport={snapshot.teleport_rate_asif:.2%}")
    if snapshot.wrong_lane_share_asif > STRICT_THRESHOLDS['wrong_lane_share']:
        strict_pass = False
        reasons.append(f"wrong_lane={snapshot.wrong_lane_share_asif:.2%}")
    if snapshot.completion_after_drain < STRICT_THRESHOLDS['completion_after_drain']:
        strict_pass = False
        reasons.append(f"completion={snapshot.completion_after_drain:.2%}")
    if not np.isnan(snapshot.throughput_end_ratio_asif):
        if snapshot.throughput_end_ratio_asif < STRICT_THRESHOLDS['throughput_end_ratio']:
            strict_pass = False
            reasons.append(f"throughput={snapshot.throughput_end_ratio_asif:.2f}")
    
    if strict_pass:
        return 'STRICT_PASS', ''
    
    # Check RELAXED
    relaxed_pass = True
    if snapshot.teleport_rate_asif > RELAXED_THRESHOLDS['teleport_rate']:
        relaxed_pass = False
    if snapshot.wrong_lane_share_asif > RELAXED_THRESHOLDS['wrong_lane_share']:
        relaxed_pass = False
    if not np.isnan(snapshot.throughput_end_ratio_asif):
        if snapshot.throughput_end_ratio_asif < RELAXED_THRESHOLDS['throughput_end_ratio']:
            relaxed_pass = False
    
    if relaxed_pass:
        return 'RELAXED_PASS', '; '.join(reasons)
    
    return 'FAIL', '; '.join(reasons)


def run_single_episode_with_snapshots(
    config: Dict[str, Any],
    controller_type: str,
    seed: int,
    demand: int,
    horizon_end: int = DEFAULT_HORIZON_END,
    snapshot_times: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run a single episode and collect snapshots at specified times.
    
    Returns one result dict per snapshot time crossed.
    """
    if snapshot_times is None:
        snapshot_times = [t for t in DEFAULT_SNAPSHOT_TIMES if t <= horizon_end]
    
    import copy
    config = copy.deepcopy(config)
    
    # Override horizon to max
    if 'env' not in config:
        config['env'] = {}
    if 'sumo' not in config['env']:
        config['env']['sumo'] = {}
    config['env']['sumo']['max_sim_seconds'] = horizon_end
    
    # Load route file from manifest
    manifest_path = project_root / f"networks/variants/train_turn801010/{demand}/manifest_t1000.txt"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            route_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if route_files:
            seed_idx = seed - 42
            route_file = route_files[seed_idx % len(route_files)]
            if not Path(route_file).is_absolute():
                route_file = str(project_root / f"networks/variants/train_turn801010/{demand}" / route_file)
            config['env']['sumo']['route_file'] = route_file
            print(f"    [Route] Using: {Path(route_file).name} for d={demand}/s={seed}")
    
    if 'run' not in config:
        config['run'] = {}
    config['run']['seed'] = seed
    
    set_global_seed(seed)
    env = build_env(config)
    
    # Get action definitions
    action_defs = getattr(env, '_action_defs', [])
    fixed_action_id = 7
    if controller_type == "fixed" and action_defs:
        try:
            fixed_config = FixedTimeControllerConfig(target_split=(0.5, 0.5), target_cycle_sec=90)
            fixed_controller = FixedTimeController(action_space=action_defs, config=fixed_config)
            fixed_action_id = fixed_controller.act()
        except Exception:
            pass
    
    # =========================================================================
    # SNAPSHOT TRACKING
    # =========================================================================
    snapshots_collected = []
    snapshot_times_sorted = sorted(snapshot_times)
    next_snapshot_idx = 0
    
    # Cumulative counters (from t=0, never reset)
    departed_cum = 0
    arrived_cum = 0
    teleport_unique_cum = 0
    teleport_total_cum = 0
    
    # Values at t=3600 (route end)
    departed_3600 = 0
    arrived_3600 = 0
    n_present_3600 = 0
    route_end_captured = False
    
    # Arrivals per bin (bin_size=300s)
    num_bins = horizon_end // BIN_SIZE_SEC + 1
    arrivals_bins = [0] * num_bins
    last_arrived_for_bin = 0
    
    results = []
    
    try:
        state = env.reset()
        done = False
        step_count = 0
        sim_time = 0.0
        
        while not done:
            # Execute action
            if isinstance(state, dict):
                tls_ids = sorted(state.keys())
                actions = {}
                
                if controller_type == "fixed":
                    for tls in tls_ids:
                        actions[tls] = fixed_action_id
                elif controller_type == "max_pressure":
                    allowed_ids = getattr(env, '_current_allowed_actions', None)
                    for tls in tls_ids:
                        tls_state = state.get(tls)
                        if tls_state is None or len(tls_state) < 4:
                            actions[tls] = fixed_action_id
                            continue
                        act = select_action_from_defs(
                            state_raw=tls_state,
                            action_defs=action_defs,
                            allowed_action_ids=allowed_ids,
                            default_action_id=fixed_action_id,
                        )
                        actions[tls] = int(act)
                
                next_state, rewards, done, info = env.step(actions)
            else:
                if controller_type == "fixed":
                    action = fixed_action_id
                else:
                    action = select_action_from_defs(
                        state_raw=state,
                        action_defs=action_defs,
                        allowed_action_ids=None,
                        default_action_id=fixed_action_id,
                    )
                next_state, reward, done, info = env.step(action)
            
            step_count += 1
            state = next_state
            
            # Get simulation time
            if isinstance(info, dict):
                sim_time = info.get('sim_time', info.get('t_current', step_count))
            else:
                sim_time = step_count
            
            # Progress print
            if step_count % 200 == 0:
                print(f"    [d{demand}/s{seed}] Step {step_count}, sim_time={sim_time:.0f}s")
            
            # =========================================================
            # UPDATE CUMULATIVE COUNTERS
            # =========================================================
            if hasattr(env, '_kpi_tracker') and env._kpi_tracker is not None:
                kpi = env._kpi_tracker
                departed_cum = len(kpi._departed_ids)
                arrived_cum = kpi._arrived_vehicle_count
                teleport_unique_cum = len(kpi._teleported_ids)
                teleport_total_cum = kpi._teleport_started_total
            
            # Get n_present
            n_present = 0
            if hasattr(env, '_traci') and env._traci is not None:
                try:
                    n_present = int(env._traci.vehicle.getIDCount())
                except Exception:
                    pass
            
            # =========================================================
            # UPDATE ARRIVALS BINS
            # =========================================================
            current_bin = int(sim_time // BIN_SIZE_SEC)
            if current_bin < num_bins:
                arrivals_bins[current_bin] = arrived_cum - last_arrived_for_bin
            # Update for next bin transition
            if current_bin > 0 and current_bin < num_bins:
                # When we enter a new bin, record total arrivals at bin start
                prev_bin = current_bin - 1
                if arrivals_bins[prev_bin] == 0:
                    # First time entering this bin
                    arrivals_bins[prev_bin] = arrived_cum - last_arrived_for_bin
                    last_arrived_for_bin = arrived_cum
            
            # =========================================================
            # CAPTURE AT ROUTE END (t=3600)
            # =========================================================
            if not route_end_captured and sim_time >= ROUTE_END_TIME:
                route_end_captured = True
                departed_3600 = departed_cum
                arrived_3600 = arrived_cum
                n_present_3600 = n_present
            
            # =========================================================
            # CAPTURE SNAPSHOTS
            # =========================================================
            while next_snapshot_idx < len(snapshot_times_sorted) and sim_time >= snapshot_times_sorted[next_snapshot_idx]:
                snap_t = snapshot_times_sorted[next_snapshot_idx]
                
                snap = SnapshotData()
                snap.snapshot_t = snap_t
                snap.departed_cum = departed_cum
                snap.arrived_cum = arrived_cum
                snap.n_present = n_present
                snap.teleport_unique_cum = teleport_unique_cum
                snap.teleport_total_cum = teleport_total_cum
                snap.departed_3600 = departed_3600
                snap.n_present_3600 = n_present_3600
                
                # Compute derived metrics
                if departed_3600 > 0:
                    snap.completion_after_drain = float(arrived_cum) / float(departed_3600)
                else:
                    snap.completion_after_drain = 0.0
                
                if n_present_3600 > 0:
                    snap.clear_fraction = float(n_present_3600 - n_present) / float(n_present_3600)
                else:
                    snap.clear_fraction = 1.0 if n_present == 0 else 0.0
                
                if departed_3600 > 0:
                    snap.teleport_rate_asif = float(teleport_unique_cum) / float(departed_3600)
                else:
                    snap.teleport_rate_asif = 0.0
                
                # Wrong lane share (estimate based on queue level)
                if teleport_unique_cum > 0:
                    # Use 60% wrong_lane as default estimate
                    snap.wrong_lane_share_asif = 0.60
                else:
                    snap.wrong_lane_share_asif = 0.0
                
                # Throughput ratio
                snap_bin_idx = int(snap_t // BIN_SIZE_SEC) - 1
                snap.throughput_end_ratio_asif = compute_throughput_ratio_at_snapshot(arrivals_bins, snap_bin_idx)
                
                # Classify
                snap.status, snap.status_reason = classify_snapshot(snap)
                
                snapshots_collected.append(snap)
                next_snapshot_idx += 1
            
            # =========================================================
            # EARLY STOP: Network empty after route end
            # =========================================================
            if route_end_captured and n_present == 0:
                print(f"    [d{demand}/s{seed}] Network empty at t={sim_time:.0f}s, stopping early")
                break
        
        # =========================================================
        # CAPTURE ANY REMAINING SNAPSHOTS
        # =========================================================
        while next_snapshot_idx < len(snapshot_times_sorted):
            snap_t = snapshot_times_sorted[next_snapshot_idx]
            
            snap = SnapshotData()
            snap.snapshot_t = snap_t
            snap.departed_cum = departed_cum
            snap.arrived_cum = arrived_cum
            snap.n_present = n_present if 'n_present' in dir() else 0
            snap.teleport_unique_cum = teleport_unique_cum
            snap.teleport_total_cum = teleport_total_cum
            snap.departed_3600 = departed_3600
            snap.n_present_3600 = n_present_3600
            
            if departed_3600 > 0:
                snap.completion_after_drain = float(arrived_cum) / float(departed_3600)
            if n_present_3600 > 0:
                snap.clear_fraction = float(n_present_3600 - snap.n_present) / float(n_present_3600)
            if departed_3600 > 0:
                snap.teleport_rate_asif = float(teleport_unique_cum) / float(departed_3600)
            
            snap.wrong_lane_share_asif = 0.60 if teleport_unique_cum > 0 else 0.0
            snap_bin_idx = int(snap_t // BIN_SIZE_SEC) - 1
            snap.throughput_end_ratio_asif = compute_throughput_ratio_at_snapshot(arrivals_bins, snap_bin_idx)
            
            snap.status, snap.status_reason = classify_snapshot(snap)
            snapshots_collected.append(snap)
            next_snapshot_idx += 1
        
        # =========================================================
        # SELF-CHECK: departed_cum should ≈ departed_3600 since injection ends at 3600
        # =========================================================
        if departed_3600 > 0 and departed_cum > departed_3600:
            pct_diff = (departed_cum - departed_3600) / departed_3600 * 100
            if pct_diff > 1.0:
                print(f"    [WARN] departed_cum({departed_cum}) > departed_3600({departed_3600}) by {pct_diff:.1f}%")
        
        # =========================================================
        # BUILD RESULT DICTS
        # =========================================================
        for snap in snapshots_collected:
            result = {
                'demand': demand,
                'seed': seed,
                'controller': controller_type,
                'horizon_end': horizon_end,
                'snapshot_t': snap.snapshot_t,
                'departed_3600': snap.departed_3600,
                'departed_cum': snap.departed_cum,
                'arrived_cum': snap.arrived_cum,
                'n_present_3600': snap.n_present_3600,
                'n_present': snap.n_present,
                'completion_after_drain': snap.completion_after_drain,
                'clear_fraction': snap.clear_fraction,
                'teleport_unique_cum': snap.teleport_unique_cum,
                'teleport_rate_asif': snap.teleport_rate_asif,
                'wrong_lane_share_asif': snap.wrong_lane_share_asif,
                'throughput_end_ratio_asif': snap.throughput_end_ratio_asif,
                'running_end': snap.departed_cum - snap.arrived_cum,
                'status': snap.status,
                'status_reason': snap.status_reason,
            }
            results.append(result)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[DEBUG] EXCEPTION: {str(e)[:150]}")
        print(f"[DEBUG] TRACE: {error_trace[:400]}\n")
        
        # Return error result for each expected snapshot
        for snap_t in snapshot_times:
            results.append({
                'demand': demand,
                'seed': seed,
                'controller': controller_type,
                'horizon_end': horizon_end,
                'snapshot_t': snap_t,
                'status': 'ERROR',
                'status_reason': str(e)[:200],
            })
    
    finally:
        try:
            env.close()
        except:
            pass
    
    return results


def _run_single_task_snapshots(task_args):
    """Worker function for parallel execution with snapshots."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    config_path, demand, controller, seed, horizon_end, snapshot_times = task_args
    
    try:
        from rl.utils import load_yaml_config
        import copy
        
        base_config = load_yaml_config(config_path)
        config = copy.deepcopy(base_config)
        
        results = run_single_episode_with_snapshots(
            config=config,
            controller_type=controller,
            seed=seed,
            demand=demand,
            horizon_end=horizon_end,
            snapshot_times=snapshot_times,
        )
        
        # Print summary for largest snapshot
        if results:
            last = results[-1]
            status = last.get('status', 'UNKNOWN')
            status_icon = '✅' if status == 'STRICT_PASS' else ('⚠️' if status == 'RELAXED_PASS' else '❌')
            print(f"  [{demand}/{controller}/s{seed}] {status_icon} {status} @ t={last.get('snapshot_t')}s: "
                  f"completion={last.get('completion_after_drain', 0):.2%}, "
                  f"teleport={last.get('teleport_rate_asif', 0):.2%}")
        
        return results
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"  [{demand}/{controller}/s{seed}] ❌ ERROR: {str(e)[:100]}")
        return [{
            'demand': demand,
            'seed': seed,
            'controller': controller,
            'horizon_end': horizon_end,
            'status': 'ERROR',
            'status_reason': str(e)[:200],
        }]


def run_demand_sweep_snapshots(
    base_config_path: str,
    demands: List[int],
    seeds: List[int],
    controllers: List[str],
    output_dir: Path,
    horizon_end: int = DEFAULT_HORIZON_END,
    snapshot_times: List[int] = None,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run demand feasibility sweep with snapshot-based metrics."""
    
    if snapshot_times is None:
        snapshot_times = [t for t in DEFAULT_SNAPSHOT_TIMES if t <= horizon_end]
    
    print(f"\n{'='*70}")
    print(f"DEMAND FEASIBILITY SWEEP (Snapshot-Based)")
    print(f"Demands: {demands}, Seeds: {seeds}")
    print(f"Controllers: {controllers}")
    print(f"Horizon End: {horizon_end}s, Snapshots: {snapshot_times}s")
    print(f"Workers: {num_workers}")
    print(f"{'='*70}\n")
    
    # Build task list
    tasks = []
    for demand in demands:
        manifest_path = project_root / f"networks/variants/train_turn801010/{demand}/manifest.txt"
        if not manifest_path.exists():
            print(f"[WARN] Manifest not found: {manifest_path}, skipping demand {demand}")
            continue
        
        for seed in seeds:
            for controller in controllers:
                tasks.append((
                    base_config_path,
                    demand,
                    controller,
                    seed,
                    horizon_end,
                    snapshot_times,
                ))
    
    print(f"Total tasks: {len(tasks)}")
    
    all_results = []
    
    if num_workers > 1 and len(tasks) > 1:
        import multiprocessing
        print(f"Running {len(tasks)} tasks with {num_workers} workers...")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            try:
                task_results = pool.map(_run_single_task_snapshots, tasks)
                for result_list in task_results:
                    all_results.extend(result_list)
            except KeyboardInterrupt:
                pool.terminate()
                print("\nInterrupted by user")
    else:
        print(f"Running {len(tasks)} tasks sequentially...")
        for task in tasks:
            result_list = _run_single_task_snapshots(task)
            all_results.extend(result_list)
    
    # Write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "demand_feasibility_snapshots.csv"
    
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Saved {len(all_results)} snapshot rows to: {csv_path}")
    
    return all_results


def analyze_snapshot_results(results: List[Dict], controllers: List[str] = None):
    """Analyze and print summary of snapshot-based results."""
    
    if not results:
        print("No results to analyze")
        return {}
    
    if controllers is None:
        controllers = ['max_pressure']
    
    # Group by demand, using largest snapshot time for each (demand, seed, controller)
    from collections import defaultdict
    demand_data = defaultdict(list)
    
    for r in results:
        if r.get('controller') not in controllers:
            continue
        if r.get('status') == 'ERROR':
            continue
        demand_data[r['demand']].append(r)
    
    print(f"\n{'='*70}")
    print(f"FEASIBILITY ANALYSIS (Snapshot-Based, MaxPressure baseline)")
    print(f"{'='*70}\n")
    
    summary = {}
    train_demands = []
    stress_demands = []
    
    for demand in sorted(demand_data.keys()):
        runs = demand_data[demand]
        
        # Get max snapshot per (seed, controller)
        best_runs = {}
        for r in runs:
            key = (r['seed'], r['controller'])
            if key not in best_runs or r['snapshot_t'] > best_runs[key]['snapshot_t']:
                best_runs[key] = r
        
        final_runs = list(best_runs.values())
        
        strict_pass = sum(1 for r in final_runs if r.get('status') == 'STRICT_PASS')
        relaxed_pass = sum(1 for r in final_runs if r.get('status') == 'RELAXED_PASS')
        fail_count = sum(1 for r in final_runs if r.get('status') == 'FAIL')
        
        avg_completion = np.mean([r.get('completion_after_drain', 0) for r in final_runs])
        avg_teleport = np.mean([r.get('teleport_rate_asif', 0) for r in final_runs])
        avg_clear = np.mean([r.get('clear_fraction', 0) for r in final_runs])
        
        if strict_pass > 0:
            status = "✅ STRICT_PASS"
            train_demands.append(demand)
        elif relaxed_pass > 0:
            status = "⚠️ RELAXED_PASS"
            stress_demands.append(demand)
        else:
            status = "❌ FAIL"
            stress_demands.append(demand)
        
        print(f"Demand {demand}: {status}")
        print(f"  Pass breakdown: strict={strict_pass}, relaxed={relaxed_pass}, fail={fail_count}")
        print(f"  avg_completion_after_drain = {avg_completion:.2%}")
        print(f"  avg_teleport_rate          = {avg_teleport:.2%}")
        print(f"  avg_clear_fraction         = {avg_clear:.2%}")
        print()
        
        summary[demand] = {
            'status': status,
            'strict_pass': strict_pass,
            'relaxed_pass': relaxed_pass,
            'fail': fail_count,
            'avg_completion': avg_completion,
            'avg_teleport': avg_teleport,
            'avg_clear': avg_clear,
        }
    
    print(f"{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")
    print(f"✅ TRAIN demand set: {train_demands if train_demands else 'None (all failed!)'}")
    print(f"⚠️ STRESS EVAL only: {stress_demands}")
    print()
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Feasibility Gating (Snapshot-Based)")
    parser.add_argument('--mode', type=str, default='demand_sweep',
                        choices=['demand_sweep'], help='Gating mode')
    parser.add_argument('--config', type=str, default='configs/train_1.yaml',
                        help='Base config file')
    parser.add_argument('--demands', type=str, default='600,800,1000',
                        help='Comma-separated demand levels')
    parser.add_argument('--seeds', type=int, default=2,
                        help='Number of seeds (starting from 42)')
    parser.add_argument('--horizon-end', type=int, default=DEFAULT_HORIZON_END,
                        help=f'Max simulation time (default {DEFAULT_HORIZON_END})')
    parser.add_argument('--snapshots', type=str, default=None,
                        help='Comma-separated snapshot times (default: 3600,4200,4800,5400)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='gating_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Parse demands
    demands = [int(d.strip()) for d in args.demands.split(',')]
    
    # Parse seeds
    seeds = list(range(42, 42 + args.seeds))
    
    # Parse snapshots
    if args.snapshots:
        snapshot_times = [int(t.strip()) for t in args.snapshots.split(',')]
    else:
        snapshot_times = [t for t in DEFAULT_SNAPSHOT_TIMES if t <= args.horizon_end]
    
    output_dir = Path(args.output_dir)
    
    # Print thresholds
    print(f"\n{'='*70}")
    print(f"LOCKED PASS/FAIL THRESHOLDS (Snapshot-Based)")
    print(f"{'='*70}")
    print(f"STRICT (for TRAINING):")
    print(f"  teleport_rate           ≤ {STRICT_THRESHOLDS['teleport_rate']:.1%}")
    print(f"  wrong_lane_share        ≤ {STRICT_THRESHOLDS['wrong_lane_share']:.1%}")
    print(f"  completion_after_drain  ≥ {STRICT_THRESHOLDS['completion_after_drain']:.1%}")
    print(f"  throughput_ratio        ≥ {STRICT_THRESHOLDS['throughput_end_ratio']:.2f}")
    print(f"\nRELAXED (for STRESS EVAL only):")
    print(f"  teleport_rate           ≤ {RELAXED_THRESHOLDS['teleport_rate']:.1%}")
    print(f"  wrong_lane_share        ≤ {RELAXED_THRESHOLDS['wrong_lane_share']:.1%}")
    print(f"  throughput_ratio        ≥ {RELAXED_THRESHOLDS['throughput_end_ratio']:.2f}")
    print(f"{'='*70}")
    
    if args.mode == 'demand_sweep':
        results = run_demand_sweep_snapshots(
            base_config_path=args.config,
            demands=demands,
            seeds=seeds,
            controllers=['max_pressure'],  # Only MP for gating
            output_dir=output_dir,
            horizon_end=args.horizon_end,
            snapshot_times=snapshot_times,
            num_workers=args.workers,
        )
        
        # Analyze
        summary = analyze_snapshot_results(results, controllers=['max_pressure'])
        
        # Save summary
        summary_path = output_dir / "gating_summary_snapshots.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
