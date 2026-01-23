from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set

import numpy as np


@dataclass
class CycleMetricsAggregator:
    """Collect per-cycle queue membership and waiting time without SUMO dependencies."""

    directions: Iterable[str]
    queue_mode: str = "distinct_cycle"

    def __post_init__(self) -> None:
        dirs = [str(d).upper() for d in self.directions]
        if len(dirs) <= 0:
            raise ValueError("directions must not be empty")
        self._directions = sorted(set(dirs))
        mode = str(self.queue_mode).lower()
        
        if mode == "snapshot_last_step":
            raise ValueError(
                "queue_count_mode='snapshot_last_step' is no longer supported.\n"
                "MDP compliance requires 'distinct_cycle' mode.\n"
                "This mode tracks distinct vehicles queued at least once per cycle."
            )
        if mode not in {"distinct_cycle"}:
            raise ValueError(f"queue_count_mode must be 'distinct_cycle', got '{mode}'")
        
        self._queue_mode = mode
        self.reset()

    def reset(self) -> None:
        self._queued: Dict[str, Set[str]] = {d: set() for d in self._directions}
        self._snapshot: Dict[str, Set[str]] = {d: set() for d in self._directions}
        self._waiting: Dict[str, Dict[str, float]] = {d: {} for d in self._directions}
        self._weights: Dict[str, Dict[str, float]] = {d: {} for d in self._directions}

    def observe(
        self,
        direction: str,
        queued_vehicle_ids: Iterable[str],
        step_sec: float,
        accumulate_waiting: bool,
        weight_lookup: Optional[Callable[[str], float]] = None,
    ) -> None:
        dir_key = str(direction).upper()
        if dir_key not in self._queued:
            raise ValueError(f"Unknown direction key: {direction}")
        veh_set = {str(v) for v in queued_vehicle_ids}
        self._snapshot[dir_key] = veh_set
        if self._queue_mode == "distinct_cycle":
            self._queued[dir_key].update(veh_set)
        else:
            self._queued[dir_key] = veh_set

        if accumulate_waiting and float(step_sec) > 0.0:
            waits = self._waiting[dir_key]
            weights = self._weights[dir_key]
            for vid in veh_set:
                waits[vid] = waits.get(vid, 0.0) + float(step_sec)
                if vid not in weights:
                    weight = 1.0
                    if weight_lookup is not None:
                        try:
                            weight = float(weight_lookup(vid))
                        except Exception:
                            weight = 1.0
                    weights[vid] = weight
        else:
            if weight_lookup is not None:
                weights = self._weights[dir_key]
                for vid in veh_set:
                    if vid not in weights:
                        try:
                            weights[vid] = float(weight_lookup(vid))
                        except Exception:
                            weights[vid] = 1.0

    def queue_counts(self, order: Iterable[str]) -> np.ndarray:
        ordered = []
        for key in order:
            k = str(key).upper()
            counts = self._queued if self._queue_mode == "distinct_cycle" else self._snapshot
            ordered.append(float(len(counts.get(k, set()))))
        return np.asarray(ordered, dtype=np.float32)

    def snapshot_counts(self, order: Iterable[str]) -> np.ndarray:
        ordered = []
        for key in order:
            k = str(key).upper()
            ordered.append(float(len(self._snapshot.get(k, set()))))
        return np.asarray(ordered, dtype=np.float32)

    def waiting_total(self, exponent: float = 1.0, use_weights: bool = False) -> float:
        exp_val = max(1.0, float(exponent))
        total = 0.0
        for dir_key, waits in self._waiting.items():
            for vid, wait_time in waits.items():
                weight = 1.0
                if use_weights:
                    weight = float(self._weights.get(dir_key, {}).get(vid, 1.0))
                total += float(weight) * (float(wait_time) ** exp_val)
        return float(total)

    def waiting_sums(self, order: Iterable[str]) -> np.ndarray:
        values = []
        for key in order:
            dir_key = str(key).upper()
            waits = self._waiting.get(dir_key, {})
            values.append(float(sum(waits.values())))
        return np.asarray(values, dtype=np.float32)

    def fairness_value(self, metric: str = "max") -> float:
        metric_key = str(metric).lower()
        if metric_key not in {"max", "p95"}:
            raise ValueError("fairness_metric must be max or p95")
        
        waits_all = []
        for waits in self._waiting.values():
            waits_all.extend(float(w) for w in waits.values())

        if len(waits_all) == 0:
            return 0.0

        if metric_key == "p95":
            waits_arr = np.asarray(waits_all, dtype=np.float32)
            return float(np.percentile(waits_arr, 95))
        else:
            return float(max(waits_all))


def compute_normalized_reward_smdp(
    wait_total: float,
    delta_t: float,
    t_ref: float,
    spill_penalty: float = 0.0,
    n_present: int = 1,
    num_downstream: int = 4,
) -> float:
    """Compute SMDP-correct reward with time exposure scaling (v5 - FINAL).
    
    Formula: R = -W / (N * t_ref) - (spill / M) * (Δt / t_ref)
    
    SMDP Rationale:
    - Cycle is an action → decision duration (Δt) varies
    - Episode horizon is time-based (1800s) → different #steps per episode
    - Term 1: waiting time per vehicle, scaled by t_ref for O(1) reward
    - Term 2: spillback exposure × (Δt/t_ref) → longer exposure = more penalty
    - This prevents "cycle hack" where longer cycles get fewer penalty accumulations
    
    Args:
        wait_total: W_global = sum of waiting time across all TLS in step (veh-sec)
        delta_t: Δt = t_step_value = cycle + transitions (sec)
        t_ref: Reference time for scaling (typically 60s, same as time-aware gamma)
        spill_penalty: α * sum(downstream_occupancy^2), dimensionless
        n_present: Number of vehicles currently in network (from vehicle.getIDCount)
        num_downstream: M = number of downstream links (typically 4)
        
    Returns:
        Normalized reward, target scale O(1) regardless of demand or cycle choice
    """
    # Vehicle normalization for demand invariance
    n_veh = max(1, int(n_present))
    
    # Reference time for scaling (avoid division by zero)
    t_ref_safe = max(1.0, float(t_ref))
    delta_t_safe = max(1.0, float(delta_t))
    
    # Term 1: Waiting time per vehicle, scaled by t_ref
    # W_global is veh-sec in this step, divide by N gives avg per vehicle
    # Divide by t_ref for O(1) scale
    # Units: (veh-sec) / (veh * sec) = dimensionless
    wait_penalty = float(wait_total) / (float(n_veh) * t_ref_safe)
    
    # Term 2: Spillback exposure × time ratio
    # spill_penalty = α * sum(occ^2), dimensionless rate proxy
    # Multiply by (Δt/t_ref) to make it "exposure over time"
    # This ensures longer decisions with same spill get more penalty
    # Units: (dimensionless) * (sec/sec) = dimensionless
    M = max(1, int(num_downstream))
    spill_exposure = (float(spill_penalty) / float(M)) * (delta_t_safe / t_ref_safe)
    
    return -wait_penalty - spill_exposure


# Keep old function for backward compatibility but mark deprecated
def compute_normalized_reward(
    wait_total: float,
    t_step: float,
    decision_cycle_sec: float,
    spill_penalty: float = 0.0,
    n_present: int = 1,
    num_downstream: int = 4,
) -> float:
    """DEPRECATED: Use compute_normalized_reward_smdp instead.
    
    This function has SMDP bias when cycle is an action and horizon is time-based.
    """
    # Redirect to new function with t_ref = t_step (old behavior)
    return compute_normalized_reward_smdp(
        wait_total=wait_total,
        delta_t=t_step,
        t_ref=t_step,  # Old behavior: t_ref = Δt
        spill_penalty=spill_penalty,
        n_present=n_present,
        num_downstream=num_downstream,
    )

