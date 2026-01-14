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


def compute_normalized_reward(
    wait_total: float,
    t_step: float,
    decision_cycle_sec: float,
    spill_penalty: float = 0.0,
) -> float:
    """Compute normalized reward using simplified formula.
    
    Formula: R = -(wait_total + spill_penalty) / T
    
    Based on Varaiya 2013 (Back-Pressure) and PressLight (KDD 2019).
    The spill_penalty uses squared occupancy for smooth gradient:
    spill_penalty = alpha * sum(downstream_occupancy^2)
    
    Args:
        wait_total: Total accumulated waiting time in the cycle
        t_step: Normalization factor (cycle_sec + transitions)
        decision_cycle_sec: Actual cycle duration in seconds
        spill_penalty: Anti-congestion penalty from downstream occupancy
        
    Returns:
        Normalized reward (negative value, higher is better)
    """
    denom = float(t_step) if float(t_step) > 0.0 else float(decision_cycle_sec)
    denom = max(1.0, float(denom))
    return -float(wait_total + spill_penalty) / float(denom)
