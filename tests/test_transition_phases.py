from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.mdp_metrics import CycleMetricsAggregator


def test_transition_exclusion_basic():
    agg = CycleMetricsAggregator(directions=["NS", "EW"], queue_mode="distinct_cycle")
    
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=False)
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=False)
    
    waiting_sums = agg.waiting_sums(order=["NS", "EW"])
    
    assert abs(waiting_sums[0] - 2.0) < 1e-6
    assert abs(waiting_sums[1] - 0.0) < 1e-6


def test_transition_inclusion_conservative():
    agg = CycleMetricsAggregator(directions=["NS"], queue_mode="distinct_cycle")
    
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    
    waiting_sums = agg.waiting_sums(order=["NS"])
    
    assert abs(waiting_sums[0] - 3.0) < 1e-6


def test_queue_still_tracked_during_transition():
    agg = CycleMetricsAggregator(directions=["NS"], queue_mode="distinct_cycle")
    
    agg.observe("NS", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("NS", ["v1", "v2"], step_sec=1.0, accumulate_waiting=False)
    
    queue_counts = agg.queue_counts(order=["NS"])
    waiting_sums = agg.waiting_sums(order=["NS"])
    
    assert abs(queue_counts[0] - 2.0) < 1e-6
    assert abs(waiting_sums[0] - 1.0) < 1e-6


def test_mixed_phases_realistic_cycle():
    agg = CycleMetricsAggregator(directions=["NS", "EW"], queue_mode="distinct_cycle")
    
    for _ in range(30):
        agg.observe("NS", ["v1", "v2", "v3"], step_sec=1.0, accumulate_waiting=True)
    
    for _ in range(2):
        agg.observe("NS", ["v1", "v2", "v3"], step_sec=1.0, accumulate_waiting=False)
    
    for _ in range(30):
        agg.observe("EW", ["v4", "v5"], step_sec=1.0, accumulate_waiting=True)
    
    for _ in range(2):
        agg.observe("EW", ["v4", "v5"], step_sec=1.0, accumulate_waiting=False)
    
    waiting_sums = agg.waiting_sums(order=["NS", "EW"])
    
    assert abs(waiting_sums[0] - 90.0) < 1e-6
    assert abs(waiting_sums[1] - 60.0) < 1e-6
    
    total_wait = agg.waiting_total()
    assert abs(total_wait - 150.0) < 1e-6


def test_all_red_phase_behavior():
    agg = CycleMetricsAggregator(directions=["NS", "EW"], queue_mode="distinct_cycle")
    
    agg.observe("NS", [], step_sec=1.0, accumulate_waiting=True)
    agg.observe("EW", ["v1", "v2"], step_sec=1.0, accumulate_waiting=True)
    
    agg.observe("NS", ["v3"], step_sec=1.0, accumulate_waiting=False)
    agg.observe("EW", ["v1", "v2"], step_sec=1.0, accumulate_waiting=False)
    
    waiting_sums = agg.waiting_sums(order=["NS", "EW"])
    
    assert abs(waiting_sums[0] - 0.0) < 1e-6
    assert abs(waiting_sums[1] - 2.0) < 1e-6


if __name__ == "__main__":
    test_transition_exclusion_basic()
    test_transition_inclusion_conservative()
    test_queue_still_tracked_during_transition()
    test_mixed_phases_realistic_cycle()
    test_all_red_phase_behavior()
    
    print("test_transition_phases passed")