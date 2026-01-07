from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.mdp_metrics import CycleMetricsAggregator


def test_distinct_queue_counts_and_snapshot():
    agg = CycleMetricsAggregator(directions=["N", "S"], queue_mode="distinct_cycle")
    agg.observe("N", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("N", ["v2"], step_sec=1.0, accumulate_waiting=True)

    counts = agg.queue_counts(order=["N", "S"])
    snapshot = agg.snapshot_counts(order=["N", "S"])

    assert counts.tolist() == [2.0, 0.0]
    assert snapshot.tolist() == [1.0, 0.0]
    assert math.isclose(agg.waiting_total(), 2.0)


def test_snapshot_last_step_rejected():
    with pytest.raises(ValueError, match="snapshot_last_step"):
        CycleMetricsAggregator(directions=["N"], queue_mode="snapshot_last_step")


def test_waiting_sums_ignore_non_accumulated_steps():
    agg = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg.observe("E", ["v1"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("E", ["v1", "v2"], step_sec=2.0, accumulate_waiting=False)

    waiting = agg.waiting_sums(order=["E"])
    snapshot_counts = agg.snapshot_counts(order=["E"])

    assert math.isclose(float(waiting[0]), 1.0)  # only the accumulated step counts
    assert snapshot_counts.tolist() == [2.0]


def test_fairness_p95_and_weighted_wait():
    agg = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg.observe("E", ["x1", "x2", "x3"], step_sec=2.0, accumulate_waiting=True)
    agg.observe("E", ["x1", "x2"], step_sec=1.0, accumulate_waiting=True)

    waits = [3.0, 3.0, 2.0]
    fairness_max = agg.fairness_value(metric="max")
    fairness_p95 = agg.fairness_value(metric="p95")
    assert math.isclose(fairness_max, max(waits))
    assert math.isclose(fairness_p95, float(np.percentile(np.asarray(waits, dtype=np.float32), 95)))

    weights = {"x1": 2.0, "x2": 1.0, "x3": 1.0}
    agg_weighted = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg_weighted.observe("E", ["x1", "x2"], step_sec=1.0, accumulate_waiting=True, weight_lookup=weights.get)
    unweighted = agg_weighted.waiting_total()
    weighted = agg_weighted.waiting_total(use_weights=True)
    assert math.isclose(unweighted, 2.0)
    assert math.isclose(weighted, 3.0)


def test_waiting_sums_and_transition_exclusion():
    agg = CycleMetricsAggregator(directions=["N", "S"], queue_mode="distinct_cycle")
    agg.observe("N", ["v1"], step_sec=0.5, accumulate_waiting=True)
    agg.observe("N", ["v1", "v2"], step_sec=0.5, accumulate_waiting=True)
    agg.observe("S", ["s1"], step_sec=0.5, accumulate_waiting=False)  # transition excluded
    sums = agg.waiting_sums(order=["N", "S"])
    # N: v1 waited 1.0s, v2 waited 0.5s
    assert math.isclose(float(sums[0]), 1.5)
    assert math.isclose(float(sums[1]), 0.0)
