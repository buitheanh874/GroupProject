from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

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


def test_snapshot_mode_uses_last_step_only():
    agg = CycleMetricsAggregator(directions=["N"], queue_mode="snapshot_last_step")
    agg.observe("N", ["a"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("N", ["b"], step_sec=1.0, accumulate_waiting=True)

    counts = agg.queue_counts(order=["N"])
    assert counts.tolist() == [1.0]
    assert math.isclose(agg.waiting_total(), 2.0)


def test_fairness_p95_and_weighted_wait():
    agg = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg.observe("E", ["x1", "x2", "x3"], step_sec=2.0, accumulate_waiting=True)
    agg.observe("E", ["x1", "x2"], step_sec=1.0, accumulate_waiting=True)

    waits = [3.0, 3.0, 2.0]
    assert math.isclose(agg.fairness_value(metric="max"), 3.0)
    assert math.isclose(agg.fairness_value(metric="p95"), float(np.percentile(waits, 95)))

    weights = {"x1": 2.0, "x2": 1.0, "x3": 1.0}
    agg_weighted = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg_weighted.observe("E", ["x1", "x2"], step_sec=1.0, accumulate_waiting=True, weight_lookup=weights.get)
    unweighted = agg_weighted.waiting_total()
    weighted = agg_weighted.waiting_total(use_weights=True)
    assert math.isclose(unweighted, 2.0)
    assert math.isclose(weighted, 3.0)


if __name__ == "__main__":
    test_distinct_queue_counts_and_snapshot()
    test_snapshot_mode_uses_last_step_only()
    test_fairness_p95_and_weighted_wait()
    print("test_distinct_queue_aggregator passed")
