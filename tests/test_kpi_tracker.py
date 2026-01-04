from __future__ import annotations

import math

import numpy as np

from env.kpi import EpisodeKpiTracker


class FakeTraci:
    def __init__(self):
        self.simulation = self
        self.vehicle = self
        self._time = 0.0
        self._departed = []
        self._arrived = []
        self._speeds = {}

    def set_state(self, time: float, departed, arrived, speeds):
        self._time = float(time)
        self._departed = list(departed)
        self._arrived = list(arrived)
        self._speeds = dict(speeds)

    # simulation API
    def getTime(self):
        return float(self._time)

    def getDepartedIDList(self):
        return list(self._departed)

    def getArrivedIDList(self):
        return list(self._arrived)

    # vehicle API
    def getIDList(self):
        return list(self._speeds.keys())

    def getSpeed(self, vid):
        return float(self._speeds.get(vid, 0.0))


def test_episode_kpi_tracker_basic_summary():
    tracker = EpisodeKpiTracker(stop_speed_threshold=0.5)
    traci = FakeTraci()

    traci.set_state(time=0.0, departed=["v1", "v2"], arrived=[], speeds={"v1": 0.0, "v2": 2.0})
    tracker.on_simulation_step(traci, queue_length=2.0)

    traci.set_state(time=1.0, departed=[], arrived=["v2"], speeds={"v1": 0.0})
    tracker.on_simulation_step(traci, queue_length=1.0)

    traci.set_state(time=2.0, departed=[], arrived=["v1"], speeds={})
    tracker.on_simulation_step(traci, queue_length=0.0)

    summary = tracker.summary()
    assert summary.arrived_vehicles == 2
    assert math.isclose(summary.avg_wait_time, 1.0, rel_tol=1e-6)
    assert math.isclose(summary.avg_travel_time, 1.5, rel_tol=1e-6)
    assert math.isclose(summary.avg_stops, 0.5, rel_tol=1e-6)
    assert math.isclose(summary.avg_queue, 1.0, rel_tol=1e-6)

    waits = np.asarray([0.0, 2.0], dtype=np.float32)
    assert math.isclose(summary.max_wait_time, float(np.max(waits)), rel_tol=1e-6)
    assert math.isclose(summary.p95_wait_time, float(np.percentile(waits, 95)), rel_tol=1e-6)
