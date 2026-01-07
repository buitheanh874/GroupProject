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
        self._teleports = []

    def set_state(self, time: float, departed, arrived, speeds, teleports=None):
        self._time = float(time)
        self._departed = list(departed)
        self._arrived = list(arrived)
        self._speeds = dict(speeds)
        self._teleports = list(teleports) if teleports is not None else []

    # simulation API
    def getTime(self):
        return float(self._time)

    def getDepartedIDList(self):
        return list(self._departed)

    def getArrivedIDList(self):
        return list(self._arrived)

    def getStartingTeleportIDList(self):
        return list(self._teleports)

    def getStartingTeleportNumber(self):
        return int(len(self._teleports))

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


def test_teleport_rate_uses_departed_denominator():
    tracker = EpisodeKpiTracker(stop_speed_threshold=0.5, teleport_time_cap_sec=100.0)
    traci = FakeTraci()

    traci.set_state(time=0.0, departed=["v1", "v2"], arrived=[], speeds={"v1": 0.0, "v2": 1.0})
    tracker.on_simulation_step(traci, queue_length=2.0)

    traci.set_state(time=1.0, departed=[], arrived=[], speeds={"v1": 0.0, "v2": 1.0}, teleports=["v1"])
    tracker.on_simulation_step(traci, queue_length=2.0)

    traci.set_state(time=2.0, departed=[], arrived=["v2"], speeds={"v1": 0.0}, teleports=[])
    tracker.on_simulation_step(traci, queue_length=1.0)

    summary = tracker.summary()
    assert summary.teleport_unique == 1
    assert math.isclose(summary.teleport_rate, 0.5, rel_tol=1e-6)
    assert summary.arrived_corr == 1
    assert summary.teleported_arrived == 0
    assert summary.failed_corr == 1
    assert math.isclose(summary.completion_rate, 0.5, rel_tol=1e-6)


def test_arrived_corr_excludes_teleported_ids():
    tracker = EpisodeKpiTracker(stop_speed_threshold=0.5, teleport_time_cap_sec=50.0)
    traci = FakeTraci()

    traci.set_state(time=0.0, departed=["v1"], arrived=[], speeds={"v1": 0.0})
    tracker.on_simulation_step(traci, queue_length=1.0)

    traci.set_state(time=1.0, departed=[], arrived=[], speeds={"v1": 0.0}, teleports=["v1"])
    tracker.on_simulation_step(traci, queue_length=1.0)

    traci.set_state(time=2.0, departed=[], arrived=["v1"], speeds={}, teleports=[])
    tracker.on_simulation_step(traci, queue_length=0.0)

    summary = tracker.summary()
    assert summary.arrived_vehicles == 1
    assert summary.teleport_unique == 1
    assert summary.arrived_corr == 0
    assert summary.teleported_arrived == 1
    assert summary.failed_corr == 1
    assert math.isclose(summary.completion_rate, 1.0, rel_tol=1e-6)
    assert summary.avg_wait_time_corr >= summary.avg_wait_time


def test_not_arrived_penalized_in_corrected_metrics():
    tracker = EpisodeKpiTracker(stop_speed_threshold=0.5, teleport_time_cap_sec=10.0)
    traci = FakeTraci()

    traci.set_state(time=0.0, departed=["v1"], arrived=[], speeds={"v1": 0.0})
    tracker.on_simulation_step(traci, queue_length=1.0)

    traci.set_state(time=3.0, departed=[], arrived=[], speeds={"v1": 0.0}, teleports=[])
    tracker.on_simulation_step(traci, queue_length=1.0)

    summary = tracker.summary()
    assert summary.arrived_vehicles == 0
    assert summary.arrived_corr == 0
    assert summary.failed_corr == 1
    assert summary.avg_wait_time == 0.0
    assert summary.avg_wait_time_corr >= 3.0
    assert summary.completion_rate == 0.0
