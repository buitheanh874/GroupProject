from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from env.kpi import EpisodeKpiTracker


class MockTraciSimulation:
    def __init__(self):
        self._time = 0.0
        self._departed = []
        self._arrived = []
        self._teleports = []
        self._min_expected = 100

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

    def getMinExpectedNumber(self):
        return int(self._min_expected)


class MockTraciVehicle:
    def __init__(self):
        self._speeds = {}
        self._ids = []

    def getIDList(self):
        return list(self._ids)

    def getSpeed(self, vid):
        return float(self._speeds.get(vid, 0.0))


class MockTraciLane:
    def __init__(self):
        self._halting = {}
        self._ids = []

    def getIDList(self):
        return list(self._ids)

    def getLastStepHaltingNumber(self, lane_id):
        return int(self._halting.get(lane_id, 0))

    def getLastStepOccupancy(self, lane_id):
        return 0.0


class MockTraciEdge:
    def __init__(self):
        self._ids = []

    def getIDList(self):
        return list(self._ids)

    def getLastStepOccupancy(self, edge_id):
        return 0.0


class MockTraci:
    def __init__(self):
        self.simulation = MockTraciSimulation()
        self.vehicle = MockTraciVehicle()
        self.lane = MockTraciLane()
        self.edge = MockTraciEdge()


def test_deadlock_trigger_with_no_arrivals():
    from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
    from env.normalization import StateNormalizer

    config = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="networks/BIGNET.net.xml",
        route_file="networks/variants/train/bignet_train_seed00042.rou.xml",
        tls_id="CENTER",
        max_sim_seconds=10,
        step_length_sec=1.0,
        terminate_on_empty=False,
        deadlock_no_arrival_sec=3.0,
        deadlock_queue_threshold=0.0,
        deadlock_downstream_occ_threshold=0.0,
        deadlock_active_min=0,
        deadlock_penalty=50.0,
        terminate_on_deadlock=True,
    )

    lanes = SumoLaneGroups(
        lanes_ns_ctrl=["N_lane"],
        lanes_ew_ctrl=["E_lane"],
    )

    phases = SumoPhaseProgram(ns_green=0, ew_green=1)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)

    env = SUMOEnv(config=config, lanes=lanes, phases=phases, normalizer=normalizer)

    env._no_arrival_steps = 0
    env._deadlock_triggered = False
    env._deadlock_reason = ""

    for _ in range(4):
        env._no_arrival_steps += 1

    penalty, terminate = env._process_deadlock_step(decision_teleport_count=0)

    assert env._deadlock_triggered is True
    assert env._deadlock_reason == "no_progress_congestion"
    assert terminate is True
    assert math.isclose(penalty, 50.0)


def test_deadlock_no_trigger_when_low_active():
    from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
    from env.normalization import StateNormalizer

    config = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="networks/BIGNET.net.xml",
        route_file="networks/variants/train/bignet_train_seed00042.rou.xml",
        tls_id="CENTER",
        max_sim_seconds=10,
        step_length_sec=1.0,
        terminate_on_empty=False,
        deadlock_no_arrival_sec=3.0,
        deadlock_queue_threshold=0.0,
        deadlock_downstream_occ_threshold=0.0,
        deadlock_active_min=100,
        deadlock_penalty=50.0,
        terminate_on_deadlock=True,
    )

    lanes = SumoLaneGroups(
        lanes_ns_ctrl=["N_lane"],
        lanes_ew_ctrl=["E_lane"],
    )

    phases = SumoPhaseProgram(ns_green=0, ew_green=1)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)

    env = SUMOEnv(config=config, lanes=lanes, phases=phases, normalizer=normalizer)

    mock_traci = MockTraci()
    mock_traci.simulation._min_expected = 5
    env._traci = mock_traci

    env._no_arrival_steps = 10
    env._deadlock_triggered = False
    env._deadlock_reason = ""

    penalty, terminate = env._process_deadlock_step(decision_teleport_count=0)

    assert env._deadlock_triggered is False
    assert terminate is False
    assert penalty == 0.0


def test_teleport_under_congestion_failure():
    from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
    from env.normalization import StateNormalizer

    config = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="networks/BIGNET.net.xml",
        route_file="networks/variants/train/bignet_train_seed00042.rou.xml",
        tls_id="CENTER",
        max_sim_seconds=10,
        step_length_sec=1.0,
        terminate_on_empty=False,
        deadlock_no_arrival_sec=100.0,
        deadlock_queue_threshold=0.0,
        deadlock_downstream_occ_threshold=0.0,
        deadlock_active_min=0,
        deadlock_penalty=75.0,
        terminate_on_deadlock=True,
        teleport_failure_when_congested=True,
    )

    lanes = SumoLaneGroups(
        lanes_ns_ctrl=["N_lane"],
        lanes_ew_ctrl=["E_lane"],
    )

    phases = SumoPhaseProgram(ns_green=0, ew_green=1)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)

    env = SUMOEnv(config=config, lanes=lanes, phases=phases, normalizer=normalizer)

    env._no_arrival_steps = 0
    env._deadlock_triggered = False
    env._deadlock_reason = ""

    penalty, terminate = env._process_deadlock_step(decision_teleport_count=3)

    assert env._deadlock_triggered is True
    assert env._deadlock_reason == "teleport_under_congestion"
    assert terminate is True
    assert math.isclose(penalty, 75.0)


def test_kpi_deadlock_fields():
    tracker = EpisodeKpiTracker(stop_speed_threshold=0.5)

    tracker.set_deadlock_info(triggered=True, reason="test_deadlock", no_arrival_sec=120.5)

    summary = tracker.summary()
    assert summary.deadlock_triggered == 1
    assert summary.deadlock_reason == "test_deadlock"
    assert math.isclose(summary.deadlock_no_arrival_sec, 120.5)

    summary_dict = tracker.summary_dict()
    assert summary_dict["deadlock_triggered"] == 1
    assert summary_dict["deadlock_reason"] == "test_deadlock"
    assert math.isclose(summary_dict["deadlock_no_arrival_sec"], 120.5)


def test_eval_csv_has_deadlock_columns():
    from scripts.eval import build_eval_row, build_failed_row

    kpi = {
        "arrived_vehicles": 100,
        "avg_wait_time": 10.0,
        "deadlock_triggered": 1,
        "deadlock_reason": "no_progress_congestion",
        "deadlock_no_arrival_sec": 150.0,
    }

    row = build_eval_row(
        controller="rl",
        scenario="test",
        run_id=0,
        total_reward=100.0,
        episode_steps=100,
        kpi=kpi,
    )

    assert "deadlock_triggered" in row
    assert "deadlock_reason" in row
    assert "deadlock_no_arrival_sec" in row
    assert row["deadlock_triggered"] == 1
    assert row["deadlock_reason"] == "no_progress_congestion"
    assert math.isclose(row["deadlock_no_arrival_sec"], 150.0)

    failed_row = build_failed_row(controller="rl", scenario="test", run_id=1)
    assert "deadlock_triggered" in failed_row
    assert "deadlock_reason" in failed_row
    assert "deadlock_no_arrival_sec" in failed_row
    assert failed_row["deadlock_triggered"] == 0
    assert failed_row["deadlock_reason"] == ""
    assert failed_row["deadlock_no_arrival_sec"] == 0.0


if __name__ == "__main__":
    test_deadlock_trigger_with_no_arrivals()
    test_deadlock_no_trigger_when_low_active()
    test_teleport_under_congestion_failure()
    test_kpi_deadlock_fields()
    test_eval_csv_has_deadlock_columns()
    print("test_deadlock_policy passed")
