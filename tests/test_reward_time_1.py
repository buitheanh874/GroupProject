from __future__ import annotations

import pytest

from env.normalization import StateNormalizer
from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram


class FakeSimulation:
    def __init__(self, step: float):
        self.time = 0.0
        self.step = float(step)

    def getTime(self) -> float:
        return float(self.time)

    def getStartingTeleportIDList(self):
        return []

    def getArrivedIDList(self):
        return []

    def getMinExpectedNumber(self):
        return 1


class FakeLane:
    def getLastStepHaltingNumber(self, lane_id: str):
        return 0.0

    def getLastStepVehicleIDs(self, lane_id: str):
        return []

    def getLastStepOccupancy(self, lane_id: str):
        return 0.0


class FakeVehicle:
    def getSpeed(self, veh_id: str):
        return 0.0

    def getTypeID(self, veh_id: str):
        return "passenger"


class FakeTrafficLight:
    def setPhase(self, tls_id: str, phase_index: int):
        return None

    def setPhaseDuration(self, tls_id: str, duration: float):
        return None


class FakeTraci:
    def __init__(self, step: float):
        self.simulation = FakeSimulation(step)
        self.lane = FakeLane()
        self.vehicle = FakeVehicle()
        self.trafficlight = FakeTrafficLight()

    def simulationStep(self):
        self.simulation.time += float(self.simulation.step)


def _make_env(step_length: float = 1.0, yellow_sec: int = 0, all_red_sec: int = 0) -> SUMOEnv:
    cfg = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="net.xml",
        route_file="route.rou.xml",
        route_pool=[],
        tls_id="CENTER",
        tls_ids=[],
        center_tls_id=None,
        downstream_links={},
        vehicle_weights={},
        step_length_sec=float(step_length),
        halt_speed_threshold=0.1,
        green_cycle_sec=60,
        yellow_sec=int(yellow_sec),
        all_red_sec=int(all_red_sec),
        max_cycles=2,
        max_sim_seconds=None,
        seed=0,
        rho_min=0.1,
        g_min_sec=5,
        lambda_fairness=0.0,
        fairness_metric="max",
        action_splits=[],
        action_table=[],
        include_transition_in_waiting=False,
        queue_count_mode="distinct_cycle",
        use_pcu_weighted_wait=False,
        use_enhanced_reward=False,
        reward_exponent=1.0,
        enable_anti_flicker=False,
        kappa=0.0,
        enable_spillback_penalty=False,
        beta=0.0,
        occ_threshold=0.0,
        terminate_on_empty=False,
        sumo_extra_args=[],
        normalize_state=True,
        return_raw_state=False,
        enable_kpi_tracker=False,
        state_dim=4,
        enable_downstream_occupancy=False,
        teleport_penalty_lambda=0.0,
        teleport_time_cap_sec=None,
        deadlock_early_no_arrival_sec=0.0,
        deadlock_no_arrival_sec=0.0,
        deadlock_queue_threshold=0.0,
        deadlock_downstream_occ_threshold=0.0,
        deadlock_active_min=0,
        deadlock_early_penalty_max=0.0,
        deadlock_penalty=0.0,
        terminate_on_deadlock=False,
        teleport_failure_when_congested=False,
        cycle_options_sec=[],
        reward_time_normalize=True,
    )
    lanes = SumoLaneGroups(lanes_ns_ctrl=["N2C_0"], lanes_ew_ctrl=["E2C_0"])
    phases = SumoPhaseProgram(ns_green=3, ew_green=0, ns_yellow=4, ew_yellow=1, all_red=2)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)
    env = SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)
    env._traci = FakeTraci(step=float(step_length))
    env._connected = True
    return env


def test_reward_time_normalization_uses_sim_time(monkeypatch):
    env = _make_env()

    def fake_reward(*args, **kwargs):
        return 30.0

    monkeypatch.setattr("env.sumo_env.compute_normalized_reward", fake_reward)
    _, reward, done, info = env._step_legacy(0)

    assert not done
    assert info["decision_steps"] == 60
    assert info["decision_duration_sec"] == pytest.approx(60.0)
    assert info["decision_cycle_sec"] == pytest.approx(60.0)
    assert reward == pytest.approx(0.5)
    assert env._traci.simulation.getTime() == pytest.approx(60.0)


def test_t_step_includes_transitions(monkeypatch):
    env = _make_env(step_length=1.0, yellow_sec=2, all_red_sec=3)

    captured = {}

    def fake_reward(wait_total, t_step, decision_cycle_sec, fairness_penalty, spill_penalty, anti_flicker_penalty):
        captured["t_step"] = t_step
        captured["decision_cycle_sec"] = decision_cycle_sec
        return float(t_step)

    monkeypatch.setattr("env.sumo_env.compute_normalized_reward", fake_reward)
    _, reward, done, info = env._step_legacy(0)

    expected_t_step = 60 + 2 * 2 + 2 * 3

    assert not done
    assert captured["t_step"] == pytest.approx(expected_t_step)
    assert captured["decision_cycle_sec"] == pytest.approx(60.0)
    assert info["decision_duration_sec"] == pytest.approx(expected_t_step)
    assert info["t_step"] == pytest.approx(expected_t_step)
    assert reward == pytest.approx(expected_t_step)
