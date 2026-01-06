from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.mdp_metrics import CycleMetricsAggregator, compute_normalized_reward
from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
from env.normalization import StateNormalizer


def _multi_lanes() -> dict:
    return {
        "CENTER": SumoLaneGroups(lanes_ns_ctrl=["N2C_0"], lanes_ew_ctrl=["E2C_0"]),
        "N1": SumoLaneGroups(lanes_ns_ctrl=["N12C_0"], lanes_ew_ctrl=["N1E2C_0"]),
    }


def _multi_phases() -> SumoPhaseProgram:
    return SumoPhaseProgram(ns_green=0, ew_green=1, ns_yellow=None, ew_yellow=None, all_red=None)


def _base_multi_config(**overrides: object) -> dict:
    base = dict(
        sumo_binary="sumo",
        net_file="net.xml",
        route_file="route.rou.xml",
        route_pool=[],
        tls_id="CENTER",
        tls_ids=["CENTER", "N1"],
        center_tls_id="CENTER",
        downstream_links={},
        vehicle_weights={},
        step_length_sec=1.0,
        halt_speed_threshold=0.1,
        green_cycle_sec=60,
        yellow_sec=0,
        all_red_sec=0,
        max_cycles=1,
        max_sim_seconds=10,
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
        terminate_on_empty=True,
        sumo_extra_args=[],
        normalize_state=True,
        return_raw_state=False,
        enable_kpi_tracker=False,
        state_dim=12,
        enable_downstream_occupancy=False,
    )
    base.update(overrides)
    return base


def _make_multi_env() -> SUMOEnv:
    cfg = SumoEnvConfig(**_base_multi_config())
    lanes = _multi_lanes()
    phases = _multi_phases()
    normalizer = StateNormalizer(mean=[0.0] * 12, std=[1.0] * 12, expected_dim=12)
    return SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)


def test_action_defs_constraints():
    env = _make_multi_env()
    action_defs = env._action_defs
    cycles = {int(a.cycle_sec) for a in action_defs}
    assert cycles == {30, 60, 90}

    for action in action_defs:
        g_ns, g_ew = env._compute_green_split(action)
        min_green_sec = max(int(round(env._config.rho_min * action.cycle_sec)), int(env._config.g_min_sec))
        assert g_ns >= min_green_sec
        assert g_ew >= min_green_sec
        assert g_ns + g_ew == int(action.cycle_sec)


def test_default_multi_action_table_full_coverage():
    env = _make_multi_env()
    action_defs = env._action_defs
    assert len(action_defs) == 15

    expected_splits = {(0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)}
    for cycle in (30, 60, 90):
        splits_for_cycle = {
            (round(a.rho_ns, 2), round(a.rho_ew, 2))
            for a in action_defs
            if int(a.cycle_sec) == int(cycle)
        }
        assert splits_for_cycle == expected_splits

    cycle_map = env.cycle_to_actions
    assert set(cycle_map.keys()) == {30, 60, 90}
    for cycle, indices in cycle_map.items():
        assert len(indices) == 5
        assert all(int(action_defs[i].cycle_sec) == int(cycle) for i in indices)


def test_reward_normalization_basic():
    reward = compute_normalized_reward(wait_total=50.0, t_step=100.0, decision_cycle_sec=100.0)
    assert math.isclose(reward, -0.5)

    reward_penalized = compute_normalized_reward(wait_total=10.0, t_step=50.0, decision_cycle_sec=100.0, fairness_penalty=5.0)
    assert math.isclose(reward_penalized, -(10.0 + 5.0) / 50.0)


def test_distinct_cycle_queue_no_leak():
    agg = CycleMetricsAggregator(directions=["N"], queue_mode="distinct_cycle")
    agg.observe("N", ["v1", "v2"], step_sec=1.0, accumulate_waiting=False)
    assert math.isclose(agg.waiting_total(), 0.0)
    assert agg.queue_counts(order=["N"]).tolist() == [2.0]


def test_fairness_p95_max_match_semantics():
    agg = CycleMetricsAggregator(directions=["E"], queue_mode="distinct_cycle")
    agg.observe("E", ["a", "b", "c"], step_sec=1.0, accumulate_waiting=True)
    agg.observe("E", ["a", "b"], step_sec=1.0, accumulate_waiting=True)
    waits = [2.0, 2.0, 1.0]
    assert math.isclose(agg.fairness_value(metric="max"), max(waits))
    assert math.isclose(
        agg.fairness_value(metric="p95"),
        float(np.percentile(np.asarray(waits, dtype=np.float32), 95)),
    )


def test_env_rejects_snapshot_last_step_mode():
    cfg = SumoEnvConfig(**_base_multi_config(queue_count_mode="snapshot_last_step"))
    lanes = _multi_lanes()
    phases = _multi_phases()
    normalizer = StateNormalizer(mean=[0.0] * 12, std=[1.0] * 12, expected_dim=12)
    with pytest.raises(ValueError, match="snapshot_last_step"):
        SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)
