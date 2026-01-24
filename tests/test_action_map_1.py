from __future__ import annotations

import pytest

from env.normalization import StateNormalizer
from env.sumo_env import DEFAULT_ACTION_SPLITS, SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram


def _make_env(**overrides: object) -> SUMOEnv:
    base_cfg = dict(
        sumo_binary="sumo",
        net_file="net.xml",
        route_file="route.rou.xml",
        route_pool=[],
        tls_id="CENTER",
        tls_ids=[],
        center_tls_id=None,
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
        action_splits=[],
        action_table=[],
        include_transition_in_waiting=False,
        queue_count_mode="distinct_cycle",
        use_pcu_weighted_wait=False,
        use_enhanced_reward=False,
        reward_exponent=1.0,
        enable_spillback_penalty=False,
        alpha_spillback=1.0,
        terminate_on_empty=True,
        sumo_extra_args=[],
        normalize_state=True,
        return_raw_state=False,
        enable_kpi_tracker=False,
        state_dim=12,
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
        reward_time_normalize=False,
    )
    base_cfg.update(overrides)
    cfg = SumoEnvConfig(**base_cfg)
    lanes = SumoLaneGroups(lanes_ns_ctrl=["N2C_0"], lanes_ew_ctrl=["E2C_0"])
    phases = SumoPhaseProgram(ns_green=0, ew_green=1, ns_yellow=None, ew_yellow=None, all_red=None)
    normalizer = StateNormalizer(mean=[0.0] * 12, std=[1.0] * 12, expected_dim=12)
    return SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)


def test_action_mapping_order():
    env = _make_env()
    actions = env._action_defs
    assert len(actions) == 15

    expected_cycles = [60] * 5 + [90] * 5 + [120] * 5
    for idx, action in enumerate(actions):
        assert int(action.cycle_sec) == expected_cycles[idx]
        expected_split = DEFAULT_ACTION_SPLITS[idx % 5]
        assert pytest.approx(action.rho_ns, rel=1e-6) == expected_split[0]
        assert pytest.approx(action.rho_ew, rel=1e-6) == expected_split[1]


def test_action_table_reorders_to_cycle_split_order():
    shuffled = []
    for cycle in [90, 60, 120]:
        for rho_ns, rho_ew in reversed(DEFAULT_ACTION_SPLITS):
            shuffled.append({"cycle_sec": cycle, "rho_ns": rho_ns, "rho_ew": rho_ew})
    env = _make_env(action_table=shuffled, action_splits=DEFAULT_ACTION_SPLITS)
    actions = env._action_defs
    expected_cycles = [60] * 5 + [90] * 5 + [120] * 5
    assert len(actions) == 15
    for idx, action in enumerate(actions):
        assert int(action.cycle_sec) == expected_cycles[idx]
        expected_split = DEFAULT_ACTION_SPLITS[idx % 5]
        assert pytest.approx(action.rho_ns, rel=1e-6) == expected_split[0]
        assert pytest.approx(action.rho_ew, rel=1e-6) == expected_split[1]


def test_invalid_cycle_count_rejected():
    with pytest.raises(ValueError):
        _make_env(cycle_options_sec=[60, 90])


def test_invalid_split_count_rejected():
    with pytest.raises(ValueError):
        _make_env(action_splits=[(0.5, 0.5), (0.6, 0.4)])


def test_min_green_enforced():
    with pytest.raises(ValueError):
        _make_env(g_min_sec=40)
