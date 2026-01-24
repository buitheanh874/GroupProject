from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
from env.normalization import StateNormalizer


def _make_env(include_transition: Optional[bool]) -> SUMOEnv:
    config_kwargs = dict(
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
        yellow_sec=2,
        all_red_sec=1,
        max_cycles=1,
        max_sim_seconds=10,
        seed=0,
        rho_min=0.1,
        g_min_sec=5,
        cycle_options_sec=[60, 90, 120],
        action_splits=[
            (0.30, 0.70),
            (0.40, 0.60),
            (0.50, 0.50),
            (0.60, 0.40),
            (0.70, 0.30),
        ],
        action_table=[],
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
        state_dim=4,
        enable_downstream_occupancy=False,
    )

    if include_transition is not None:
        config_kwargs["include_transition_in_waiting"] = include_transition

    config = SumoEnvConfig(**config_kwargs)

    lanes = SumoLaneGroups(
        lanes_ns_ctrl=["N2C_0"],
        lanes_ew_ctrl=["E2C_0"],
        lanes_right_turn_slip_ns=[],
        lanes_right_turn_slip_ew=[],
    )
    phases = SumoPhaseProgram(ns_green=0, ew_green=1, ns_yellow=2, ew_yellow=3, all_red=4)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)
    env = SUMOEnv(config=config, lanes=lanes, phases=phases, normalizer=normalizer)
    return env


def _non_green_flags(env: SUMOEnv, intervals: List[tuple[int, int, bool]]) -> List[bool]:
    green_phases = {env._phases.ns_green, env._phases.ew_green}
    return [acc for (phase_idx, _, acc) in intervals if phase_idx not in green_phases]


def test_transition_waiting_excluded_when_flag_false():
    env = _make_env(include_transition=False)
    action_def = env._action_defs[0]
    intervals = env._build_intervals_for_tls(
        tls_id=str(env._config.tls_id), action_def=action_def, include_transition=False
    )
    flags = _non_green_flags(env, intervals)
    assert len(flags) > 0
    assert all(flag is False for flag in flags)


def test_transition_waiting_included_when_flag_true():
    env = _make_env(include_transition=True)
    action_def = env._action_defs[0]
    intervals = env._build_intervals_for_tls(
        tls_id=str(env._config.tls_id), action_def=action_def, include_transition=True
    )
    flags = _non_green_flags(env, intervals)
    assert len(flags) > 0
    assert all(flag is True for flag in flags)


def test_transition_waiting_defaults_to_false():
    env = _make_env(include_transition=None)
    assert env._include_transition_in_waiting is False
