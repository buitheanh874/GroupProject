from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.sumo_env import SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram, SUMOEnv
from env.normalization import StateNormalizer


def _build_env_with_pool(seed: int) -> SUMOEnv:
    cfg = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="net.xml",
        route_file="route.rou.xml",
        route_pool=["a.rou.xml", "b.rou.xml", "c.rou.xml"],
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
        seed=seed,
        rho_min=0.1,
        g_min_sec=5,
        lambda_fairness=0.0,
        fairness_metric="max",
        action_splits=[(0.5, 0.5)],
        action_table=[],
        include_transition_in_waiting=True,
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
        state_dim=4,
        enable_downstream_occupancy=False,
    )
    lanes = SumoLaneGroups(lanes_ns_ctrl=["N2C_0"], lanes_ew_ctrl=["E2C_0"])
    phases = SumoPhaseProgram(ns_green=0, ew_green=1, ns_yellow=None, ew_yellow=None, all_red=None)
    normalizer = StateNormalizer(mean=[0.0] * 4, std=[1.0] * 4, expected_dim=4)
    return SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)


def test_route_pool_selection_deterministic_and_varied():
    env_a = _build_env_with_pool(seed=123)
    env_b = _build_env_with_pool(seed=123)

    picks_a = [env_a._select_route_from_pool(episode_index=i) for i in range(1, 6)]
    picks_b = [env_b._select_route_from_pool(episode_index=i) for i in range(1, 6)]

    assert picks_a == picks_b  # deterministic by seed
    assert len(set(picks_a)) > 1  # not always the same route


def test_route_pool_selection_does_not_mutate_rng_state():
    env = _build_env_with_pool(seed=321)
    control = _build_env_with_pool(seed=321)
    pick_first = env._select_route_from_pool(episode_index=1)
    control_rand = control._rng.random()
    rand_after = env._rng.random()
    assert rand_after == control_rand
    pick_repeat = env._select_route_from_pool(episode_index=1)
    assert pick_first == pick_repeat


def test_route_pool_selection_stable_per_episode_index():
    env = _build_env_with_pool(seed=11)
    pick_one = env._select_route_from_pool(episode_index=1)
    env._select_route_from_pool(episode_index=3)
    pick_one_again = env._select_route_from_pool(episode_index=1)
    assert pick_one == pick_one_again
