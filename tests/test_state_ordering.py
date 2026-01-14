from __future__ import annotations

import numpy as np

from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram
from env.normalization import StateNormalizer


def _make_multi_env_for_state():
    cfg = SumoEnvConfig(
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
        state_dim=12,
        enable_downstream_occupancy=False,
    )
    lanes = {
        "CENTER": SumoLaneGroups(lanes_ns_ctrl=["N2C_0"], lanes_ew_ctrl=["E2C_0"]),
        "N1": SumoLaneGroups(lanes_ns_ctrl=["N12C_0"], lanes_ew_ctrl=["N1E2C_0"]),
    }
    phases = SumoPhaseProgram(ns_green=0, ew_green=1, ns_yellow=None, ew_yellow=None, all_red=None)
    normalizer = StateNormalizer(mean=[0.0] * 12, std=[1.0] * 12, expected_dim=12)
    return SUMOEnv(config=cfg, lanes=lanes, phases=phases, normalizer=normalizer)


def test_state_ordering_multi_tls():
    env = _make_multi_env_for_state()
    last_q_dir = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w_dir = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    state = env._build_state_vector(tls_id="CENTER", last_q_dir=last_q_dir, w_dir=w_dir)

    assert state.shape[0] == 12
    assert state.tolist()[0:4] == [1.0, 2.0, 3.0, 4.0]
    assert state.tolist()[4:8] == [5.0, 6.0, 7.0, 8.0]
    assert state.tolist()[8:12] == [0.0, 0.0, 0.0, 0.0]
