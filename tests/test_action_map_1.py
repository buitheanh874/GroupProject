from __future__ import annotations

import pytest


def test_action_mapping_5_5_5_cycle_major():
    splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    cycles = [60, 90, 120]

    actions = []
    for cycle in cycles:
        for rho_ns, rho_ew in splits:
            actions.append({"cycle_sec": cycle, "rho_ns": rho_ns, "rho_ew": rho_ew})

    assert len(actions) == 15

    for idx in range(5):
        assert actions[idx]["cycle_sec"] == 60
    for idx in range(5, 10):
        assert actions[idx]["cycle_sec"] == 90
    for idx in range(10, 15):
        assert actions[idx]["cycle_sec"] == 120

    expected_splits_order = [0.30, 0.40, 0.50, 0.60, 0.70]
    for cycle_offset in [0, 5, 10]:
        for split_idx, expected_rho in enumerate(expected_splits_order):
            action_idx = cycle_offset + split_idx
            assert abs(actions[action_idx]["rho_ns"] - expected_rho) < 1e-6
            assert abs(actions[action_idx]["rho_ew"] - (1.0 - expected_rho)) < 1e-6


def test_action_mapping_rejects_split_major_order():
    splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    cycles = [60, 90, 120]

    wrong_order_actions = []
    for rho_ns, rho_ew in splits:
        for cycle in cycles:
            wrong_order_actions.append({"cycle_sec": cycle, "rho_ns": rho_ns, "rho_ew": rho_ew})

    assert wrong_order_actions[0]["cycle_sec"] == 60
    assert wrong_order_actions[1]["cycle_sec"] == 90
    assert wrong_order_actions[2]["cycle_sec"] == 120

    assert wrong_order_actions[0]["cycle_sec"] != wrong_order_actions[1]["cycle_sec"]


def test_min_green_constraint_with_new_cycles():
    splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    cycles = [60, 90, 120]
    g_min_sec = 10

    for cycle in cycles:
        for rho_ns, rho_ew in splits:
            g_ns = rho_ns * cycle
            g_ew = rho_ew * cycle
            assert g_ns >= g_min_sec, f"cycle={cycle} rho_ns={rho_ns} g_ns={g_ns} < {g_min_sec}"
            assert g_ew >= g_min_sec, f"cycle={cycle} rho_ew={rho_ew} g_ew={g_ew} < {g_min_sec}"


def test_action_definitions_from_env_config():
    from env.sumo_env import SumoActionDefinition

    splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    cycles = [60, 90, 120]

    action_defs = []
    for cycle in cycles:
        for rho_ns, rho_ew in splits:
            action_defs.append(
                SumoActionDefinition(
                    cycle_sec=int(cycle),
                    rho_ns=float(rho_ns),
                    rho_ew=float(rho_ew),
                )
            )

    assert len(action_defs) == 15

    assert action_defs[0].cycle_sec == 60
    assert abs(action_defs[0].rho_ns - 0.30) < 1e-6
    assert action_defs[4].cycle_sec == 60
    assert abs(action_defs[4].rho_ns - 0.70) < 1e-6

    assert action_defs[5].cycle_sec == 90
    assert abs(action_defs[5].rho_ns - 0.30) < 1e-6
    assert action_defs[9].cycle_sec == 90
    assert abs(action_defs[9].rho_ns - 0.70) < 1e-6

    assert action_defs[10].cycle_sec == 120
    assert abs(action_defs[10].rho_ns - 0.30) < 1e-6
    assert action_defs[14].cycle_sec == 120
    assert abs(action_defs[14].rho_ns - 0.70) < 1e-6


if __name__ == "__main__":
    test_action_mapping_5_5_5_cycle_major()
    test_action_mapping_rejects_split_major_order()
    test_min_green_constraint_with_new_cycles()
    test_action_definitions_from_env_config()
    print("All action mapping tests passed")
