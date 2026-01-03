from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validation import validate_action_table, validate_scalar_params


def _base_config() -> dict:
    return {
        "allowed_cycles_sec": [30, 60],
        "rho_min": 0.1,
        "g_min_sec": 5,
        "state_dim": 4,
    }


def test_invalid_action_cycle_rejected():
    cfg = _base_config()
    try:
        validate_action_table(
            action_table_raw=[{"cycle_sec": 45, "rho_ns": 0.5, "rho_ew": 0.5}],
            action_splits=[(0.5, 0.5)],
            state_dim=cfg["state_dim"],
            allowed_cycles=cfg["allowed_cycles_sec"],
            rho_min=cfg["rho_min"],
            g_min_sec=cfg["g_min_sec"],
        )
        assert False, "Expected ValueError for disallowed cycle_sec"
    except ValueError:
        pass


def test_g_min_enforced():
    cfg = _base_config()
    cfg["allowed_cycles_sec"] = [20]
    cfg["g_min_sec"] = 12
    try:
        validate_action_table(
            action_table_raw=[{"cycle_sec": 20, "rho_ns": 0.5, "rho_ew": 0.5}],
            action_splits=[(0.5, 0.5)],
            state_dim=cfg["state_dim"],
            allowed_cycles=cfg["allowed_cycles_sec"],
            rho_min=cfg["rho_min"],
            g_min_sec=cfg["g_min_sec"],
        )
        assert False, "Expected ValueError for violating g_min_sec"
    except ValueError:
        pass


def test_rho_min_enforced():
    cfg = _base_config()
    cfg["rho_min"] = 0.2
    try:
        validate_action_table(
            action_table_raw=[{"cycle_sec": 30, "rho_ns": 0.1, "rho_ew": 0.9}],
            action_splits=[(0.5, 0.5)],
            state_dim=cfg["state_dim"],
            allowed_cycles=cfg["allowed_cycles_sec"],
            rho_min=cfg["rho_min"],
            g_min_sec=cfg["g_min_sec"],
        )
        assert False, "Expected ValueError for violating rho_min"
    except ValueError:
        pass


def test_spillback_threshold_validation():
    try:
        validate_scalar_params(
            yellow_sec=0,
            all_red_sec=0,
            rho_min=0.1,
            g_min_sec=5,
            lambda_fairness=0.12,
            fairness_metric="max",
            queue_count_mode="distinct_cycle",
            halt_speed_threshold=0.1,
            use_enhanced_reward=False,
            reward_exponent=1.0,
            enable_anti_flicker=False,
            kappa=0.0,
            enable_spillback_penalty=True,
            beta=1.0,
            occ_threshold=1.5,
            allowed_cycles=[30],
        )
        assert False, "Expected ValueError for invalid occ_threshold"
    except ValueError:
        pass


def test_anti_flicker_requires_nonnegative_kappa():
    try:
        validate_scalar_params(
            yellow_sec=0,
            all_red_sec=0,
            rho_min=0.1,
            g_min_sec=5,
            lambda_fairness=0.12,
            fairness_metric="max",
            queue_count_mode="distinct_cycle",
            halt_speed_threshold=0.1,
            use_enhanced_reward=False,
            reward_exponent=1.0,
            enable_anti_flicker=True,
            kappa=-1.0,
            enable_spillback_penalty=False,
            beta=0.0,
            occ_threshold=0.0,
            allowed_cycles=[30],
        )
        assert False, "Expected ValueError for negative kappa"
    except ValueError:
        pass


if __name__ == "__main__":
    test_invalid_action_cycle_rejected()
    test_g_min_enforced()
    test_rho_min_enforced()
    test_spillback_threshold_validation()
    test_anti_flicker_requires_nonnegative_kappa()
    print("test_action_table_validation_strict passed")
