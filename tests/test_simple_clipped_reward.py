"""Test simplified reward function with clipping."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.mdp_metrics import compute_simple_clipped_reward


def test_reward_zero_wait():
    """Zero waiting -> reward = 0."""
    r = compute_simple_clipped_reward(wait_total=0.0, n_present=10)
    assert r == 0.0, f"Expected 0.0, got {r}"


def test_reward_clipping_min():
    """Very high waiting -> clipped to -5."""
    r = compute_simple_clipped_reward(wait_total=100000.0, n_present=10, t_ref=60.0)
    assert r == -5.0, f"Expected -5.0, got {r}"


def test_reward_clipping_max():
    """Zero or negative waiting (edge case) -> clipped to 0."""
    r = compute_simple_clipped_reward(wait_total=0.0, n_present=100)
    assert r == 0.0, f"Expected 0.0, got {r}"


def test_reward_formula():
    """Test formula: r = -W/N/t_ref = -120/10/60 = -0.2."""
    r = compute_simple_clipped_reward(wait_total=120.0, n_present=10, t_ref=60.0)
    assert math.isclose(r, -0.2), f"Expected -0.2, got {r}"


def test_reward_in_range():
    """Moderate waiting -> reward in valid range."""
    r = compute_simple_clipped_reward(wait_total=600.0, n_present=20, t_ref=60.0)
    assert -5.0 <= r <= 0.0, f"Reward {r} out of range [-5, 0]"


def test_reward_demand_invariance():
    """Same avg wait should give same reward regardless of vehicle count."""
    # 100 veh-sec / 10 vehicles = 10 avg wait -> -10/60 = -0.167
    r1 = compute_simple_clipped_reward(wait_total=100.0, n_present=10, t_ref=60.0)
    # 200 veh-sec / 20 vehicles = 10 avg wait -> -10/60 = -0.167
    r2 = compute_simple_clipped_reward(wait_total=200.0, n_present=20, t_ref=60.0)
    assert math.isclose(r1, r2), f"Demand invariance failed: {r1} != {r2}"


if __name__ == "__main__":
    test_reward_zero_wait()
    test_reward_clipping_min()
    test_reward_clipping_max()
    test_reward_formula()
    test_reward_in_range()
    test_reward_demand_invariance()
    print("All simple clipped reward tests passed!")
