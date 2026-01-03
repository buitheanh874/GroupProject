from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.mdp_metrics import compute_anti_flicker_penalty, compute_normalized_reward
from rl.agent import AgentConfig, DQNAgent


def test_compute_normalized_reward_time_scaling():
    reward = compute_normalized_reward(
        wait_total=120.0,
        fairness_penalty=30.0,
        spill_penalty=0.0,
        anti_flicker_penalty=0.0,
        t_step=150.0,
        decision_cycle_sec=200.0,
    )
    assert math.isclose(reward, -(120.0 + 30.0) / 150.0)


def test_compute_normalized_reward_with_penalties():
    reward = compute_normalized_reward(
        wait_total=50.0,
        fairness_penalty=5.0,
        spill_penalty=2.0,
        anti_flicker_penalty=3.0,
        t_step=60.0,
        decision_cycle_sec=60.0,
    )
    assert math.isclose(reward, -60.0 / 60.0)


def test_anti_flicker_penalty_toggle():
    prev_cycle = 30
    kappa = 2.5
    assert math.isclose(compute_anti_flicker_penalty(prev_cycle_sec=prev_cycle, cycle_sec=30, enabled=True, kappa=kappa), 0.0)
    assert math.isclose(compute_anti_flicker_penalty(prev_cycle_sec=prev_cycle, cycle_sec=60, enabled=True, kappa=kappa), 2.5)
    assert math.isclose(compute_anti_flicker_penalty(prev_cycle_sec=prev_cycle, cycle_sec=90, enabled=False, kappa=kappa), 0.0)


def test_time_aware_gamma_computation():
    cfg = AgentConfig(
        state_dim=4,
        action_dim=2,
        hidden_dims=[8, 8],
        gamma=0.9,
        use_time_aware_gamma=True,
        gamma_0=0.9,
        t_ref=10.0,
        learning_rate=1e-3,
        batch_size=1,
        replay_buffer_size=10,
        target_update_freq=1,
        seed=0,
    )
    agent = DQNAgent(cfg, device=torch.device("cpu"))

    gamma_eff = agent.compute_gamma(t_step=5.0)
    assert math.isclose(gamma_eff, 0.9 ** 0.5)

    gamma_default = agent.compute_gamma(t_step=None)
    assert math.isclose(gamma_default, agent.gamma)


if __name__ == "__main__":
    test_compute_normalized_reward_time_scaling()
    test_compute_normalized_reward_with_penalties()
    test_anti_flicker_penalty_toggle()
    test_time_aware_gamma_computation()
    print("test_reward_and_penalties passed")
