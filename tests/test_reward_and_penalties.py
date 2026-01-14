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


def test_compute_normalized_reward_with_variable_cycle():
    reward_short = compute_normalized_reward(
        wait_total=30.0,
        fairness_penalty=0.0,
        spill_penalty=0.0,
        anti_flicker_penalty=0.0,
        t_step=30.0,
        decision_cycle_sec=30.0,
    )
    reward_long = compute_normalized_reward(
        wait_total=30.0,
        fairness_penalty=0.0,
        spill_penalty=0.0,
        anti_flicker_penalty=0.0,
        t_step=90.0,
        decision_cycle_sec=90.0,
    )
    assert reward_short < reward_long
    assert math.isclose(reward_short, -1.0)
    assert math.isclose(reward_long, -(30.0 / 90.0))


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


def test_reward_monotonic_with_weights_and_exponent():
    from env.mdp_metrics import CycleMetricsAggregator

    agg = CycleMetricsAggregator(directions=["N"], queue_mode="distinct_cycle")
    weights = {"a": 2.0, "b": 1.0}
    agg.observe("N", ["a", "b"], step_sec=1.0, accumulate_waiting=True, weight_lookup=weights.get)
    wait_weighted = agg.waiting_total(use_weights=True)
    wait_unweighted = agg.waiting_total(use_weights=False)
    assert wait_weighted > wait_unweighted

    reward_weighted = compute_normalized_reward(wait_total=wait_weighted, t_step=10.0, decision_cycle_sec=10.0)
    reward_unweighted = compute_normalized_reward(wait_total=wait_unweighted, t_step=10.0, decision_cycle_sec=10.0)
    assert reward_weighted < reward_unweighted

    agg_exp = CycleMetricsAggregator(directions=["N"], queue_mode="distinct_cycle")
    agg_exp.observe("N", ["a", "b"], step_sec=1.0, accumulate_waiting=True)
    agg_exp.observe("N", ["a"], step_sec=1.0, accumulate_waiting=True)  # a waits longer
    wait_linear = agg_exp.waiting_total(exponent=1.0)
    wait_quad = agg_exp.waiting_total(exponent=2.0)
    assert wait_quad > wait_linear

    reward_linear = compute_normalized_reward(wait_total=wait_linear, t_step=10.0, decision_cycle_sec=10.0)
    reward_quad = compute_normalized_reward(wait_total=wait_quad, t_step=10.0, decision_cycle_sec=10.0)
    assert reward_quad < reward_linear


if __name__ == "__main__":
    test_compute_normalized_reward_time_scaling()
    test_compute_normalized_reward_with_penalties()
    test_anti_flicker_penalty_toggle()
    test_time_aware_gamma_computation()
    print("test_reward_and_penalties passed")
