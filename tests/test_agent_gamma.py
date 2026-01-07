import pytest
import torch

from rl.agent import AgentConfig, DQNAgent


def _make_config(**kwargs) -> AgentConfig:
    base = dict(
        state_dim=4,
        action_dim=2,
        hidden_dims=[8, 8],
        gamma=0.98,
        learning_rate=1e-3,
        batch_size=2,
        replay_buffer_size=10,
        target_update_freq=5,
        seed=1,
    )
    base.update(kwargs)
    return AgentConfig(**base)


def test_compute_gamma_constant_when_disabled():
    cfg = _make_config(use_time_aware_gamma=False, gamma=0.97, gamma_0=0.9)
    agent = DQNAgent(cfg, torch.device("cpu"))
    assert pytest.approx(agent.compute_gamma(None)) == 0.97
    assert pytest.approx(agent.compute_gamma(30.0)) == 0.97


def test_compute_gamma_time_aware_monotonic():
    cfg = _make_config(use_time_aware_gamma=True, gamma_0=0.98, t_ref=60.0)
    agent = DQNAgent(cfg, torch.device("cpu"))
    gamma_short = agent.compute_gamma(30.0)
    gamma_long = agent.compute_gamma(60.0)
    assert gamma_short > gamma_long
    assert gamma_short <= 1.0
    with pytest.raises(ValueError, match="t_step"):
        agent.compute_gamma(0.0)


def test_time_aware_gamma_requires_positive_t_ref():
    cfg = _make_config(use_time_aware_gamma=True, t_ref=0.0)
    with pytest.raises(ValueError, match="t_ref"):
        DQNAgent(cfg, torch.device("cpu"))
