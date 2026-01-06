from __future__ import annotations

import types

from scripts import eval as eval_mod


class _DummyEnv:
    def __init__(self) -> None:
        self._action_defs = [(0.5, 0.5, 90)]


def test_resolve_fixed_time_config_unbalanced():
    cfg = eval_mod._resolve_fixed_time_config("path/unbalanced.yaml", "")
    assert cfg.target_split == (0.7, 0.3)
    assert cfg.target_cycle_sec == 90


def test_resolve_fixed_time_config_balanced():
    cfg = eval_mod._resolve_fixed_time_config("path/balanced.yaml", "")
    assert cfg.target_split == (0.5, 0.5)
    assert cfg.target_cycle_sec == 90


def test_resolve_action_space_prefers_env_defs():
    env = _DummyEnv()
    action_space = eval_mod._resolve_action_space(env, {"env": {"sumo": {}}})
    assert action_space == [(0.5, 0.5, 90)]
