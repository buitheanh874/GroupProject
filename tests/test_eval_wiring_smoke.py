from __future__ import annotations

import pytest

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


def test_resolve_action_space_raises_when_empty():
    class _EmptyEnv:
        pass

    with pytest.raises(ValueError, match="Action space is empty"):
        eval_mod._resolve_action_space(_EmptyEnv(), {"env": {"sumo": {}}})


def test_validate_fixed_action_id_out_of_bounds():
    with pytest.raises(ValueError, match="out of bounds"):
        eval_mod._validate_fixed_action_id(3, [0, 1])
