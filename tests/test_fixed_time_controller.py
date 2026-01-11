from __future__ import annotations

from dataclasses import dataclass

import pytest

from controllers.fixed_time import FixedTimeController, FixedTimeControllerConfig


def test_selects_exact_match_with_cycle():
    action_space = [
        (0.5, 0.5, 90),
        (0.7, 0.3, 90),
        (0.7, 0.3, 60),
    ]
    config = FixedTimeControllerConfig(target_split=(0.7, 0.3), target_cycle_sec=90)
    controller = FixedTimeController(action_space=action_space, config=config)
    assert controller.act() == 1
    assert controller.selected_split == (0.7, 0.3)
    assert controller.selected_cycle_sec == 90


def test_ignores_cycle_penalty_when_action_missing_cycle():
    action_space = [
        (0.6, 0.4),
        (0.7, 0.3),
    ]
    config = FixedTimeControllerConfig(target_split=(0.7, 0.3), target_cycle_sec=90)
    controller = FixedTimeController(action_space=action_space, config=config)
    assert controller.act() == 1
    assert controller.selected_split == (0.7, 0.3)
    assert controller.selected_cycle_sec is None


def test_raises_on_empty_action_space():
    with pytest.raises(ValueError):
        FixedTimeController(action_space=[], config=FixedTimeControllerConfig())


@dataclass
class _DummyAction:
    rho_ns: float
    rho_ew: float
    cycle_sec: int


def test_parses_dict_and_object_actions():
    action_space = [
        {"ns_ratio": 0.4, "rho_ew": 0.6, "cycle_sec": 60},
        _DummyAction(rho_ns=0.7, rho_ew=0.3, cycle_sec=90),
    ]
    config = FixedTimeControllerConfig(target_split=(0.7, 0.3), target_cycle_sec=90)
    controller = FixedTimeController(action_space=action_space, config=config)
    assert controller.act() == 1
    assert controller.selected_split == (0.7, 0.3)
    assert controller.selected_cycle_sec == 90
