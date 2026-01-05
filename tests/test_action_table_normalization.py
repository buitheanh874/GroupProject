from __future__ import annotations

import pytest

from scripts.config_normalization import normalize_action_table_schema


def test_action_table_rho_ns_passthrough():
    cfg = {"action_table": [{"rho_ns": 0.7, "other": 1}]}
    normalized = normalize_action_table_schema(cfg)
    assert normalized["action_table"][0]["rho_ns"] == 0.7


def test_action_table_ns_ratio_to_rho_ns():
    cfg = {"action_table": [{"ns_ratio": 0.3}]}
    normalized = normalize_action_table_schema(cfg)
    assert normalized["action_table"][0]["rho_ns"] == 0.3


def test_action_table_both_prefers_rho_ns():
    cfg = {"action_table": [{"rho_ns": 0.6, "ns_ratio": 0.4}]}
    normalized = normalize_action_table_schema(cfg)
    assert normalized["action_table"][0]["rho_ns"] == 0.6


def test_action_table_missing_ratio_raises():
    cfg = {"action_table": [{}]}
    with pytest.raises(ValueError):
        normalize_action_table_schema(cfg)
