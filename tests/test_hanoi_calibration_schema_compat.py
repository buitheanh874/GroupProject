from __future__ import annotations

import pytest

from scripts.hanoi_calibration import normalize_calibration_schema, validate_calibration


def test_spec_root_keys_normalize_and_validate():
    calib = {
        "entry_edges": ["A"],
        "exit_edges": ["X"],
        "pcu_weights": {"motorcycle": 0.25},
        "vehicle_mix_mean": {"motorcycle": 1.0},
        "vehicle_mix_kappa": 10,
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5},
        "turning_overrides": {"A": [0.3, 0.5, 0.2]},
    }
    normalized = normalize_calibration_schema({"scenario": calib})
    assert normalized["vehicle_mix"]["mean"]["motorcycle"] == 1.0
    assert normalized["vehicle_mix"]["kappa"] == 10
    assert normalized["turning"]["turning_overrides"]["A"] == [0.3, 0.5, 0.2]
    validated = validate_calibration({"scenario": calib})
    assert validated["entry_edges"] == ["A"]


def test_nested_legacy_keys_still_valid():
    calib = {
        "entry_edges": ["A"],
        "exit_edges": ["X"],
        "pcu_weights": {"motorcycle": 0.25},
        "vehicle_mix": {"mean": {"motorcycle": 1.0}, "kappa": 2.0},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5, "turning_overrides": {"A": [0.2, 0.6, 0.2]}},
    }
    validated = validate_calibration({"scenario": calib})
    assert validated["vehicle_mix"]["kappa"] == 2.0
    assert "turning_overrides" in validated["turning"]


def test_root_overrides_take_priority():
    calib = {
        "entry_edges": ["A"],
        "exit_edges": ["X"],
        "pcu_weights": {"motorcycle": 0.25},
        "vehicle_mix_mean": {"motorcycle": 0.7},
        "vehicle_mix_kappa": 9,
        "vehicle_mix": {"mean": {"motorcycle": 0.1}, "kappa": 1},
        "turning_overrides": {"A": [0.1, 0.8, 0.1]},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5, "turning_overrides": {"A": [0.2, 0.6, 0.2]}},
    }
    normalized = normalize_calibration_schema({"scenario": calib})
    assert normalized["vehicle_mix"]["mean"]["motorcycle"] == 0.7
    assert normalized["vehicle_mix"]["kappa"] == 9
    assert normalized["turning"]["turning_overrides"]["A"] == [0.1, 0.8, 0.1]
    validate_calibration({"scenario": calib})
