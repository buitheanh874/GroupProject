from __future__ import annotations

import pytest

from scripts.hanoi_calibration import validate_calibration


def _base_calib():
    return {
        "net_file": "net.xml",
        "map_prefix": "hanoi",
        "entry_edges": ["A", "B"],
        "exit_edges": ["X", "Y"],
        "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
        "demand": {
            "total_pcu_per_hour": {"low": 1000, "high": 2000},
            "entry_dirichlet_alpha": 2.0,
        },
        "vehicle_mix": {"mean": {"motorcycle": 0.8, "passenger": 0.2}, "kappa": 10},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 20},
        "stages": {"enabled": False, "intervals": []},
        "min_total_vehicles": 10,
        "horizon_sec": 3600,
    }


def test_missing_entry_edges_raises():
    calib = _base_calib()
    calib["entry_edges"] = []
    with pytest.raises(ValueError, match="entry_edges"):
        validate_calibration(calib)


def test_missing_exit_edges_raises():
    calib = _base_calib()
    calib["exit_edges"] = []
    with pytest.raises(ValueError, match="exit_edges"):
        validate_calibration(calib)


def test_stage_overlap_detected():
    calib = _base_calib()
    calib["stages"] = {
        "enabled": True,
        "intervals": [
            {"begin": 0, "end": 100, "level": "low"},
            {"begin": 50, "end": 120, "level": "high"},
        ],
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_calibration(calib)


def test_valid_config_passes():
    calib = _base_calib()
    calib["stages"] = {
        "enabled": True,
        "intervals": [
            {"begin": 0, "end": 100, "level": "low"},
            {"begin": 100, "end": 200, "level": "high"},
        ],
    }
    validated = validate_calibration({"scenario": calib})
    assert validated["entry_edges"] == calib["entry_edges"]
