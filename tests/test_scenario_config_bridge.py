from __future__ import annotations

import yaml
from pathlib import Path

from scripts.scenario_config_bridge import apply_calibration_overrides


def _write_calibration(tmp_path: Path) -> Path:
    calib = {
        "scenario": {
            "entry_edges": ["A"],
            "exit_edges": ["X"],
            "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
            "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5},
        }
    }
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(calib), encoding="utf-8")
    return path


def test_calibration_bridge_sets_vehicle_weights(tmp_path: Path):
    calib_path = _write_calibration(tmp_path)
    config = {"scenario_calibration": str(calib_path)}
    updated = apply_calibration_overrides(config, project_root=tmp_path)
    assert updated["env"]["sumo"]["vehicle_weights"] == {"motorcycle": 0.25, "passenger": 1.0}


def test_user_vehicle_weights_preserved_by_default(tmp_path: Path):
    calib_path = _write_calibration(tmp_path)
    config = {
        "scenario_calibration": str(calib_path),
        "env": {"sumo": {"vehicle_weights": {"bus": 3.0}}},
    }
    updated = apply_calibration_overrides(config, project_root=tmp_path)
    assert updated["env"]["sumo"]["vehicle_weights"] == {"bus": 3.0}


def test_force_override_replaces_user_weights(tmp_path: Path):
    calib_path = _write_calibration(tmp_path)
    config = {
        "scenario_calibration": str(calib_path),
        "force_calibration_overrides": True,
        "env": {"sumo": {"vehicle_weights": {"bus": 3.0}}},
    }
    updated = apply_calibration_overrides(config, project_root=tmp_path)
    assert updated["env"]["sumo"]["vehicle_weights"] == {"motorcycle": 0.25, "passenger": 1.0}


def test_relative_path_resolves_under_project_root(tmp_path: Path):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    calib_dir = project_root / "calib"
    calib_dir.mkdir()
    calib_path = calib_dir / "calib.yaml"
    calib = {
        "scenario": {
            "entry_edges": ["A"],
            "exit_edges": ["X"],
            "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
            "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5},
        }
    }
    calib_path.write_text(yaml.safe_dump(calib), encoding="utf-8")

    config = {"scenario_calibration": "calib/calib.yaml"}
    updated = apply_calibration_overrides(config, project_root=project_root)
    assert updated["env"]["sumo"]["vehicle_weights"] == {"motorcycle": 0.25, "passenger": 1.0}
