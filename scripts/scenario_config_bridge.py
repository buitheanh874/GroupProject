from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.hanoi_calibration import validate_calibration


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def apply_calibration_overrides(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    If config specifies scenario_calibration, inject calibration-derived overrides.

    Currently supported:
    - pcu_weights -> env.sumo.vehicle_weights (unless already set and force_calibration_overrides is False)
    """
    calib_path_str = config.get("scenario_calibration")
    if not calib_path_str:
        return config

    calib_path = _resolve_path(str(calib_path_str), project_root)
    if not calib_path.exists():
        raise FileNotFoundError(f"scenario_calibration file not found: {calib_path}")

    force_override = bool(config.get("force_calibration_overrides", False))

    calib_data = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
    if not isinstance(calib_data, dict):
        raise ValueError("scenario_calibration file must contain a mapping/object")
    calib = validate_calibration(calib_data)
    pcu_weights = calib.get("pcu_weights", {})

    sumo_cfg = config.setdefault("env", {}).setdefault("sumo", {})
    if not force_override and sumo_cfg.get("vehicle_weights"):
        return config

    if pcu_weights:
        sumo_cfg["vehicle_weights"] = dict(pcu_weights)
    return config
