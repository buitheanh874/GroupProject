from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple


def _resolve_scenario(calib: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(calib, dict):
        raise ValueError("Calibration config must be a dictionary")
    if "scenario" in calib and isinstance(calib["scenario"], dict):
        return calib["scenario"]
    return calib


def _validate_turning(turning: Dict[str, Any]) -> None:
    mean_lsr = turning.get("mean_LSR", turning.get("mean_lsr", []))
    if not isinstance(mean_lsr, (list, tuple)) or len(mean_lsr) != 3:
        raise ValueError("turning.mean_LSR must be a list of length 3 (L/S/R)")
    total = sum(float(x) for x in mean_lsr)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"turning.mean_LSR must sum to 1.0 (got {total:.4f})")


def _normalize_interval(raw: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    begin = raw.get("begin_sec", raw.get("begin", raw.get("start", None)))
    end = raw.get("end_sec", raw.get("end", None))
    duration = raw.get("duration_sec")

    if begin is None:
        begin = 0.0
    begin = float(begin)

    if end is None:
        if duration is None:
            raise ValueError("Stage interval must include end/end_sec or duration_sec")
        end = begin + float(duration)
    end = float(end)

    if end <= begin:
        raise ValueError(f"Stage interval end must be > begin (got begin={begin}, end={end})")

    normalized = dict(raw)
    normalized["begin_sec"] = begin
    normalized["end_sec"] = end
    return begin, end, normalized


def _validate_stages(stage_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(stage_cfg, dict):
        return []
    intervals_raw = stage_cfg.get("intervals", [])
    normalized: List[Dict[str, Any]] = []
    if not isinstance(intervals_raw, list):
        raise ValueError("stages.intervals must be a list")

    spans: List[Tuple[float, float]] = []
    for item in intervals_raw:
        begin, end, norm = _normalize_interval(item)
        spans.append((begin, end))
        normalized.append(norm)

    spans_sorted = sorted(spans, key=lambda x: x[0])
    for i in range(1, len(spans_sorted)):
        prev_end = spans_sorted[i - 1][1]
        cur_begin = spans_sorted[i][0]
        if cur_begin < prev_end:
            raise ValueError("Stage intervals must not overlap (begin of interval must be >= end of previous interval)")

    return normalized


def normalize_calibration_schema(calib: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize calibration schema to support both legacy nested keys and spec root keys.

    Rules:
    - vehicle_mix_mean / vehicle_mix_kappa at root -> populate vehicle_mix.mean / vehicle_mix.kappa
      (root overrides nested on conflict).
    - turning_overrides at root -> merged into turning.turning_overrides (root wins on conflict).
    """
    scenario = _resolve_scenario(calib)
    cfg = deepcopy(scenario)

    root_mean = cfg.pop("vehicle_mix_mean", None)
    root_kappa = cfg.pop("vehicle_mix_kappa", None)
    veh_mix = cfg.get("vehicle_mix", {})
    mean = root_mean if root_mean is not None else veh_mix.get("mean")
    kappa = root_kappa if root_kappa is not None else veh_mix.get("kappa")
    if mean is not None or kappa is not None:
        cfg["vehicle_mix"] = {
            "mean": mean if mean is not None else {},
            "kappa": kappa if kappa is not None else veh_mix.get("kappa", 1.0),
        }

    root_overrides = cfg.pop("turning_overrides", None)
    turning_cfg = cfg.get("turning", {})
    merged_overrides = {}
    merged_overrides.update(turning_cfg.get("turning_overrides", {}) or {})
    if isinstance(root_overrides, dict):
        merged_overrides.update(root_overrides)
    if len(merged_overrides) > 0:
        turning_cfg = dict(turning_cfg)
        turning_cfg["turning_overrides"] = merged_overrides
        cfg["turning"] = turning_cfg

    return cfg


def validate_calibration(calib: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Hanoi calibration config according to HANOI_SCENARIO_SPEC_MDP_STYLE_v1.

    Raises ValueError with clear messages on validation failure.
    Returns a normalized copy of the calibration body (scenario block if present).
    """
    cfg = normalize_calibration_schema(calib)

    entry_edges = cfg.get("entry_edges", [])
    exit_edges = cfg.get("exit_edges", [])
    if not entry_edges or len(entry_edges) == 0:
        raise ValueError("entry_edges must not be empty")
    if not exit_edges or len(exit_edges) == 0:
        raise ValueError("exit_edges must not be empty")

    pcu_weights = cfg.get("pcu_weights", {})
    if not isinstance(pcu_weights, dict) or len(pcu_weights) == 0:
        raise ValueError("pcu_weights must be provided")
    bad_weights = {k: v for k, v in pcu_weights.items() if v is None or float(v) <= 0.0}
    if len(bad_weights) > 0:
        raise ValueError(f"pcu_weights must be >0 for all vehicle types (invalid: {bad_weights})")

    turning_cfg = cfg.get("turning", {})
    _validate_turning(turning_cfg)

    stage_cfg = cfg.get("stages", {})
    normalized_intervals = _validate_stages(stage_cfg) if stage_cfg else []
    if stage_cfg:
        cfg["stages"] = dict(stage_cfg)
        cfg["stages"]["intervals"] = normalized_intervals

    horizon = cfg.get("horizon_sec", None)
    if horizon is not None and float(horizon) <= 0.0:
        raise ValueError("horizon_sec must be positive if provided")

    entry_alpha = cfg.get("demand", {}).get("entry_dirichlet_alpha", None)
    if isinstance(entry_alpha, (list, tuple)) and len(entry_alpha) not in (0, len(entry_edges)):
        raise ValueError("entry_dirichlet_alpha length must match entry_edges or be a scalar")

    return cfg
