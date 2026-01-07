from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass
class FixedTimeControllerConfig:
    """Config for selecting a fixed-time action based on split and cycle."""

    target_split: Tuple[float, float] = (0.5, 0.5)
    target_cycle_sec: Optional[int] = None
    cycle_mismatch_penalty: float = 1000.0


class FixedTimeController:
    """Pick the action closest to a target split/cycle for fixed-time baseline."""

    def __init__(self, action_space: Sequence[Any], config: FixedTimeControllerConfig = FixedTimeControllerConfig()):
        if not action_space:
            raise ValueError("action_space must not be empty")

        self.config = config
        self._action_id, self.selected_split, self.selected_cycle_sec = self._find_best_matching_action(
            action_space=action_space
        )

    def _extract_action_params(self, item: Any) -> Tuple[float, float, Optional[int]]:
        """Extract (rho_ns, rho_ew, cycle_sec) from various action encodings."""

        def _coerce_ratio(value: Any, field: str) -> float:
            if value is None:
                raise ValueError(f"Action missing {field}")
            try:
                ratio = float(value)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Action {field} must be numeric, got {value}") from exc
            return ratio

        rho_ns_val: Any = None
        rho_ew_val: Any = None
        cycle_val: Any = None

        if isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise ValueError("Action tuple/list must include at least rho_ns and rho_ew")
            rho_ns_val = item[0]
            rho_ew_val = item[1]
            if len(item) >= 3:
                cycle_val = item[2]
        elif isinstance(item, dict):
            rho_ns_val = item.get("rho_ns", item.get("ns_ratio"))
            rho_ew_val = item.get("rho_ew", item.get("ew_ratio"))
            cycle_val = item.get("cycle_sec", item.get("cycle", item.get("cycle_length_sec")))
        else:
            rho_ns_val = getattr(item, "rho_ns", getattr(item, "ns_ratio", None))
            rho_ew_val = getattr(item, "rho_ew", getattr(item, "ew_ratio", None))
            cycle_val = getattr(item, "cycle_sec", getattr(item, "cycle", None))

        rho_ns = _coerce_ratio(rho_ns_val, "rho_ns")
        rho_ew: Optional[float]
        if rho_ew_val is None:
            complement = 1.0 - rho_ns
            if complement < 0.0 or complement > 1.0:
                raise ValueError("Cannot infer rho_ew from rho_ns; provide both ratios explicitly")
            rho_ew = complement
        else:
            rho_ew = _coerce_ratio(rho_ew_val, "rho_ew")

        cycle_sec = int(cycle_val) if cycle_val is not None else None
        return float(rho_ns), float(rho_ew), cycle_sec

    def _find_best_matching_action(self, action_space: Sequence[Any]) -> Tuple[int, Tuple[float, float], Optional[int]]:
        target_ns, target_ew = float(self.config.target_split[0]), float(self.config.target_split[1])
        target_cycle = self.config.target_cycle_sec
        penalty = float(self.config.cycle_mismatch_penalty)

        best_index = 0
        best_error = float("inf")
        best_split = (0.0, 0.0)
        best_cycle: Optional[int] = None

        for idx, item in enumerate(action_space):
            rho_ns, rho_ew, cycle_sec = self._extract_action_params(item)
            error = abs(rho_ns - target_ns) + abs(rho_ew - target_ew)
            if target_cycle is not None and cycle_sec is not None and int(cycle_sec) != int(target_cycle):
                error += penalty  # heavy penalty when cycles differ and are available
            if error < best_error:
                best_error = error
                best_index = int(idx)
                best_split = (rho_ns, rho_ew)
                best_cycle = cycle_sec

        return best_index, best_split, best_cycle

    def act(self, state: Any = None) -> int:
        """Return the selected fixed action id (state ignored)."""

        return int(self._action_id)
