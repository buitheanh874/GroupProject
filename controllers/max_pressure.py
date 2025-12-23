from __future__ import annotations

from typing import Optional, Sequence


class MaxPressureSplitController:
    """
    Pressure-based split heuristic for discrete NS green ratios (not classical phase max-pressure).
    """

    def __init__(self, lanes_ns: Sequence[str], lanes_ew: Sequence[str], splits_ns: Sequence[float], default_action: Optional[int] = None):
        self._lanes_ns = [str(lane) for lane in lanes_ns]
        self._lanes_ew = [str(lane) for lane in lanes_ew]

        if len(self._lanes_ns) <= 0 or len(self._lanes_ew) <= 0:
            raise ValueError("lanes_ns and lanes_ew must not be empty")

        splits = [float(rho) for rho in splits_ns]
        if len(splits) <= 0:
            raise ValueError("splits_ns must not be empty")

        for rho in splits:
            if rho < 0.0 or rho > 1.0:
                raise ValueError(f"split ratio must be in [0, 1], got {rho}")

        self._splits_ns = splits

        mid_index = len(self._splits_ns) // 2
        default_id = mid_index if default_action is None else int(default_action)
        if default_id < 0 or default_id >= len(self._splits_ns):
            raise ValueError(f"default_action {default_id} is out of range for {len(self._splits_ns)} splits")
        self._default_action = default_id

    def select_action(self, env: object) -> int:
        traci = getattr(env, "_traci", None) or getattr(env, "traci", None)
        if traci is None:
            raise RuntimeError("TraCI handle not found on env (expected _traci or traci).")

        q_ns = self._sum_halting(traci, self._lanes_ns)
        q_ew = self._sum_halting(traci, self._lanes_ew)

        total = q_ns + q_ew
        if total <= 0.0:
            return int(self._default_action)

        target_rho_ns = q_ns / total

        best_action = 0
        best_diff = float("inf")
        for idx, rho_ns in enumerate(self._splits_ns):
            diff = abs(rho_ns - target_rho_ns)
            if diff < best_diff:
                best_diff = diff
                best_action = idx

        return int(best_action)

    def _sum_halting(self, traci: object, lanes: Sequence[str]) -> float:
        total = 0.0
        for lane_id in lanes:
            try:
                total += float(traci.lane.getLastStepHaltingNumber(str(lane_id)))
            except Exception:
                continue
        return float(total)


def _self_test() -> None:
    class _MockLane:
        def __init__(self, counts):
            self._counts = counts

        def getLastStepHaltingNumber(self, lane_id: str) -> float:
            return float(self._counts.get(lane_id, 0.0))

    class _MockTraci:
        def __init__(self, counts):
            self.lane = _MockLane(counts)

    class _MockEnv:
        def __init__(self, counts):
            self._traci = _MockTraci(counts)

    splits = [0.3, 0.5, 0.7]
    controller = MaxPressureSplitController(lanes_ns=["n1"], lanes_ew=["e1"], splits_ns=splits)

    # empty road returns middle action
    assert controller.select_action(_MockEnv({"n1": 0, "e1": 0})) == 1
    # q_ns >> q_ew returns action with largest rho_ns
    assert controller.select_action(_MockEnv({"n1": 10, "e1": 0})) == 2


if __name__ == "__main__":
    _self_test()
