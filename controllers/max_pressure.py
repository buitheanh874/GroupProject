from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class MaxPressureSplitController:
    def __init__(
        self,
        lanes_ns: Sequence[str],
        lanes_ew: Sequence[str],
        splits_ns: Sequence[float],
        default_action: Optional[int] = None,
    ):
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
            raise ValueError(f"default_action {default_id} out of range")
        self._default_action = default_id

    def select_action(self, state_raw: np.ndarray) -> int:
        state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        if state.shape[0] < 2:
            raise ValueError(f"state_raw must have at least 2 elements (q_NS, q_EW), got {state.shape}")

        q_ns = float(state[0])
        q_ew = float(state[1])
        total = q_ns + q_ew

        if total <= 1e-6 or not np.isfinite(total):
            return int(self._default_action)

        target_rho_ns = q_ns / total
        target_rho_ns = float(np.clip(target_rho_ns, 0.0, 1.0))

        best_action = 0
        best_diff = float("inf")
        for idx, rho_ns in enumerate(self._splits_ns):
            diff = abs(float(rho_ns) - float(target_rho_ns))
            if diff < best_diff:
                best_diff = diff
                best_action = idx

        return int(best_action)


def _self_test() -> None:
    splits = [0.3, 0.5, 0.7]
    controller = MaxPressureSplitController(
        lanes_ns=["n1"],
        lanes_ew=["e1"],
        splits_ns=splits,
    )

    assert controller.select_action(np.array([0.0, 0.0, 0.0, 0.0])) == 1
    assert controller.select_action(np.array([10.0, 0.1, 0.0, 0.0])) == 2
    assert controller.select_action(np.array([0.1, 10.0, 0.0, 0.0])) == 0
    assert controller.select_action(np.array([5.0, 5.0, 0.0, 0.0])) == 1

    print("MaxPressureSplitController: All tests passed")


if __name__ == "__main__":
    _self_test()