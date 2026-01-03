from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class FakeEnv:
    def __init__(self, cycle_to_actions: Dict[int, List[int]], center_tls_id: str):
        self.cycle_to_actions = cycle_to_actions
        self.center_tls_id = center_tls_id


class FakeAgent:
    def select_action(self, state, epsilon: float, allowed_action_ids: Optional[List[int]] = None) -> int:
        if allowed_action_ids is not None and len(allowed_action_ids) > 0:
            return int(min(allowed_action_ids))
        return 0


def build_actions_fixed(tls_ids_sorted: List[str], fixed_action_id: int) -> Dict[str, int]:
    return {tls: int(fixed_action_id) for tls in tls_ids_sorted}


def build_actions_rl(
    fake_env: FakeEnv,
    fake_agent: FakeAgent,
    state: Dict[str, List[float]],
    baseline_fixed_action_id: int,
    center_action_override: Optional[int] = None,
) -> Tuple[Dict[str, int], Optional[List[int]]]:
    tls_ids_sorted = sorted(state.keys())
    center_id = fake_env.center_tls_id if fake_env.center_tls_id in tls_ids_sorted else tls_ids_sorted[0]

    center_action = center_action_override
    if center_action is None:
        center_action = int(fake_agent.select_action(state[state_id], epsilon=0.0)) if (state_id := center_id) else 0

    allowed_ids: Optional[List[int]] = None
    for ids in fake_env.cycle_to_actions.values():
        if center_action in ids:
            allowed_ids = [int(x) for x in ids]
            break
    if allowed_ids is None:
        for ids in fake_env.cycle_to_actions.values():
            if int(baseline_fixed_action_id) in ids:
                allowed_ids = [int(x) for x in ids]
                break

    actions: Dict[str, int] = {}
    for tls in tls_ids_sorted:
        actions[tls] = int(fake_agent.select_action(state[state_id], epsilon=0.0, allowed_action_ids=allowed_ids)) if (state_id := tls) else 0
    return actions, allowed_ids


def test_fixed_controller_all_tls_same_action():
    tls_ids = ["tls0", "tls1", "tls2"]
    actions = build_actions_fixed(tls_ids_sorted=tls_ids, fixed_action_id=7)
    assert all(val == 7 for val in actions.values())


def test_rl_masking_center_bucket():
    env = FakeEnv(
        cycle_to_actions={30: [0, 1, 2, 3, 4], 60: [5, 6, 7, 8, 9], 90: [10, 11, 12, 13, 14]},
        center_tls_id="tls0",
    )
    agent = FakeAgent()
    state = {"tls0": [0.0], "tls1": [1.0], "tls2": [2.0]}
    actions, allowed = build_actions_rl(
        fake_env=env,
        fake_agent=agent,
        state=state,
        baseline_fixed_action_id=2,
        center_action_override=8,
    )
    assert allowed == [5, 6, 7, 8, 9]
    assert all(val == min(allowed) for val in actions.values())


def test_rl_masking_fallback_to_baseline_bucket():
    env = FakeEnv(
        cycle_to_actions={30: [0, 1, 2, 3, 4], 60: [5, 6, 7, 8, 9], 90: [10, 11, 12, 13, 14]},
        center_tls_id="tlsX",
    )
    agent = FakeAgent()
    state = {"tls0": [0.0], "tls1": [1.0], "tls2": [2.0]}
    actions, allowed = build_actions_rl(
        fake_env=env,
        fake_agent=agent,
        state=state,
        baseline_fixed_action_id=2,
        center_action_override=999,
    )
    assert allowed == [0, 1, 2, 3, 4]
    assert all(val == min(allowed) for val in actions.values())


if __name__ == "__main__":
    test_fixed_controller_all_tls_same_action()
    test_rl_masking_center_bucket()
    test_rl_masking_fallback_to_baseline_bucket()
    print("All dict-mode controller tests passed")
