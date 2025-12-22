from __future__ import annotations

import numpy as np

from env.normalization import StateNormalizer
from rl.agent import DQNAgent


def rl_controller_step(state_raw: np.ndarray, agent: DQNAgent, normalizer: StateNormalizer) -> int:
    state_array = np.asarray(state_raw, dtype=np.float32).reshape(-1)
    if state_array.shape != (4,):
        raise ValueError(f"state_raw must have shape (4,), got {state_array.shape}")

    state_norm = normalizer.normalize(state_array)
    action_id = agent.select_action(state=state_norm, epsilon=0.0)
    return int(action_id)
