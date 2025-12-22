from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from env.base_env import BaseEnv


@dataclass
class ToyQueueEnvConfig:
    max_steps: int
    arrival_prob: float
    serve_slow_rate: int
    serve_fast_rate: int
    seed: int


class ToyQueueEnv(BaseEnv):
    def __init__(self, config: ToyQueueEnvConfig):
        self._config = config
        self._random_state = np.random.default_rng(config.seed)
        self._queue_length = 0
        self._step_count = 0

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        self._queue_length = 0
        self._step_count = 0
        return self._get_state()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action_id not in (0, 1):
            raise ValueError(f"Invalid action_id: {action_id}")

        arrivals = int(self._random_state.random() < self._config.arrival_prob)

        if action_id == 0:
            service_capacity = self._config.serve_slow_rate
        else:
            service_capacity = self._config.serve_fast_rate

        served = min(self._queue_length, service_capacity)
        next_queue_length = self._queue_length + arrivals - served

        self._queue_length = int(max(0, next_queue_length))
        self._step_count += 1

        reward = -float(self._queue_length)
        done = self._step_count >= self._config.max_steps

        info: Dict[str, Any] = {
            "queue_length": int(self._queue_length),
            "arrivals": int(arrivals),
            "served": int(served),
            "step_count": int(self._step_count),
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        state = np.array([float(self._queue_length)], dtype=np.float32)
        return state
