from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    gammas: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, state_dim: int):
        self._capacity = int(capacity)
        self._state_dim = int(state_dim)
        self._random_state = np.random.default_rng(int(seed))

        self._states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._actions = np.zeros((self._capacity, 1), dtype=np.int64)
        self._rewards = np.zeros((self._capacity, 1), dtype=np.float32)
        self._gammas = np.ones((self._capacity, 1), dtype=np.float32)
        self._next_states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._dones = np.zeros((self._capacity, 1), dtype=np.float32)

        self._size = 0
        self._pos = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, gamma: float = 1.0) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(1, -1)

        if state_array.shape[1] != self._state_dim or next_state_array.shape[1] != self._state_dim:
            raise ValueError("State shape does not match state_dim")

        index = int(self._pos)

        self._states[index] = state_array[0]
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._gammas[index] = float(gamma)
        self._next_states[index] = next_state_array[0]
        self._dones[index] = float(1.0 if bool(done) else 0.0)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if self._size <= 0:
            raise RuntimeError("ReplayBuffer is empty")

        actual_batch_size = int(min(int(batch_size), int(self._size)))
        indices = self._random_state.choice(self._size, size=actual_batch_size, replace=False)

        states = torch.as_tensor(self._states[indices], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self._actions[indices], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=device)
        gammas = torch.as_tensor(self._gammas[indices], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self._next_states[indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self._dones[indices], dtype=torch.float32, device=device)

        return TransitionBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            gammas=gammas,
            next_states=next_states,
            dones=dones,
        )

    def __len__(self) -> int:
        return int(self._size)
