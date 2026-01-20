from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

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
    episode_uids: Optional[torch.Tensor] = None  # For curriculum tracking


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
        # Episode UID for curriculum tracking: -1 = unknown (backward compat)
        self._episode_uids = np.full((self._capacity, 1), -1, dtype=np.int64)

        self._size = 0
        self._pos = 0

    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool, 
        gamma: float = 1.0,
        episode_uid: int = -1,  # Backward compat: -1 = unknown
    ) -> None:
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
        self._episode_uids[index] = int(episode_uid)

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
        episode_uids = torch.as_tensor(self._episode_uids[indices], dtype=torch.int64, device=device)

        return TransitionBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            gammas=gammas,
            next_states=next_states,
            dones=dones,
            episode_uids=episode_uids,
        )
    
    def get_phase_histogram(self, episode_to_phase: Dict[int, int]) -> Dict[int, int]:
        """
        Get histogram of phases currently in buffer.
        
        Args:
            episode_to_phase: Mapping from episode_uid to phase_idx
            
        Returns:
            Dict mapping phase_idx to count
        """
        histogram: Dict[int, int] = {}
        for i in range(self._size):
            uid = int(self._episode_uids[i, 0])
            if uid < 0:
                phase = -1  # Unknown
            else:
                phase = episode_to_phase.get(uid, -1)
            histogram[phase] = histogram.get(phase, 0) + 1
        return histogram

    def __len__(self) -> int:
        return int(self._size)

