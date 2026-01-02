from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from rl.dueling_dqn import DuelingDQN
from rl.replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    hidden_dims: Sequence[int]
    gamma: float
    learning_rate: float
    batch_size: int
    replay_buffer_size: int
    target_update_freq: int
    seed: int


class DQNAgent:
    def __init__(self, config: AgentConfig, device: torch.device):
        self._config = config
        self._device = device

        self._random_state = np.random.default_rng(int(config.seed))

        self.online_net = DuelingDQN(
            state_dim=int(config.state_dim),
            action_dim=int(config.action_dim),
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        self.target_net = DuelingDQN(
            state_dim=int(config.state_dim),
            action_dim=int(config.action_dim),
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(config.learning_rate))
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            capacity=int(config.replay_buffer_size),
            seed=int(config.seed),
            state_dim=int(config.state_dim),
        )

        self.update_step_count = 0

    @property
    def gamma(self) -> float:
        return float(self._config.gamma)

    @property
    def action_dim(self) -> int:
        return int(self._config.action_dim)

    def select_action(self, state: np.ndarray, epsilon: float, allowed_action_ids: Optional[Sequence[int]] = None) -> int:
        epsilon_value = float(epsilon)

        allowed: Optional[np.ndarray] = None
        if allowed_action_ids is not None:
            allowed = np.asarray(list(allowed_action_ids), dtype=np.int64).reshape(-1)
            allowed = allowed[(allowed >= 0) & (allowed < int(self._config.action_dim))]
            if allowed.size <= 0:
                allowed = None

        if self._random_state.random() < epsilon_value:
            if allowed is not None:
                return int(self._random_state.choice(allowed))
            return int(self._random_state.integers(0, int(self._config.action_dim)))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device).view(1, -1)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            if allowed is not None:
                mask = torch.full_like(q_values, fill_value=-1e9)
                idx = torch.as_tensor(allowed, dtype=torch.long, device=self._device)
                mask[0, idx] = q_values[0, idx]
                action_id = int(torch.argmax(mask, dim=1).item())
            else:
                action_id = int(torch.argmax(q_values, dim=1).item())

        return action_id

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < int(self._config.batch_size):
            return None

        batch = self.replay_buffer.sample(batch_size=int(self._config.batch_size), device=self._device)

        with torch.no_grad():
            next_q_online = self.online_net(batch.next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(batch.next_states).gather(1, next_actions)
            target_q = batch.rewards + float(self._config.gamma) * next_q_target * (1.0 - batch.dones)

        current_q = self.online_net(batch.states).gather(1, batch.actions)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step_count += 1
        if int(self.update_step_count) % int(self._config.target_update_freq) == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save_model(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "state_dim": int(self._config.state_dim),
            "action_dim": int(self._config.action_dim),
            "hidden_dims": [int(x) for x in self._config.hidden_dims],
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
        }
        torch.save(payload, str(path))

    def load_model(self, path: str) -> None:
        payload = torch.load(str(path), map_location=self._device)

        online_state_dict = payload.get("online_state_dict")
        target_state_dict = payload.get("target_state_dict")

        if online_state_dict is None:
            raise ValueError("Missing online_state_dict in model file")

        self.online_net.load_state_dict(online_state_dict)

        if target_state_dict is not None:
            self.target_net.load_state_dict(target_state_dict)
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.target_net.eval()

    def to_eval_mode(self) -> None:
        self.online_net.eval()
        self.target_net.eval()

    def to_train_mode(self) -> None:
        self.online_net.train()
        self.target_net.eval()
