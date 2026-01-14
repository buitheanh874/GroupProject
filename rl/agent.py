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
    use_time_aware_gamma: bool = False
    gamma_0: float = 0.98
    t_ref: float = 60.0
    clip_grad_norm: Optional[float] = 10.0 


class DQNAgent:
    def __init__(self, config: AgentConfig, device: torch.device):
        self._config = config
        self._device = device

        self._random_state = np.random.default_rng(int(config.seed))
        self._use_time_aware_gamma = bool(getattr(config, "use_time_aware_gamma", False))
        self._gamma_base = float(getattr(config, "gamma_0", config.gamma))
        self._t_ref = float(getattr(config, "t_ref", 60.0))
        
        if self._use_time_aware_gamma:
            if self._t_ref <= 0.0:
                raise ValueError(f"t_ref must be >0 when use_time_aware_gamma is True, got {self._t_ref}")
            if not (0.0 < self._gamma_base <= 1.0):
                raise ValueError(f"gamma_base must be in (0, 1] when use_time_aware_gamma is True, got {self._gamma_base}")

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
        if self._use_time_aware_gamma:
            return float(self._gamma_base)
        return float(self._config.gamma)

    @property
    def use_time_aware_gamma(self) -> bool:
        return bool(self._use_time_aware_gamma)

    @property
    def t_ref(self) -> float:
        return float(self._t_ref)

    def compute_gamma(self, t_step: Optional[float]) -> float:
        if not self._use_time_aware_gamma or t_step is None:
            return float(self.gamma)
        t_step_value = float(t_step)
        if t_step_value <= 0.0:
            raise ValueError(f"t_step must be >0 when use_time_aware_gamma is enabled, got {t_step_value}")
        
        exponent = t_step_value / float(self._t_ref)
        result = float(self._gamma_base ** exponent)
        
        result = max(1e-9, min(result, 1.0))
        
        return result

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

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, gamma: Optional[float] = None) -> None:
        gamma_value = float(gamma) if gamma is not None else float(self.gamma)
        self.replay_buffer.push(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            gamma=float(gamma_value),
        )

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < int(self._config.batch_size):
            return None

        batch = self.replay_buffer.sample(batch_size=int(self._config.batch_size), device=self._device)

        with torch.no_grad():
            next_q_online = self.online_net(batch.next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(batch.next_states).gather(1, next_actions)
            target_q = batch.rewards + batch.gammas * next_q_target * (1.0 - batch.dones)

        current_q = self.online_net(batch.states).gather(1, batch.actions)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self._config.clip_grad_norm is not None and float(self._config.clip_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=float(self._config.clip_grad_norm))
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

    def save_checkpoint(self, path: str, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """Save full checkpoint for resuming training.
        
        Args:
            path: Path to save checkpoint file
            extra_state: Additional state to save (episode, global_step, phase_idx, etc.)
        """
        payload: Dict[str, Any] = {
            "state_dim": int(self._config.state_dim),
            "action_dim": int(self._config.action_dim),
            "hidden_dims": [int(x) for x in self._config.hidden_dims],
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_step_count": self.update_step_count,
        }
        if extra_state:
            payload.update(extra_state)
        torch.save(payload, str(path))

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint and return extra_state for resuming training.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Extra state dict (episode, global_step, phase_idx, best_reward, etc.)
        """
        payload = torch.load(str(path), map_location=self._device)
        
        self.online_net.load_state_dict(payload["online_state_dict"])
        self.target_net.load_state_dict(payload["target_state_dict"])
        
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        
        self.update_step_count = payload.get("update_step_count", 0)
        self.target_net.eval()
        
        # Return extra state for training loop
        excluded_keys = {
            "state_dim", "action_dim", "hidden_dims",
            "online_state_dict", "target_state_dict",
            "optimizer_state_dict", "update_step_count"
        }
        return {k: v for k, v in payload.items() if k not in excluded_keys}

