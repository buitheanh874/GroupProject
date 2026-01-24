from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()

        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must have length 2")

        hidden_1 = int(hidden_dims[0])
        hidden_2 = int(hidden_dims[1])

        self.feature_net = nn.Sequential(
            nn.Linear(int(state_dim), hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(hidden_2, 1)
        self.advantage_head = nn.Linear(hidden_2, int(action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        return q_values
