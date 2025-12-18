from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseEnv(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return
