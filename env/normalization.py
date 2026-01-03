from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class StateNormalizer:
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        eps: float = 1e-6,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
        expected_dim: Optional[int] = None,
    ):
        mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_array = np.asarray(std, dtype=np.float32).reshape(-1)

        if mean_array.ndim != 1 or std_array.ndim != 1:
            raise ValueError("Normalization stats must be 1D arrays")

        dim = int(expected_dim) if expected_dim is not None else int(mean_array.shape[0])

        if mean_array.shape[0] != dim or std_array.shape[0] != dim:
            raise ValueError(f"Normalization stats must have shape ({dim},), got mean={mean_array.shape}, std={std_array.shape}")

        self._mean = mean_array
        self._std = std_array
        self._dim = dim
        self._eps = float(eps)
        self._clip_min = float(clip_min)
        self._clip_max = float(clip_max)

    def normalize(self, state_raw: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state_raw, dtype=np.float32).reshape(-1)

        if state_array.shape[0] != self._dim:
            raise ValueError(f"state_raw must have shape ({self._dim},), got {state_array.shape}")

        normalized = (state_array - self._mean) / (self._std + self._eps)
        clipped = np.clip(normalized, self._clip_min, self._clip_max)
        return clipped.astype(np.float32)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        return self._std.copy()

    @property
    def dim(self) -> int:
        return int(self._dim)
