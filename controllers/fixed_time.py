from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class FixedTimeControllerConfig:
    target_split: Tuple[float, float] = (0.5, 0.5)


class FixedTimeController:
    def __init__(self, action_splits: Sequence[Tuple[float, float]], config: FixedTimeControllerConfig = FixedTimeControllerConfig()):
        splits = [(float(x[0]), float(x[1])) for x in action_splits]
        if len(splits) <= 0:
            raise ValueError("action_splits must not be empty")

        target = np.asarray([float(config.target_split[0]), float(config.target_split[1])], dtype=np.float32)
        diffs = []
        for index, split in enumerate(splits):
            value = np.asarray([float(split[0]), float(split[1])], dtype=np.float32)
            diffs.append((float(np.sum(np.abs(value - target))), int(index)))

        diffs.sort(key=lambda x: x[0])
        self._action_id = int(diffs[0][1])

    def act(self) -> int:
        return int(self._action_id)
