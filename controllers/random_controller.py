"""
Random Controller for baseline comparison.

Simply selects a random action from the action space.
This is the weakest baseline - any trained RL should beat it.
"""
from __future__ import annotations

import random
from typing import Any, Sequence


class RandomController:
    """Random action selection controller."""
    
    def __init__(self, action_space: Sequence[Any], seed: int = 42):
        self.action_space_size = len(action_space)
        self._rng = random.Random(seed)
    
    def reset(self, seed: int = None) -> None:
        """Reset RNG with optional new seed."""
        if seed is not None:
            self._rng = random.Random(seed)
    
    def act(self, state: Any = None, *args, **kwargs) -> int:
        """Return random action. Ignores state."""
        return self._rng.randint(0, self.action_space_size - 1)
