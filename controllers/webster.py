"""
Webster-ish Fixed-Time Controller

Implements a fixed-time controller based on Webster's formula:
- Optimal cycle length: C_opt = (1.5L + 5) / (1 - Y)
- Green split proportional to flow ratios

Simplified version for research baseline:
- Uses queue counts as proxy for flow ratios
- Falls back to balanced split (50/50) when queues are balanced
- Selects from discrete action space for compatibility with RL env

Reference: Webster (1958) "Traffic Signal Settings"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass
class WebsterControllerConfig:
    """Configuration for Webster controller."""
    lost_time_per_phase_sec: float = 4.0   # Lost time per phase (yellow + all-red)
    min_cycle_sec: int = 60                 # Minimum cycle length
    max_cycle_sec: int = 120                # Maximum cycle length
    default_cycle_sec: int = 90             # Default when can't compute
    saturation_flow_pcu_hr: float = 1800.0  # Saturation flow rate
    min_split_ratio: float = 0.3            # Minimum split for any phase


class WebsterController:
    """
    Webster formula-based fixed-time controller.
    
    For discrete action space:
    1. Computes optimal cycle using Webster's formula (or uses default)
    2. Computes split based on queue ratio
    3. Selects action from discrete space that best matches
    
    Unlike pure FixedTimeController, this adapts split based on
    observed queue counts, making it "responsive" within fixed-time paradigm.
    """
    
    def __init__(
        self,
        action_space: Sequence[Any],
        config: WebsterControllerConfig = WebsterControllerConfig(),
    ):
        self.config = config
        self._action_space = list(action_space)
        self._cached_action: Optional[int] = None
        
        # Warm-up: track queue observations to compute average
        self._queue_history_ns: list[float] = []
        self._queue_history_ew: list[float] = []
        self._warmup_samples: int = 10
    
    def reset(self) -> None:
        """Reset controller state for new episode."""
        self._cached_action = None
        self._queue_history_ns.clear()
        self._queue_history_ew.clear()
    
    def _extract_queues(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract NS and EW queue counts from state vector."""
        state = np.asarray(state, dtype=np.float32).flatten()
        
        q_ns = 0.0
        q_ew = 0.0
        
        if len(state) >= 1:
            q_ns += float(state[0])
        if len(state) >= 3:
            q_ns += float(state[2])
        if len(state) >= 2:
            q_ew += float(state[1])
        if len(state) >= 4:
            q_ew += float(state[3])
            
        return q_ns, q_ew
    
    def _compute_webster_cycle(self, y_ns: float, y_ew: float) -> int:
        """
        Compute optimal cycle using Webster's formula.
        
        C_opt = (1.5L + 5) / (1 - Y)
        where L = total lost time, Y = sum of critical flow ratios
        """
        L = 2 * self.config.lost_time_per_phase_sec  # 2 phases
        Y = y_ns + y_ew
        
        # Prevent division by zero or negative
        if Y >= 0.95:
            return self.config.max_cycle_sec
        
        if Y <= 0.1:
            return self.config.min_cycle_sec
        
        C_opt = (1.5 * L + 5) / (1.0 - Y)
        C_opt = int(round(C_opt))
        
        # Clamp to valid range
        C_opt = max(self.config.min_cycle_sec, min(self.config.max_cycle_sec, C_opt))
        
        return C_opt
    
    def _find_best_action(
        self,
        target_rho_ns: float,
        target_cycle_sec: int,
    ) -> int:
        """Find action closest to target split and cycle."""
        best_action = 0
        best_error = float("inf")
        
        for idx, action_def in enumerate(self._action_space):
            rho_ns = None
            cycle_sec = None
            
            if hasattr(action_def, 'rho_ns'):
                rho_ns = float(action_def.rho_ns)
                cycle_sec = getattr(action_def, 'cycle_sec', None)
            elif isinstance(action_def, dict):
                rho_ns = action_def.get('rho_ns', action_def.get('ns_ratio'))
                cycle_sec = action_def.get('cycle_sec', action_def.get('cycle'))
            elif isinstance(action_def, (list, tuple)) and len(action_def) >= 1:
                rho_ns = float(action_def[0])
                if len(action_def) >= 3:
                    cycle_sec = action_def[2]
            
            if rho_ns is None:
                continue
            
            # Error = split error + cycle error (weighted)
            error = abs(float(rho_ns) - target_rho_ns)
            
            if cycle_sec is not None:
                cycle_error = abs(int(cycle_sec) - target_cycle_sec) / 60.0  # Normalize
                error += 0.5 * cycle_error  # Weight cycle less than split
            
            if error < best_error:
                best_error = error
                best_action = idx
        
        return best_action
    
    def act(
        self,
        state: np.ndarray,
        tls_id: str = "default",
    ) -> int:
        """
        Select action based on Webster's formula.
        
        During warmup, collects queue samples.
        After warmup, computes optimal timing and caches action.
        
        Args:
            state: State vector with queue counts
            tls_id: Traffic light ID (ignored, same action for all)
            
        Returns:
            Action ID from discrete action space
        """
        q_ns, q_ew = self._extract_queues(state)
        
        # Collect warmup samples
        if len(self._queue_history_ns) < self._warmup_samples:
            self._queue_history_ns.append(q_ns)
            self._queue_history_ew.append(q_ew)
        
        # After warmup, compute and cache optimal action
        if self._cached_action is None and len(self._queue_history_ns) >= self._warmup_samples:
            # Average queues as proxy for flow
            avg_ns = np.mean(self._queue_history_ns)
            avg_ew = np.mean(self._queue_history_ew)
            
            # Convert to flow ratios (simplified)
            # y = q / saturation_flow (approximation)
            total = avg_ns + avg_ew
            if total > 1e-6:
                y_ns = (avg_ns / total) * 0.5  # Scale to reasonable Y
                y_ew = (avg_ew / total) * 0.5
            else:
                y_ns = 0.25
                y_ew = 0.25
            
            # Compute optimal cycle
            opt_cycle = self._compute_webster_cycle(y_ns, y_ew)
            
            # Compute split
            if total > 1e-6:
                rho_ns = avg_ns / total
            else:
                rho_ns = 0.5
            
            # Apply min split constraint
            rho_ns = max(self.config.min_split_ratio, 
                        min(1.0 - self.config.min_split_ratio, rho_ns))
            
            self._cached_action = self._find_best_action(rho_ns, opt_cycle)
        
        # Return cached action or default during warmup
        if self._cached_action is not None:
            return self._cached_action
        else:
            # During warmup, use balanced split with default cycle
            return self._find_best_action(0.5, self.config.default_cycle_sec)
    
    def act_multi(
        self,
        states: dict,
        current_time: float = 0.0,
    ) -> dict:
        """
        Select actions for multiple TLS.
        
        Webster controller uses same action for all TLS
        (centralized timing plan assumption).
        """
        # Use first TLS state to compute action
        first_state = next(iter(states.values()), np.zeros(4))
        action = self.act(first_state)
        
        return {tls_id: action for tls_id in states.keys()}


def _self_test() -> None:
    """Basic self-test."""
    from dataclasses import dataclass
    
    @dataclass
    class MockAction:
        rho_ns: float
        cycle_sec: int
    
    action_space = [
        MockAction(0.3, 60), MockAction(0.4, 60), MockAction(0.5, 60),
        MockAction(0.3, 90), MockAction(0.4, 90), MockAction(0.5, 90),
        MockAction(0.6, 90), MockAction(0.7, 90),
        MockAction(0.3, 120), MockAction(0.5, 120), MockAction(0.7, 120),
    ]
    
    controller = WebsterController(action_space)
    
    # Warmup with NS-heavy traffic
    for _ in range(10):
        state = np.array([10.0, 3.0, 8.0, 4.0])  # NS heavier
        action = controller.act(state)
    
    # After warmup, action should favor NS
    final_action = controller.act(np.zeros(4))
    selected = action_space[final_action]
    assert selected.rho_ns >= 0.5, f"Expected NS-favoring split, got {selected.rho_ns}"
    
    print(f"WebsterController: Selected action {final_action} (rho_ns={selected.rho_ns}, cycle={selected.cycle_sec})")
    print("WebsterController: All tests passed")


if __name__ == "__main__":
    _self_test()
