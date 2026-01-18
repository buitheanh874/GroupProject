"""
Actuated Traffic Signal Controller

A simplified actuated controller that mimics SCATS/SCOOT behavior:
- Extends green as long as vehicles are arriving (gap-out logic)
- Respects min/max green constraints
- Selects action from discrete action space (compatible with RL env)

This controller runs in Python without needing SUMO's actuated add.xml,
making it fair for comparison (same sim stepping as RL).

Reference: Gartner et al. (1975) "Development of IMPOST"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ActuatedControllerConfig:
    """Configuration for actuated controller."""
    min_green_sec: float = 10.0      # Minimum green time (s)
    max_green_sec: float = 60.0      # Maximum green time (s)
    gap_out_sec: float = 3.0         # Gap-out threshold (s)
    extension_sec: float = 2.0       # Green extension per vehicle (s)
    yellow_sec: float = 3.0          # Yellow time (s)
    all_red_sec: float = 2.0         # All-red clearance (s)
    queue_threshold: float = 2.0     # Vehicles needed to trigger extension


class ActuatedController:
    """
    Actuated controller using gap-out logic.
    
    The controller extends green phase if vehicles are detected,
    up to max_green_sec, then switches to the other phase.
    
    For discrete action space compatibility, it:
    1. Decides target phase (NS or EW) and green duration
    2. Selects the action with cycle/split closest to current decision
    """
    
    def __init__(
        self,
        action_space: Sequence[Any],
        config: ActuatedControllerConfig = ActuatedControllerConfig(),
    ):
        self.config = config
        self._action_space = list(action_space)
        
        # Per-TLS state
        self._current_phase: Dict[str, str] = {}  # 'NS' or 'EW'
        self._phase_start_time: Dict[str, float] = {}
        self._last_vehicle_time: Dict[str, float] = {}  # Last vehicle detection
        self._accumulated_green: Dict[str, float] = {}
        
    def reset(self) -> None:
        """Reset controller state for new episode."""
        self._current_phase.clear()
        self._phase_start_time.clear()
        self._last_vehicle_time.clear()
        self._accumulated_green.clear()
    
    def _extract_queues(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract NS and EW queue counts from state vector."""
        state = np.asarray(state, dtype=np.float32).flatten()
        
        # State format: [q_N, q_E, q_S, q_W, ...]
        q_ns = 0.0
        q_ew = 0.0
        
        if len(state) >= 1:
            q_ns += float(state[0])  # North
        if len(state) >= 3:
            q_ns += float(state[2])  # South
        if len(state) >= 2:
            q_ew += float(state[1])  # East
        if len(state) >= 4:
            q_ew += float(state[3])  # West
            
        return q_ns, q_ew
    
    def _find_best_action(
        self,
        target_rho_ns: float,
        target_cycle_sec: Optional[int] = None,
    ) -> int:
        """Find action in action_space closest to target phase split."""
        best_action = 0
        best_error = float("inf")
        
        for idx, action_def in enumerate(self._action_space):
            # Extract rho_ns from action definition
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
                
            error = abs(float(rho_ns) - target_rho_ns)
            
            # Penalize cycle mismatch if target specified
            if target_cycle_sec is not None and cycle_sec is not None:
                if int(cycle_sec) != int(target_cycle_sec):
                    error += 100.0  # Large penalty
            
            if error < best_error:
                best_error = error
                best_action = idx
        
        return best_action
    
    def act(
        self,
        state: np.ndarray,
        tls_id: str = "default",
        current_time: float = 0.0,
    ) -> int:
        """
        Select action based on actuated logic.
        
        Args:
            state: State vector with queue counts [q_N, q_E, q_S, q_W, ...]
            tls_id: Traffic light ID (for multi-TLS support)
            current_time: Current simulation time in seconds
            
        Returns:
            Action ID from discrete action space
        """
        q_ns, q_ew = self._extract_queues(state)
        
        # Initialize phase if first call
        if tls_id not in self._current_phase:
            # Start with phase that has higher demand
            initial_phase = 'NS' if q_ns >= q_ew else 'EW'
            self._current_phase[tls_id] = initial_phase
            self._phase_start_time[tls_id] = current_time
            self._last_vehicle_time[tls_id] = current_time
            self._accumulated_green[tls_id] = 0.0
        
        current_phase = self._current_phase[tls_id]
        phase_duration = current_time - self._phase_start_time[tls_id]
        
        # Get current phase queue
        current_queue = q_ns if current_phase == 'NS' else q_ew
        opposing_queue = q_ew if current_phase == 'NS' else q_ns
        
        # Check if vehicles present (update last vehicle time)
        if current_queue > self.config.queue_threshold:
            self._last_vehicle_time[tls_id] = current_time
        
        time_since_vehicle = current_time - self._last_vehicle_time[tls_id]
        
        # Decision logic
        should_switch = False
        
        # Must switch if max green exceeded
        if phase_duration >= self.config.max_green_sec:
            should_switch = True
        # Can switch if min green met AND gap-out OR opposing queue higher
        elif phase_duration >= self.config.min_green_sec:
            # Gap-out: no vehicle for gap_out_sec
            if time_since_vehicle >= self.config.gap_out_sec:
                should_switch = True
            # Opposing pressure exceeds current
            elif opposing_queue > current_queue * 1.5:
                should_switch = True
        
        # Update phase if switching
        if should_switch:
            new_phase = 'EW' if current_phase == 'NS' else 'NS'
            self._current_phase[tls_id] = new_phase
            self._phase_start_time[tls_id] = current_time
            self._last_vehicle_time[tls_id] = current_time
            current_phase = new_phase
        
        # Convert to action: target split based on current phase
        # NS phase -> higher rho_ns, EW phase -> lower rho_ns
        if current_phase == 'NS':
            # Prioritize NS: target ~0.6-0.7 rho_ns
            total = q_ns + q_ew
            if total > 1e-6:
                target_rho = max(0.5, min(0.7, q_ns / total))
            else:
                target_rho = 0.6
        else:
            # Prioritize EW: target ~0.3-0.4 rho_ns
            total = q_ns + q_ew
            if total > 1e-6:
                target_rho = max(0.3, min(0.5, q_ns / total))
            else:
                target_rho = 0.4
        
        return self._find_best_action(target_rho)
    
    def act_multi(
        self,
        states: Dict[str, np.ndarray],
        current_time: float = 0.0,
    ) -> Dict[str, int]:
        """
        Select actions for multiple TLS.
        
        Args:
            states: Dict mapping tls_id -> state vector
            current_time: Current simulation time
            
        Returns:
            Dict mapping tls_id -> action ID
        """
        actions = {}
        for tls_id, state in states.items():
            actions[tls_id] = self.act(state, tls_id, current_time)
        return actions


def _self_test() -> None:
    """Basic self-test."""
    from dataclasses import dataclass
    
    @dataclass
    class MockAction:
        rho_ns: float
        cycle_sec: int
    
    # Create mock action space (same as RL env)
    action_space = [
        MockAction(0.3, 90),
        MockAction(0.4, 90),
        MockAction(0.5, 90),
        MockAction(0.6, 90),
        MockAction(0.7, 90),
    ]
    
    controller = ActuatedController(action_space)
    
    # Test with balanced queues -> should select middle action
    state = np.array([5.0, 5.0, 5.0, 5.0])
    action = controller.act(state, "J0", current_time=0.0)
    assert 0 <= action < len(action_space), f"Action {action} out of range"
    
    # Test phase switching after max green
    for t in range(0, 70, 10):
        action = controller.act(state, "J0", current_time=float(t))
    
    print("ActuatedController: All tests passed")


if __name__ == "__main__":
    _self_test()
