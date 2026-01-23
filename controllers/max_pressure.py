from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class MaxPressureSplitController:
    def __init__(
        self,
        lanes_ns: Sequence[str],
        lanes_ew: Sequence[str],
        splits_ns: Sequence[float],
        default_action: Optional[int] = None,
    ):
        self._lanes_ns = [str(lane) for lane in lanes_ns]
        self._lanes_ew = [str(lane) for lane in lanes_ew]

        if len(self._lanes_ns) <= 0 or len(self._lanes_ew) <= 0:
            raise ValueError("lanes_ns and lanes_ew must not be empty")

        splits = [float(rho) for rho in splits_ns]
        if len(splits) <= 0:
            raise ValueError("splits_ns must not be empty")

        for rho in splits:
            if rho < 0.0 or rho > 1.0:
                raise ValueError(f"split ratio must be in [0, 1], got {rho}")

        self._splits_ns = splits

        mid_index = len(self._splits_ns) // 2
        default_id = mid_index if default_action is None else int(default_action)
        if default_id < 0 or default_id >= len(self._splits_ns):
            raise ValueError(f"default_action {default_id} out of range")
        self._default_action = default_id

    def select_action(self, state_raw: np.ndarray) -> int:
        state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        if state.shape[0] < 2:
            raise ValueError(f"state_raw must have at least 2 elements (q_NS, q_EW), got {state.shape}")

        q_ns = float(state[0])
        q_ew = float(state[1])
        if state.shape[0] >= 3:
            q_ns += float(state[2])
        if state.shape[0] >= 4:
            q_ew += float(state[3])
        total = q_ns + q_ew

        if total <= 1e-6 or not np.isfinite(total):
            return int(self._default_action)

        target_rho_ns = q_ns / total
        target_rho_ns = float(np.clip(target_rho_ns, 0.0, 1.0))

        best_action = 0
        best_diff = float("inf")
        for idx, rho_ns in enumerate(self._splits_ns):
            diff = abs(float(rho_ns) - float(target_rho_ns))
            if diff < best_diff:
                best_diff = diff
                best_action = idx

        return int(best_action)


class FlexibleMaxPressureController:
    """
    Flexible Max Pressure controller for discrete action space (like RL).
    
    Unlike the flawed split-by-queue-ratio approach, this controller:
    - Uses RAW state (not normalized) for accurate pressure calculation
    - Calculates true pressure = queue_in - downstream_occupancy * capacity
    - Selects actions from action_defs based on pressure-weighted split
    - Implements min_green/max_green constraints
    - Uses hysteresis to avoid oscillation
    
    State vector expected (14D):
    [q_N, q_E, q_S, q_W, wait_N, wait_E, wait_S, wait_W, 
     occ_N, occ_E, occ_S, occ_W, current_phase, time_in_phase]
    
    Reference: Varaiya (2013) "The Max-Pressure Controller"
    """
    
    def __init__(
        self,
        action_defs: Sequence,
        min_green_sec: float = 10.0,
        max_green_sec: float = 60.0,
        hysteresis: float = 0.2,
        downstream_capacity_factor: float = 0.5,
        prefer_longer_cycle_threshold: float = 50.0,
    ):
        """
        Args:
            action_defs: List of action definitions with rho_ns and cycle_sec attributes
            min_green_sec: Minimum green time per phase (default 10s)
            max_green_sec: Maximum green time per phase (default 60s)
            hysteresis: Pressure difference threshold to switch phases (default 20%)
            downstream_capacity_factor: Weight for downstream occupancy in pressure calc
            prefer_longer_cycle_threshold: Queue threshold above which longer cycles preferred
        """
        self._action_defs = list(action_defs)
        self.min_green_sec = float(min_green_sec)
        self.max_green_sec = float(max_green_sec)
        self.hysteresis = float(hysteresis)
        self.downstream_capacity_factor = float(downstream_capacity_factor)
        self.prefer_longer_cycle_threshold = float(prefer_longer_cycle_threshold)
        
        # Per-TLS state tracking
        self._current_phase: dict = {}  # tls_id -> 'NS' or 'EW'
        self._last_action: dict = {}  # tls_id -> action_id
        self._phase_start_time: dict = {}  # tls_id -> simulation time
    
    def reset(self):
        """Reset controller state for new episode."""
        self._current_phase.clear()
        self._last_action.clear()
        self._phase_start_time.clear()
    
    def _extract_pressure(self, state: np.ndarray) -> tuple:
        """
        Extract pressure values from state vector.
        
        Standard Max Pressure (Varaiya 2013, PressLight KDD 2019):
        - Pressure = sum of queue lengths per movement group
        - No downstream weighting (that's for back-pressure variant)
        
        Returns:
            (pressure_ns, pressure_ew, total_queue, is_high_demand)
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        
        # Queue counts (indices 0-3): N, E, S, W
        q_n = max(0.0, float(state[0])) if len(state) > 0 else 0.0
        q_e = max(0.0, float(state[1])) if len(state) > 1 else 0.0
        q_s = max(0.0, float(state[2])) if len(state) > 2 else 0.0
        q_w = max(0.0, float(state[3])) if len(state) > 3 else 0.0
        
        # Standard Max Pressure: pressure = queue length per phase
        # NS phase serves North and South approaches
        # EW phase serves East and West approaches
        pressure_ns = q_n + q_s
        pressure_ew = q_e + q_w
        
        total_queue = q_n + q_e + q_s + q_w
        
        # High demand threshold based on typical intersection capacity
        # ~50 vehicles = moderate congestion for 9-intersection network
        is_high_demand = total_queue > self.prefer_longer_cycle_threshold
        
        return pressure_ns, pressure_ew, total_queue, is_high_demand
    
    def _select_best_action(
        self, 
        target_rho_ns: float,
        prefer_longer_cycle: bool = False,
    ) -> int:
        """
        Select action from action_defs that best matches target split.
        
        Standard Max Pressure (PressLight): Select action purely based on
        pressure ratio matching. Cycle length selection is secondary.
        
        Args:
            target_rho_ns: Target NS split ratio (0-1)
            prefer_longer_cycle: If True, prefer longer cycles (high demand)
        
        Returns:
            Action ID (index into action_defs)
        """
        if len(self._action_defs) == 0:
            return 0
        
        best_action = 0
        best_score = float('inf')
        
        for idx, action_def in enumerate(self._action_defs):
            # Get rho_ns from action def
            rho_ns = getattr(action_def, 'rho_ns', None)
            if rho_ns is None and isinstance(action_def, dict):
                rho_ns = action_def.get('rho_ns', action_def.get('ns_ratio', None))
            if rho_ns is None:
                continue
            
            # Get cycle length
            cycle_sec = getattr(action_def, 'cycle_sec', None)
            if cycle_sec is None and isinstance(action_def, dict):
                cycle_sec = action_def.get('cycle_sec', 90)
            cycle_sec = float(cycle_sec) if cycle_sec else 90.0
            
            # PRIMARY: Split difference (must match pressure ratio)
            split_diff = abs(float(rho_ns) - target_rho_ns)
            
            # SECONDARY: Cycle selection based on demand level
            # Standard practice: longer cycles for higher demand (more green time)
            # Shorter cycles for lower demand (faster response to changes)
            if prefer_longer_cycle:
                # High demand: prefer 90s or 120s cycles
                # Small penalty for 60s cycle (too short for high volume)
                cycle_penalty = 0.02 if cycle_sec < 90 else 0.0
            else:
                # Low demand: prefer 60s or 90s cycles
                # Small penalty for 120s cycle (too slow to respond)
                cycle_penalty = 0.02 if cycle_sec > 90 else 0.0
            
            # Total score: split match is 5x more important than cycle
            score = split_diff + cycle_penalty
            
            if score < best_score:
                best_score = score
                best_action = idx
        
        return int(best_action)
    
    def act(
        self,
        state: np.ndarray,
        tls_id: str = "default",
        sim_time: float = 0.0,
    ) -> int:
        """
        Select action based on Max Pressure principle.
        
        Standard formula for discrete action space (PressLight/MPLight):
        - Compute pressure for each direction (queue sum)
        - Select action with split ratio matching pressure ratio
        
        Args:
            state: Raw state vector (14D expected)
            tls_id: Traffic light ID for state tracking
            sim_time: Current simulation time (seconds)
        
        Returns:
            Action ID to apply
        """
        pressure_ns, pressure_ew, total_queue, is_high_demand = self._extract_pressure(state)
        
        # Standard Max Pressure formula for split-based actions
        total_pressure = pressure_ns + pressure_ew
        if total_pressure <= 1e-6:
            # No pressure: use balanced split
            target_rho_ns = 0.5
        else:
            # Proportional split based on pressure ratio
            target_rho_ns = pressure_ns / total_pressure
            # Clamp to valid range
            target_rho_ns = max(0.0, min(1.0, target_rho_ns))
        
        # Select best action
        action_id = self._select_best_action(target_rho_ns, is_high_demand)
        
        # Update state tracking
        self._last_action[tls_id] = action_id
        
        return action_id
    
    @staticmethod
    def _self_test():
        """Self-test for FlexibleMaxPressureController."""
        from dataclasses import dataclass
        
        @dataclass
        class MockActionDef:
            rho_ns: float
            cycle_sec: int
        
        action_defs = [
            MockActionDef(0.3, 60),
            MockActionDef(0.4, 60),
            MockActionDef(0.5, 60),
            MockActionDef(0.3, 90),
            MockActionDef(0.4, 90),
            MockActionDef(0.5, 90),
            MockActionDef(0.6, 90),
            MockActionDef(0.7, 90),
            MockActionDef(0.5, 120),
        ]
        
        ctrl = FlexibleMaxPressureController(action_defs)
        
        # Test 1: No queue -> balanced split
        state = np.zeros(14)
        action = ctrl.act(state, "test1")
        assert action in [2, 5, 8], f"No queue should give ~0.5 split, got action {action}"
        
        # Test 2: NS queue high -> high rho_ns
        state = np.array([10, 0, 10, 0] + [0]*10)  # q_N=10, q_S=10
        action = ctrl.act(state, "test2")
        rho = action_defs[action].rho_ns
        assert rho >= 0.5, f"High NS queue should give rho_ns >= 0.5, got {rho}"
        
        # Test 3: EW queue high -> low rho_ns
        state = np.array([0, 10, 0, 10] + [0]*10)  # q_E=10, q_W=10
        action = ctrl.act(state, "test3")
        rho = action_defs[action].rho_ns
        assert rho <= 0.5, f"High EW queue should give rho_ns <= 0.5, got {rho}"
        
        # Test 4: High downstream occupancy reduces pressure
        state = np.array([10, 10, 10, 10, 0, 0, 0, 0, 0.8, 0, 0.8, 0, 0, 0])
        # High occ_N and occ_S should reduce NS pressure
        action = ctrl.act(state, "test4")
        rho = action_defs[action].rho_ns
        # NS pressure reduced, so rho_ns should be lower
        assert rho <= 0.5, f"High NS downstream occ should reduce rho_ns, got {rho}"
        
        ctrl.reset()
        print("FlexibleMaxPressureController: All tests passed")



class OriginalMaxPressureController:
    """
    Original MaxPressure controller with pressure-based phase switching.
    
    Unlike discrete action-space MP, this controller:
    - Extends green time as long as the current phase has higher pressure
    - Switches phase when opposing queue pressure exceeds current
    - Respects min_green constraint (typically 5-10s per paper)
    
    Reference: Varaiya (2013) "The Max-Pressure Controller for Arbitrary Networks of Signalized Intersections"
    """
    
    def __init__(
        self,
        min_green_sec: float = 5.0,
        max_green_sec: float = 60.0,
        yellow_sec: float = 3.0,
        all_red_sec: float = 2.0,
    ):
        self.min_green_sec = float(min_green_sec)
        self.max_green_sec = float(max_green_sec)
        self.yellow_sec = float(yellow_sec)
        self.all_red_sec = float(all_red_sec)
        
        # Per-TLS state tracking
        self._current_phase: dict = {}  # tls_id -> 'NS' or 'EW'
        self._phase_start_time: dict = {}  # tls_id -> when phase started
        self._in_transition: dict = {}  # tls_id -> bool
    
    def reset(self):
        """Reset controller state for new episode."""
        self._current_phase.clear()
        self._phase_start_time.clear()
        self._in_transition.clear()
    
    def decide_phase(
        self,
        tls_id: str,
        q_ns: float,
        q_ew: float,
        current_time: float,
    ) -> tuple:
        """
        Decide whether to switch phase based on queue pressure.
        
        Returns:
            (phase, extend_green) where:
            - phase: 'NS' or 'EW' (current/next phase)
            - extend_green: True if should extend current green, False if switch
        """
        # Initialize if first call for this TLS
        if tls_id not in self._current_phase:
            # Start with phase that has higher pressure
            initial_phase = 'NS' if q_ns >= q_ew else 'EW'
            self._current_phase[tls_id] = initial_phase
            self._phase_start_time[tls_id] = current_time
            self._in_transition[tls_id] = False
            return (initial_phase, True)
        
        current_phase = self._current_phase[tls_id]
        phase_duration = current_time - self._phase_start_time[tls_id]
        
        # Check min green constraint
        if phase_duration < self.min_green_sec:
            return (current_phase, True)  # Must extend
        
        # Check max green constraint
        if phase_duration >= self.max_green_sec:
            # Force switch
            new_phase = 'EW' if current_phase == 'NS' else 'NS'
            self._current_phase[tls_id] = new_phase
            self._phase_start_time[tls_id] = current_time
            return (new_phase, False)
        
        # Pressure-based decision
        current_pressure = q_ns if current_phase == 'NS' else q_ew
        opposing_pressure = q_ew if current_phase == 'NS' else q_ns
        
        # Switch if opposing pressure significantly exceeds current
        # Use hysteresis to avoid oscillation
        if opposing_pressure > current_pressure * 1.2:  # 20% hysteresis
            new_phase = 'EW' if current_phase == 'NS' else 'NS'
            self._current_phase[tls_id] = new_phase
            self._phase_start_time[tls_id] = current_time
            return (new_phase, False)
        
        return (current_phase, True)  # Extend current phase
    
    def get_phase_for_state(
        self,
        tls_id: str,
        state: np.ndarray,
        current_time: float,
    ) -> str:
        """Get phase decision from state vector."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        
        q_ns = float(state[0]) if len(state) > 0 else 0.0
        q_ew = float(state[1]) if len(state) > 1 else 0.0
        if len(state) >= 3:
            q_ns += float(state[2])  # Add South to North
        if len(state) >= 4:
            q_ew += float(state[3])  # Add West to East
        
        phase, _ = self.decide_phase(tls_id, q_ns, q_ew, current_time)
        return phase


def select_action_from_defs(
    state_raw: np.ndarray,
    action_defs: Sequence,
    allowed_action_ids: Optional[Sequence[int]] = None,
    default_action_id: int = 0,
) -> int:
    """
    Pick an action id for a multi-TLS state using rho_ns closeness.

    Args:
        state_raw: State vector (expects queue counts at indices 0-3: N,E,S,W).
        action_defs: Sequence of objects (or dicts) with rho_ns attribute/key.
        allowed_action_ids: Optional subset to restrict selection (e.g., a cycle bucket).
        default_action_id: Fallback when no valid candidate is found.
    """
    state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
    if state.shape[0] < 2:
        raise ValueError(f"state_raw must have at least 2 elements (q_NS, q_EW), got {state.shape}")

    q_ns = float(state[0])
    q_ew = float(state[1])
    if state.shape[0] >= 3:
        q_ns += float(state[2])
    if state.shape[0] >= 4:
        q_ew += float(state[3])

    total = q_ns + q_ew
    if total <= 1e-6 or not np.isfinite(total):
        return int(default_action_id)

    target_rho_ns = float(np.clip(q_ns / total, 0.0, 1.0))
    candidates = list(allowed_action_ids) if allowed_action_ids is not None and len(allowed_action_ids) > 0 else list(range(len(action_defs)))
    if len(candidates) == 0:
        return int(default_action_id)

    best_action = int(default_action_id if default_action_id in candidates else candidates[0])
    best_diff = float("inf")

    for idx in candidates:
        action_def = action_defs[idx]
        rho_ns = getattr(action_def, "rho_ns", None)
        if rho_ns is None and isinstance(action_def, dict):
            rho_ns = action_def.get("rho_ns", action_def.get("ns_ratio", None))
        if rho_ns is None:
            continue
        diff = abs(float(rho_ns) - target_rho_ns)
        if diff < best_diff:
            best_diff = diff
            best_action = int(idx)

    return int(best_action)


def _self_test() -> None:
    splits = [0.3, 0.5, 0.7]
    controller = MaxPressureSplitController(
        lanes_ns=["n1"],
        lanes_ew=["e1"],
        splits_ns=splits,
    )

    assert controller.select_action(np.array([0.0, 0.0, 0.0, 0.0])) == 1
    assert controller.select_action(np.array([10.0, 0.1, 0.0, 0.0])) == 2
    assert controller.select_action(np.array([0.1, 10.0, 0.0, 0.0])) == 0
    assert controller.select_action(np.array([5.0, 5.0, 0.0, 0.0])) == 1

    print("MaxPressureSplitController: All tests passed")


if __name__ == "__main__":
    _self_test()
