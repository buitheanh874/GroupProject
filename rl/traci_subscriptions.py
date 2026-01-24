"""
TraCI subscription helpers for performance optimization.

This module provides:
- TraCISubscriptionManager: Batch subscriptions for lanes/vehicles
- Equivalence test utilities for verifying subscription == explicit calls

Usage:
    manager = TraCISubscriptionManager(traci, lanes, scalar_only=True)
    manager.subscribe_all()  # Call after traci.start()
    
    # In step loop:
    traci.simulationStep()
    results = manager.get_results()  # Returns dict[lane_id -> dict[var -> value]]

Safety: Subscription results are expected to match explicit calls per TraCI docs.
This must be verified by equivalence test, not assumed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np


class TraCISubscriptionManager:
    """
    Manages TraCI subscriptions for batch data retrieval.
    
    Reduces IPC round-trips by subscribing to multiple variables at once
    and reading all results in a single call.
    """
    
    # TraCI constants (will use traci.constants if available)
    LAST_STEP_VEHICLE_HALTING_NUMBER = 0x14
    LAST_STEP_VEHICLE_ID_LIST = 0x12
    LAST_STEP_MEAN_SPEED = 0x11
    LAST_STEP_OCCUPANCY = 0x13
    
    def __init__(
        self,
        traci_module: Any,
        lane_ids: List[str],
        scalar_only: bool = True,
        include_id_list: bool = False,
    ):
        """
        Args:
            traci_module: The traci module (imported)
            lane_ids: List of lane IDs to subscribe to
            scalar_only: If True, only subscribe to scalar variables (safer, faster)
            include_id_list: If True, include VEHICLE_ID_LIST (needed for distinct_cycle)
        """
        self._traci = traci_module
        self._lane_ids = list(lane_ids)
        self._scalar_only = scalar_only
        self._include_id_list = include_id_list
        self._subscribed = False
        
        # Try to get constants from traci module
        try:
            import traci.constants as tc
            self.LAST_STEP_VEHICLE_HALTING_NUMBER = tc.LAST_STEP_VEHICLE_HALTING_NUMBER
            self.LAST_STEP_VEHICLE_ID_LIST = tc.LAST_STEP_VEHICLE_ID_LIST
            self.LAST_STEP_MEAN_SPEED = tc.LAST_STEP_MEAN_SPEED
            self.LAST_STEP_OCCUPANCY = tc.LAST_STEP_OCCUPANCY
        except ImportError:
            pass
    
    def subscribe_all(self) -> None:
        """
        Subscribe to all lanes. Call after traci.start().
        """
        vars_to_subscribe = [self.LAST_STEP_VEHICLE_HALTING_NUMBER]
        
        if not self._scalar_only:
            vars_to_subscribe.append(self.LAST_STEP_MEAN_SPEED)
            vars_to_subscribe.append(self.LAST_STEP_OCCUPANCY)
        
        if self._include_id_list:
            vars_to_subscribe.append(self.LAST_STEP_VEHICLE_ID_LIST)
        
        for lane_id in self._lane_ids:
            self._traci.lane.subscribe(lane_id, vars_to_subscribe)
        
        self._subscribed = True
    
    def get_results(self) -> Dict[str, Dict[int, Any]]:
        """
        Get subscription results for all lanes.
        
        Call after traci.simulationStep().
        
        Returns:
            Dict mapping lane_id -> {var_code: value}
        """
        if not self._subscribed:
            return {}
        
        return self._traci.lane.getAllSubscriptionResults()
    
    def get_halting_numbers(self) -> Dict[str, int]:
        """
        Get halting vehicle counts for all subscribed lanes.
        
        Equivalent to calling traci.lane.getLastStepHaltingNumber(lane_id)
        for each lane, but in a single batch.
        """
        results = self.get_results()
        return {
            lane_id: data.get(self.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            for lane_id, data in results.items()
        }
    
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all lanes."""
        for lane_id in self._lane_ids:
            try:
                self._traci.lane.unsubscribe(lane_id)
            except Exception:
                pass
        self._subscribed = False


def run_subscription_equivalence_test(
    traci_module: Any,
    lane_ids: List[str],
    num_steps: int = 100,
    scalar_only: bool = True,
) -> tuple[bool, str]:
    """
    Verify that subscription results match explicit TraCI calls.
    
    This is the equivalence test required before enabling subscriptions.
    
    Args:
        traci_module: The traci module (with active connection)
        lane_ids: Lanes to test
        num_steps: Number of simulation steps to compare
        scalar_only: Whether to test scalar-only mode
        
    Returns:
        (passed: bool, message: str)
    """
    manager = TraCISubscriptionManager(
        traci_module=traci_module,
        lane_ids=lane_ids,
        scalar_only=scalar_only,
        include_id_list=False,
    )
    
    manager.subscribe_all()
    
    mismatches = []
    
    for step in range(num_steps):
        traci_module.simulationStep()
        
        # Get subscription results
        sub_results = manager.get_halting_numbers()
        
        # Get explicit results
        explicit_results = {}
        for lane_id in lane_ids:
            try:
                explicit_results[lane_id] = traci_module.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                explicit_results[lane_id] = 0
        
        # Compare
        for lane_id in lane_ids:
            sub_val = sub_results.get(lane_id, 0)
            exp_val = explicit_results.get(lane_id, 0)
            
            if sub_val != exp_val:
                mismatches.append({
                    "step": step,
                    "lane": lane_id,
                    "subscription": sub_val,
                    "explicit": exp_val,
                })
    
    manager.unsubscribe_all()
    
    if len(mismatches) == 0:
        return True, f"All {num_steps} steps × {len(lane_ids)} lanes match"
    else:
        return False, f"{len(mismatches)} mismatches found: {mismatches[:5]}..."


# Test script entry point
if __name__ == "__main__":
    print("TraCI Subscription Manager")
    print("Usage: Import and use with active TraCI connection")
    print("See run_subscription_equivalence_test() for verification")
