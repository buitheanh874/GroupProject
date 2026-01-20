#!/usr/bin/env python
"""
Baseline Semantics Audit - Verify controller fairness

Audits each controller to verify:
1. action_id outputs are in {0..14}
2. Actions map to cycles in {60, 90, 120}
3. Actions map to splits in {0.3, 0.4, 0.5, 0.6, 0.7}

Usage:
    python scripts/audit_baselines.py --episodes 100
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


@dataclass
class AuditResult:
    controller: str
    total_decisions: int
    action_id_range: tuple  # (min, max)
    action_id_distribution: Dict[int, int]
    cycle_distribution: Dict[int, int]
    split_distribution: Dict[float, int]
    is_internal_baseline: bool
    constraint_violations: List[str]


def build_action_space():
    """Build the 15-discrete action space: 3 cycles × 5 splits."""
    @dataclass
    class ActionDef:
        cycle_sec: int
        rho_ns: float
        rho_ew: float
    
    cycles = [60, 90, 120]
    splits = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    action_space = []
    for cycle in cycles:
        for split in splits:
            action_space.append(ActionDef(
                cycle_sec=cycle,
                rho_ns=split,
                rho_ew=1.0 - split,
            ))
    
    return action_space


def generate_random_states(n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate random state vectors for testing."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        # State format: [q_N, q_E, q_S, q_W, ...] with 14 dims
        state = rng.uniform(0, 20, size=14).astype(np.float32)
        states.append(state)
    return states


def audit_controller(
    controller_name: str,
    controller: Any,
    action_space: List[Any],
    states: List[np.ndarray],
) -> AuditResult:
    """Audit a single controller's decisions."""
    
    action_ids = []
    cycles = []
    splits = []
    violations = []
    
    for i, state in enumerate(states):
        try:
            if controller_name == "fixed":
                action_id = controller.act()
            else:
                # Actuated/Webster/MaxPressure need state
                action_id = controller.act(state, tls_id="J0", current_time=float(i * 90))
        except Exception as e:
            violations.append(f"act() failed at step {i}: {str(e)[:50]}")
            continue
        
        action_ids.append(action_id)
        
        # Validate action_id range
        if not (0 <= action_id < len(action_space)):
            violations.append(f"action_id={action_id} out of range [0,{len(action_space)-1}]")
        else:
            action_def = action_space[action_id]
            cycles.append(action_def.cycle_sec)
            splits.append(round(action_def.rho_ns, 1))
    
    # Check if cycles are in {60, 90, 120}
    valid_cycles = {60, 90, 120}
    for c in set(cycles):
        if c not in valid_cycles:
            violations.append(f"cycle={c} not in {valid_cycles}")
    
    # Check if splits are in {0.3, 0.4, 0.5, 0.6, 0.7}
    valid_splits = {0.3, 0.4, 0.5, 0.6, 0.7}
    for s in set(splits):
        if s not in valid_splits:
            violations.append(f"split={s} not in {valid_splits}")
    
    is_internal = len(violations) == 0
    
    return AuditResult(
        controller=controller_name,
        total_decisions=len(action_ids),
        action_id_range=(min(action_ids) if action_ids else -1, max(action_ids) if action_ids else -1),
        action_id_distribution=dict(Counter(action_ids)),
        cycle_distribution=dict(Counter(cycles)),
        split_distribution=dict(Counter(splits)),
        is_internal_baseline=is_internal,
        constraint_violations=violations,
    )


def print_audit_report(results: List[AuditResult]) -> None:
    """Print audit report in human-readable format."""
    
    print("\n" + "=" * 70)
    print("BASELINE SEMANTICS AUDIT REPORT")
    print("=" * 70)
    print(f"Valid action_id range: [0, 14]")
    print(f"Valid cycles: {{60, 90, 120}}")
    print(f"Valid splits (rho_ns): {{0.3, 0.4, 0.5, 0.6, 0.7}}")
    print("=" * 70 + "\n")
    
    for r in results:
        status = "✅ INTERNAL BASELINE" if r.is_internal_baseline else "⚠️ EXTERNAL BASELINE"
        print(f"[{r.controller}] {status}")
        print(f"  Total decisions: {r.total_decisions}")
        print(f"  Action ID range: {r.action_id_range}")
        
        # Cycle distribution
        cycle_str = ", ".join(f"{c}s:{n}" for c, n in sorted(r.cycle_distribution.items()))
        print(f"  Cycle distribution: {cycle_str if cycle_str else 'N/A'}")
        
        # Split distribution
        split_str = ", ".join(f"{s:.1f}:{n}" for s, n in sorted(r.split_distribution.items()))
        print(f"  Split distribution: {split_str if split_str else 'N/A'}")
        
        if r.constraint_violations:
            print(f"  Violations ({len(r.constraint_violations)}):")
            for v in r.constraint_violations[:5]:
                print(f"    - {v}")
            if len(r.constraint_violations) > 5:
                print(f"    ... and {len(r.constraint_violations) - 5} more")
        print()
    
    # Summary
    internal = sum(1 for r in results if r.is_internal_baseline)
    external = len(results) - internal
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Internal baselines (same constraints as RL): {internal}")
    print(f"External baselines (noted constraint differences): {external}")
    
    if external > 0:
        print("\n⚠️ External baselines should be reported with constraint differences noted.")
        print("   Consider fixing controller to map to valid action_ids if possible.")
    else:
        print("\n✅ All baselines use same semantics and constraints as RL agent.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit baseline controller semantics")
    parser.add_argument("--episodes", type=int, default=100, help="Number of decisions to audit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for state generation")
    args = parser.parse_args(argv)
    
    action_space = build_action_space()
    states = generate_random_states(args.episodes, args.seed)
    
    print(f"Auditing baselines with {args.episodes} decisions each...")
    
    results = []
    
    # 1. Fixed Time Controller
    try:
        from controllers.fixed_time import FixedTimeController, FixedTimeControllerConfig
        config = FixedTimeControllerConfig(target_split=(0.5, 0.5), target_cycle_sec=90)
        controller = FixedTimeController(action_space=action_space, config=config)
        result = audit_controller("fixed", controller, action_space, states)
        results.append(result)
    except Exception as e:
        print(f"Failed to audit FixedTimeController: {e}")
    
    # 2. Actuated Controller
    try:
        from controllers.actuated import ActuatedController
        controller = ActuatedController(action_space=action_space)
        controller.reset()
        result = audit_controller("actuated", controller, action_space, states)
        results.append(result)
    except Exception as e:
        print(f"Failed to audit ActuatedController: {e}")
    
    # 3. Webster Controller
    try:
        from controllers.webster import WebsterController
        controller = WebsterController(action_space=action_space)
        controller.reset()
        result = audit_controller("webster", controller, action_space, states)
        results.append(result)
    except Exception as e:
        print(f"Failed to audit WebsterController: {e}")
    
    # 4. MaxPressure (uses select_action_from_defs, not a class)
    try:
        from controllers.max_pressure import select_action_from_defs
        
        action_ids = []
        for state in states:
            action_id = select_action_from_defs(
                state_raw=state,
                action_defs=action_space,
                allowed_action_ids=None,
                default_action_id=7,
            )
            action_ids.append(action_id)
        
        cycles = [action_space[a].cycle_sec for a in action_ids if 0 <= a < len(action_space)]
        splits = [round(action_space[a].rho_ns, 1) for a in action_ids if 0 <= a < len(action_space)]
        
        result = AuditResult(
            controller="max_pressure",
            total_decisions=len(action_ids),
            action_id_range=(min(action_ids), max(action_ids)),
            action_id_distribution=dict(Counter(action_ids)),
            cycle_distribution=dict(Counter(cycles)),
            split_distribution=dict(Counter(splits)),
            is_internal_baseline=all(0 <= a < len(action_space) for a in action_ids),
            constraint_violations=[],
        )
        results.append(result)
    except Exception as e:
        print(f"Failed to audit MaxPressure: {e}")
    
    print_audit_report(results)
    
    # Return non-zero if any external baselines
    external_count = sum(1 for r in results if not r.is_internal_baseline)
    return 1 if external_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
