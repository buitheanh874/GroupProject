"""
TraCI subscription equivalence test.

Verifies that subscription results == explicit TraCI calls for 1000 steps.

Usage:
    python tests/test_traci_subscription_equivalence.py --config configs/train_final_design.yaml --steps 1000

Acceptance criteria:
    - All subscription values must exactly match explicit get() calls
    - No mismatches allowed

This test must PASS before enabling use_traci_subscriptions in performance config.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="TraCI subscription equivalence test")
    parser.add_argument("--config", default="configs/train_final_design.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    from rl.utils import load_yaml_config, set_global_seed
    from scripts.common import build_env
    
    config = load_yaml_config(args.config)
    config["run"] = config.get("run", {})
    config["run"]["seed"] = args.seed
    
    set_global_seed(args.seed)
    
    print(f"Building env from {args.config}...")
    env = build_env(config)
    
    # Get lane IDs from config
    sumo_cfg = config.get("env", {}).get("sumo", {})
    lane_groups = sumo_cfg.get("lane_groups_by_tls", {})
    
    all_lanes = set()
    for tls_id, groups in lane_groups.items():
        for key in ["lanes_ns_ctrl", "lanes_ew_ctrl"]:
            lanes = groups.get(key, [])
            all_lanes.update(lanes)
    
    lane_ids = sorted(all_lanes)
    print(f"Testing {len(lane_ids)} lanes for {args.steps} steps")
    
    # Reset env to start SUMO
    env.reset()
    
    # Get traci module from env
    traci = env._traci
    
    from rl.traci_subscriptions import TraCISubscriptionManager
    
    manager = TraCISubscriptionManager(
        traci_module=traci,
        lane_ids=lane_ids,
        scalar_only=True,
        include_id_list=False,
    )
    
    manager.subscribe_all()
    
    mismatches = []
    total_comparisons = 0
    
    print(f"Running {args.steps} simulation steps...")
    
    for step in range(args.steps):
        traci.simulationStep()
        
        # Get subscription results
        sub_results = manager.get_halting_numbers()
        
        # Get explicit results
        for lane_id in lane_ids:
            sub_val = sub_results.get(lane_id, 0)
            
            try:
                exp_val = traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                exp_val = 0
            
            total_comparisons += 1
            
            if sub_val != exp_val:
                mismatches.append({
                    "step": step,
                    "lane": lane_id,
                    "subscription": sub_val,
                    "explicit": exp_val,
                })
        
        if (step + 1) % 200 == 0:
            print(f"  Step {step + 1}/{args.steps}...")
    
    env.close()
    
    # Report results
    print()
    print("=" * 50)
    print("TRACI SUBSCRIPTION EQUIVALENCE TEST RESULTS")
    print("=" * 50)
    print(f"  Steps tested: {args.steps}")
    print(f"  Lanes tested: {len(lane_ids)}")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Mismatches: {len(mismatches)}")
    print()
    
    if len(mismatches) == 0:
        print("RESULT: PASS")
        print("Subscription values exactly match explicit get() calls.")
        return 0
    else:
        print("RESULT: FAIL")
        print(f"Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:
            print(f"  Step {m['step']}, Lane {m['lane']}: sub={m['subscription']} vs exp={m['explicit']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        return 1


if __name__ == "__main__":
    sys.exit(main())
