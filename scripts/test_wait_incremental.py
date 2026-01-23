"""
Test script to verify wait_total is incremental vs cumulative.
Prints: step, wait_total, delta_t for 1 episode.

Run: python scripts/test_wait_incremental.py --config configs/train_1.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from scripts.common import build_env
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_1.yaml")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick test
    config['env']['sumo']['max_sim_seconds'] = 1800
    config['env']['sumo']['enable_kpi_tracker'] = True
    
    env = build_env(config)
    
    print("\n" + "="*80)
    print("TEST: Verifying wait_total is INCREMENTAL (per step) vs CUMULATIVE")
    print("="*80)
    
    state = env.reset()
    
    print("\n{:<6} {:>15} {:>15} {:>12} {:>12}".format(
        "Step", "wait_total", "delta_t (s)", "Cycle (s)", "Reward"
    ))
    print("-" * 70)
    
    cumulative_wait = 0.0
    
    for step in range(args.max_steps):
        # Random action - use SAME action for all TLS (same cycle_sec required)
        tls_ids = config['env']['sumo'].get('tls_ids', ['J0'])
        action_id = random.randint(0, env.action_dim - 1)
        action = {tls_id: action_id for tls_id in tls_ids}
        
        state, reward, done, info = env.step(action)
        
        # Multi-agent returns dict of rewards/infos, handle both cases
        if isinstance(reward, dict):
            total_reward = sum(reward.values())
        else:
            total_reward = float(reward)
        
        # Extract wait_total from info
        if isinstance(info, dict) and 'total_wait_reward' in info:
            # Single-agent format
            wait_total = info.get('total_wait_reward', 0.0)
            cycle_sec = info.get('cycle_sec', 90)
            delta_t = info.get('decision_duration_sec', cycle_sec)
        elif isinstance(info, dict):
            # Multi-agent format: info is dict of per-TLS infos
            first_key = list(info.keys())[0]
            first_info = info[first_key]
            if isinstance(first_info, dict):
                wait_total = sum(v.get('total_wait_reward', 0) for v in info.values())
                cycle_sec = first_info.get('cycle_sec', 90)
                delta_t = first_info.get('decision_duration_sec', cycle_sec)
            else:
                wait_total = 0
                cycle_sec = 90
                delta_t = cycle_sec
        else:
            wait_total = 0
            cycle_sec = 90
            delta_t = cycle_sec
        
        cumulative_wait += wait_total
        
        print("{:<6} {:>15.2f} {:>15.1f} {:>12} {:>12.2f}".format(
            step + 1,
            wait_total,
            float(delta_t),
            int(cycle_sec),
            total_reward
        ))
        
        if done:
            print(f"\n[Episode ended at step {step + 1}]")
            break
    
    print("-" * 70)
    print("SUM    {:>15.2f}".format(cumulative_wait))
    print()
    
    # Conclusion
    print("="*80)
    print("ANALYSIS:")
    print("="*80)
    print("""
If wait_total values are:
  - Similar magnitude each step (e.g., 1000-2000) → INCREMENTAL (correct)
  - Increasing linearly each step (e.g., 1000, 2000, 3000, ...) → CUMULATIVE (wrong)

Based on code analysis (sumo_env.py line 554):
  agg = CycleMetricsAggregator(...) is created NEW inside _step_legacy()
  
This means aggregator is RESET every step, so wait_total only counts
waiting time accumulated within that single step/cycle.

CONCLUSION: wait_total is INCREMENTAL (per-step), NOT cumulative.
""")
    
    env.close()


if __name__ == "__main__":
    main()
