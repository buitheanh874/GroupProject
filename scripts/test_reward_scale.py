"""
Test script to verify NEW reward normalization (demand-invariant).
Expected: reward scale should be O(1) to O(10), not O(1000+).

Run: python scripts/test_reward_scale.py --config configs/train_1.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from scripts.common import build_env
import random
import numpy as np


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
    print("TEST: Verifying NEW reward normalization (demand-invariant)")
    print("Expected: reward scale should be O(1) to O(10), not O(1000+)")
    print("="*80)
    
    state = env.reset()
    
    print("\n{:<6} {:>12} {:>12} {:>12} {:>12}".format(
        "Step", "N_present", "Wait_total", "Cycle (s)", "Reward"
    ))
    print("-" * 60)
    
    rewards_list = []
    
    for step in range(args.max_steps):
        # Random action - use SAME action for all TLS (same cycle_sec required)
        tls_ids = config['env']['sumo'].get('tls_ids', ['J0'])
        action_id = random.randint(0, env.action_dim - 1)
        action = {tls_id: action_id for tls_id in tls_ids}
        
        state, reward, done, info = env.step(action)
        
        # Multi-agent returns dict of rewards/infos
        if isinstance(reward, dict):
            total_reward = sum(reward.values())
        else:
            total_reward = float(reward)
        
        rewards_list.append(total_reward)
        
        # Extract info - multi-agent returns global info dict
        if isinstance(info, dict):
            # Check if this is multi-agent format (has n_present directly)
            if 'n_present' in info:
                n_present = info.get('n_present', 'N/A')
                wait_total = info.get('total_wait_reward', 0)
                cycle_sec = info.get('cycle_sec', 90)
            else:
                # Single-agent or per-TLS format
                first_key = list(info.keys())[0] if info else None
                if first_key and isinstance(info.get(first_key), dict):
                    first_info = info[first_key]
                    n_present = first_info.get('n_present', 'N/A')
                    wait_total = sum(v.get('total_wait_reward', 0) for v in info.values())
                    cycle_sec = first_info.get('cycle_sec', 90)
                else:
                    n_present = 'N/A'
                    wait_total = 0
                    cycle_sec = 90
        else:
            n_present = 'N/A'
            wait_total = 0
            cycle_sec = 90
        
        print("{:<6} {:>12} {:>12.0f} {:>12} {:>12.4f}".format(
            step + 1,
            n_present,
            wait_total,
            int(cycle_sec),
            total_reward
        ))
        
        if done:
            print(f"\n[Episode ended at step {step + 1}]")
            break
    
    print("-" * 60)
    
    # Statistics
    rewards_arr = np.array(rewards_list)
    print("\n" + "="*80)
    print("REWARD STATISTICS:")
    print("="*80)
    print(f"  Mean:   {np.mean(rewards_arr):.4f}")
    print(f"  Std:    {np.std(rewards_arr):.4f}")
    print(f"  Min:    {np.min(rewards_arr):.4f}")
    print(f"  Max:    {np.max(rewards_arr):.4f}")
    print(f"  Scale:  O({int(abs(np.mean(rewards_arr))):,})")
    
    # Check if scale is correct
    mean_abs = abs(np.mean(rewards_arr))
    if mean_abs < 100:
        print("\n[✓] PASS: Reward scale is O(1) to O(100) - good for training!")
    else:
        print(f"\n[✗] FAIL: Reward scale is O({int(mean_abs):,}) - too large, needs more normalization")
    
    print()
    env.close()


if __name__ == "__main__":
    main()
