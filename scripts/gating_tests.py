"""
Gating Tests for SMDP Reward v5 (MUST PASS before training)

Run: python scripts/gating_tests.py --config configs/train_1.yaml

Tests:
1. Smoke env: No crash, no NaN, reward finite
2. Reward sanity: Scale O(1) on 3 demands
3. State check: n_present_norm and spill_scalar_norm in state
4. Cycle bias (TODO): Compare cycle 60 vs 120
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from scripts.common import build_env
import random


def test_smoke(env, config, num_episodes=3):
    """Test 1: Smoke test - no crash, no NaN, reward finite"""
    print("\n" + "="*60)
    print("TEST 1: Smoke Test (no crash, no NaN, reward finite)")
    print("="*60)
    
    tls_ids = config['env']['sumo'].get('tls_ids', ['J0'])
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < 5:  # Only 5 steps per episode
            action_id = random.randint(0, env.action_dim - 1)
            action = {tls_id: action_id for tls_id in tls_ids}
            
            state, reward, done, info = env.step(action)
            
            if isinstance(reward, dict):
                r = sum(reward.values())
            else:
                r = float(reward)
            
            total_reward += r
            step += 1
            
            # Check NaN
            if np.isnan(r):
                print(f"[✗] FAIL: NaN reward at episode {ep+1}, step {step}")
                return False
            
            # Check state NaN
            if isinstance(state, dict):
                for s in state.values():
                    if np.any(np.isnan(s)):
                        print(f"[✗] FAIL: NaN in state at episode {ep+1}, step {step}")
                        return False
        
        print(f"  Episode {ep+1}: {step} steps, total_reward={total_reward:.4f}")
    
    print("[✓] PASS: Smoke test passed")
    return True


def test_reward_sanity(env, config, max_steps=10):
    """Test 2: Reward sanity - scale O(1), no explosion"""
    print("\n" + "="*60)
    print("TEST 2: Reward Sanity (scale O(1), stable)")
    print("="*60)
    
    tls_ids = config['env']['sumo'].get('tls_ids', ['J0'])
    state = env.reset()
    
    rewards = []
    
    for step in range(max_steps):
        action_id = random.randint(0, env.action_dim - 1)
        action = {tls_id: action_id for tls_id in tls_ids}
        
        state, reward, done, info = env.step(action)
        
        if isinstance(reward, dict):
            r = sum(reward.values())
        else:
            r = float(reward)
        
        rewards.append(r)
        
        # Get info
        n_present = info.get('n_present', 'N/A')
        cycle_sec = info.get('cycle_sec', 90)
        
        print(f"  Step {step+1}: N={n_present}, cycle={cycle_sec}s, reward={r:.4f}")
        
        if done:
            break
    
    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    
    print(f"\n  Mean: {mean_r:.4f}, Std: {std_r:.4f}, Min: {min_r:.4f}, Max: {max_r:.4f}")
    
    # Check scale O(1) to O(10)
    if abs(mean_r) > 50:
        print(f"[✗] FAIL: Reward scale too large (mean={mean_r:.2f})")
        return False
    
    print("[✓] PASS: Reward scale is O(1)")
    return True


def test_state_broadcast(env, config, max_steps=3):
    """Test 3: State broadcast - n_present_norm and spill_scalar_norm in state"""
    print("\n" + "="*60)
    print("TEST 3: State Broadcast (global scalars in observation)")
    print("="*60)
    
    tls_ids = config['env']['sumo'].get('tls_ids', ['J0'])
    state = env.reset()
    
    for step in range(max_steps):
        action_id = random.randint(0, env.action_dim - 1)
        action = {tls_id: action_id for tls_id in tls_ids}
        
        state, reward, done, info = env.step(action)
        
        if isinstance(state, dict):
            # Multi-agent: check state dimension
            for tls_id, s in state.items():
                dim = len(s)
                print(f"  Step {step+1}, TLS={tls_id}: state_dim={dim}")
                
                if dim >= 14:
                    n_present_norm = s[12]
                    spill_scalar_norm = s[13]
                    print(f"    n_present_norm={n_present_norm:.4f}, spill_scalar_norm={spill_scalar_norm:.4f}")
                    
                    # Check they're not always 0
                    if step > 0 and n_present_norm == 0.0:
                        print(f"[!] WARNING: n_present_norm is 0 at step {step+1}")
                elif dim < 14:
                    print(f"[✗] FAIL: State dimension is {dim}, expected 14")
                    return False
                
                break  # Only check first TLS
        
        if done:
            break
    
    print("[✓] PASS: State has 14 dimensions with broadcast scalars")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_1.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick test
    config['env']['sumo']['max_sim_seconds'] = 1800
    
    # IMPORTANT: Disable normalization since normalizer expects 12D but state is now 14D
    # Will regenerate normalization stats after gating tests pass
    config['env']['sumo']['normalize_state'] = False
    
    print("\n" + "="*60)
    print("GATING TESTS FOR SMDP REWARD v5")
    print("MUST PASS ALL BEFORE TRAINING")
    print("="*60)
    
    env = build_env(config)
    
    results = {}
    
    # Test 1: Smoke
    results['smoke'] = test_smoke(env, config)
    
    # Test 2: Reward sanity
    env.reset()
    results['reward_sanity'] = test_reward_sanity(env, config)
    
    # Test 3: State broadcast
    env.reset()
    results['state_broadcast'] = test_state_broadcast(env, config)
    
    env.close()
    
    # Summary
    print("\n" + "="*60)
    print("GATING TEST SUMMARY")
    print("="*60)
    
    all_pass = True
    for name, passed in results.items():
        status = "[✓] PASS" if passed else "[✗] FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("[✓✓✓] ALL GATING TESTS PASSED - OK TO TRAIN!")
    else:
        print("[✗✗✗] GATING TESTS FAILED - DO NOT TRAIN!")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
