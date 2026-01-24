"""
Test script to validate simplified configuration.

This script loads the simplified config and verifies:
1. Action space is 5 (1 cycle × 5 splits)
2. Spillback penalty is disabled
3. Time-aware gamma is disabled
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram
from env.normalization import StateNormalizer

def test_simple_config():
    print("=== Testing Simplified Configuration ===\n")
    
    # Load config
    config_path = project_root / "configs" / "train_simple.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    sumo_config = config_dict['env']['sumo']
    
    # Test 1: Check action space configuration
    print("\n[TEST 1] Action Space Configuration")
    cycle_options = sumo_config['cycle_options_sec']
    action_splits = sumo_config['action_splits']
    
    print(f"  Cycle options: {cycle_options}")
    print(f"  Action splits: {action_splits}")
    print(f"  Expected actions: {len(cycle_options)} × {len(action_splits)} = {len(cycle_options) * len(action_splits)}")
    
    assert len(cycle_options) == 1, f"Expected 1 cycle option, got {len(cycle_options)}"
    assert cycle_options[0] == 90, f"Expected cycle 90s, got {cycle_options[0]}"
    assert len(action_splits) == 5, f"Expected 5 splits, got {len(action_splits)}"
    print("  [OK] PASSED: Action space is 1x5 = 5 actions")
    
    # Test 2: Check spillback disabled
    print("\n[TEST 2] Spillback Penalty")
    enable_spillback = sumo_config['enable_spillback_penalty']
    alpha_spillback = sumo_config['alpha_spillback']
    
    print(f"  enable_spillback_penalty: {enable_spillback}")
    print(f"  alpha_spillback: {alpha_spillback}")
    
    assert enable_spillback == False, "Spillback should be disabled"
    assert alpha_spillback == 0.0, "Alpha spillback should be 0.0"
    print("  [OK] PASSED: Spillback penalty disabled")
    
    # Test 3: Check time-aware gamma disabled
    print("\n[TEST 3] Time-Aware Gamma")
    agent_config = config_dict['agent']
    use_time_aware_gamma = agent_config['use_time_aware_gamma']
    gamma = agent_config['gamma']
    
    print(f"  use_time_aware_gamma: {use_time_aware_gamma}")
    print(f"  gamma: {gamma}")
    
    assert use_time_aware_gamma == False, "Time-aware gamma should be disabled"
    assert gamma == 0.99, "Gamma should be 0.99"
    print("  [OK] PASSED: Time-aware gamma disabled")
    
    # Test 4: Check simplified network
    print("\n[TEST 4] Network Architecture")
    hidden_dims = agent_config['hidden_dims']
    
    print(f"  hidden_dims: {hidden_dims}")
    
    assert hidden_dims == [128, 128], f"Expected [128, 128], got {hidden_dims}"
    print("  [OK] PASSED: Simplified network architecture")
    
    # Test 5: Check curriculum disabled
    print("\n[TEST 5] Curriculum Learning")
    curriculum_enabled = config_dict.get('curriculum', {}).get('enabled', False)
    
    print(f"  curriculum.enabled: {curriculum_enabled}")
    
    assert curriculum_enabled == False, "Curriculum should be disabled"
    print("  [OK] PASSED: Curriculum disabled")
    
    # Test 6: Check reward simplification
    print("\n[TEST 6] Reward Simplification")
    reward_time_normalize = sumo_config['reward_time_normalize']
    use_enhanced_reward = sumo_config.get('use_enhanced_reward', False)
    
    print(f"  reward_time_normalize: {reward_time_normalize}")
    print(f"  use_enhanced_reward: {use_enhanced_reward}")
    
    assert reward_time_normalize == False, "Reward time normalization should be disabled"
    assert use_enhanced_reward == False, "Enhanced reward should be disabled"
    print("  [OK] PASSED: Reward simplified (waiting time only)")
    
    print("\n" + "="*50)
    print("[OK] ALL TESTS PASSED!")
    print("="*50)
    print("\nSimplified configuration is ready to use:")
    print(f"  - Action space: 5 actions (fixed 90s cycle)")
    print(f"  - Reward: Simple waiting time penalty")
    print(f"  - Network: [128, 128]")
    print(f"  - Training: 500 episodes, no curriculum")

if __name__ == "__main__":
    try:
        test_simple_config()
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
