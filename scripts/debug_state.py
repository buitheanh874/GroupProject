"""Debug script to check state values during simulation."""
import sys
sys.path.insert(0, '.')

from scripts.common import build_env
from rl.utils import load_yaml_config
from scripts.route_pool_loader import load_route_pool_from_config
import numpy as np

from pathlib import Path

def main():
    config = load_yaml_config('configs/train_bignet_short.yaml')
    routes = load_route_pool_from_config(config, split='train', project_root=Path('.'))
    
    # Temporarily disable normalization to see raw values
    config['env']['sumo']['normalize_state'] = False
    
    env = build_env(config)
    
    print("="*60)
    print("STATE VECTOR DEBUG")
    print("="*60)
    print(f"State dim: {env.state_dim}")
    print(f"TLS IDs: {env._tls_ids}")
    print(f"Center TLS: {env._center_tls_id}")
    print()
    
    state = env.reset()
    
    print("Initial state (after reset):")
    if isinstance(state, dict):
        for tls_id, s in state.items():
            print(f"  [{tls_id}]: {s}")
    else:
        print(f"  {state}")
    
    # Run a few steps and collect states
    all_states = []
    for step in range(10):
        # Same action for all TLS (required: same cycle_sec)
        action_id = np.random.randint(0, env.action_dim)
        actions = {tls_id: action_id for tls_id in env._tls_ids}
        next_state, reward, done, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        if isinstance(next_state, dict):
            for tls_id, s in next_state.items():
                all_states.append(s)
                q = s[0:4]  # queue N,E,S,W
                w = s[4:8]  # wait N,E,S,W
                occ = s[8:12]  # downstream occ
                print(f"  [{tls_id}] queue={q}, wait={w}, occ={occ}")
        
        if done:
            print("Episode done!")
            break
    
    # Summary
    all_states = np.array(all_states)
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL STATES:")
    print("="*60)
    print("Feature index meanings:")
    print("  0-3: Queue [N, E, S, W]")
    print("  4-7: Wait  [N, E, S, W]")
    print("  8-11: Downstream Occupancy [N, E, S, W]")
    print()
    
    for i in range(12):
        vals = all_states[:, i]
        print(f"  Feature {i:2d}: min={vals.min():10.2f}, max={vals.max():10.2f}, mean={vals.mean():10.2f}, std={vals.std():10.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
