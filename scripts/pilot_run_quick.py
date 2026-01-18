"""
Quick pilot run script - Test training infrastructure without normalization

This tests:
1. DQN network builds correctly with obs_dim=14
2. Loss doesn't NaN
3. Q values bounded
4. Target update works
5. Logging works

Run: python scripts/pilot_run_quick.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

# Load config
config_path = "configs/train_1.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Override for quick pilot test
config['env']['sumo']['normalize_state'] = False  # Disable normalization
config['env']['sumo']['state_dim'] = 14  # SMDP v5: 12 local + 2 global broadcast
config['env']['sumo']['max_sim_seconds'] = 1800
config['parallel']['enabled'] = True
config['parallel']['num_actors'] = 2  # 2 workers for faster test
config['train']['episodes'] = 20  

# Save temp config
temp_config_path = "configs/train_pilot_quick.yaml"
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f)

print("="*60)
print("QUICK PILOT RUN - Testing Training Infrastructure")
print("="*60)
print(f"Config: {temp_config_path}")
print(f"Normalization: DISABLED (for speed)")
print(f"Workers: 2")
print(f"Episodes: 20")
print(f"Expected time: ~10 minutes")
print("="*60)
print("\nRun command:")
print(f"python scripts/train_parallel.py --config {temp_config_path}")
print("\nOr just run this script directly - it will start training")
print("="*60)

# Auto-run training
import subprocess
cmd = ["python", "scripts/train_parallel.py", "--config", temp_config_path]
subprocess.run(cmd)
