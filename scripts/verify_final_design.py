"""Quick verification of final design config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import load_config_with_inheritance, build_env

print("=== Verifying Final Design Config ===\n")

# 1. Load config
config = load_config_with_inheritance("configs/train_final_design.yaml")
print("[OK] Config loaded successfully")

# 2. Check key settings
sumo_cfg = config["env"]["sumo"]
print(f"  action_splits: {sumo_cfg['action_splits']}")
print(f"  cycle_options: {sumo_cfg['cycle_options_sec']}")
print(f"  reward_mode: {sumo_cfg['reward_mode']}")
print(f"  state_dim: {sumo_cfg['state_dim']}")
print(f"  enable_spillback_penalty: {sumo_cfg['enable_spillback_penalty']}")

# 3. Check curriculum
curriculum = config["curriculum"]
print("[OK] Curriculum phases:")
for phase in curriculum["phases"]:
    print(f"  - {phase['name']}: {phase['episodes']} episodes")

# 4. Check exploration
eps = config["exploration"]
print("[OK] Exploration:")
print(f"  eps_start: {eps['eps_start']}")
print(f"  eps_end: {eps['eps_end']}")
print(f"  decay_steps: {eps['eps_decay_steps']}")

# 5. Try building env (quick test)
print("[OK] Building env... (may take a moment)")

# Read actual route from manifest
manifest = Path("networks/variants/train_final/manifest_easy.txt")
first_route = manifest.read_text().splitlines()[0].strip()
route_path = f"networks/variants/train_final/{first_route}"
sumo_cfg["route_file"] = route_path
print(f"  Using route: {route_path}")
try:
    env = build_env(config)
    print(f"  TLS IDs: {env._tls_ids}")
    print(f"  Action dim: {len(env._action_defs)}")
    env.close()
    print("\n=== ALL CHECKS PASSED ===")
except Exception as e:
    print(f"  ERROR: {e}")
