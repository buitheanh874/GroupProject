from __future__ import annotations

import yaml
from pathlib import Path


def verify_normalize_state():
    config_dir = Path("configs")
    
    errors = []
    
    for config_file in config_dir.glob("train_*.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        norm_state = cfg.get("env", {}).get("sumo", {}).get("normalize_state")
        return_raw = cfg.get("env", {}).get("sumo", {}).get("return_raw_state")
        
        if norm_state != True:
            errors.append(f"{config_file.name}: normalize_state should be True")
        if return_raw != False:
            errors.append(f"{config_file.name}: return_raw_state should be False")
    
    if errors:
        print("\n".join(errors))
        return False
    
    print("Config verification complete - all train configs OK")
    return True


def verify_termination_priority():
    config_dir = Path("configs")
    
    for config_file in config_dir.glob("train_*.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        sumo_cfg = cfg.get("env", {}).get("sumo", {})
        max_sim_seconds = sumo_cfg.get("max_sim_seconds")
        max_cycles = sumo_cfg.get("max_cycles", 0)
        
        if max_sim_seconds is not None and max_sim_seconds > 0 and max_cycles > 0:
            print(f"{config_file.name}: Both time and cycle limits set (time will take priority)")
        elif max_sim_seconds is not None and max_sim_seconds > 0:
            print(f"{config_file.name}: Time-based termination ({max_sim_seconds}s)")
        elif max_cycles > 0:
            print(f"{config_file.name}: Cycle-based termination ({max_cycles} cycles)")


if __name__ == "__main__":
    print("="*80)
    print("VERIFYING TRAIN CONFIGS")
    print("="*80)
    
    print("\n[1] Checking normalize_state settings...")
    verify_normalize_state()
    
    print("\n[2] Checking termination settings...")
    verify_termination_priority()
    
    print("\n" + "="*80)