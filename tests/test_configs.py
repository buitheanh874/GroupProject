from __future__ import annotations

import yaml
from pathlib import Path


def test_all_train_configs_have_normalization():
    config_dir = Path("configs")
    
    for config_file in config_dir.glob("train_*.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        sumo_cfg = cfg.get("env", {}).get("sumo", {})
        if len(sumo_cfg) == 0:
            continue
        norm_cfg = cfg.get("normalization", {})

        assert sumo_cfg.get("normalize_state") == True, \
            f"{config_file.name}: normalize_state must be True"
        
        has_file = "file" in norm_cfg
        has_inline = "mean" in norm_cfg and "std" in norm_cfg
        
        assert has_file or has_inline, \
            f"{config_file.name}: must have normalization.file or mean/std"
        
        print(f"{config_file.name}")


def test_all_train_configs_have_time_based_termination():
    config_dir = Path("configs")
    
    for config_file in config_dir.glob("train_*.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        sumo_cfg = cfg.get("env", {}).get("sumo", {})
        
        max_sim_seconds = sumo_cfg.get("max_sim_seconds")
        max_cycles = sumo_cfg.get("max_cycles", 0)
        
        if max_sim_seconds is not None and max_sim_seconds > 0:
            print(f"{config_file.name}: time-based ({max_sim_seconds}s)")
        elif max_cycles > 0:
            print(f"{config_file.name}: cycle-based ({max_cycles} cycles)")
        else:
            print(f"{config_file.name}: no termination limit set")


if __name__ == "__main__":
    print("="*80)
    print("Testing normalization settings...")
    print("="*80)
    test_all_train_configs_have_normalization()
    
    print("\n" + "="*80)
    print("Testing termination settings...")
    print("="*80)
    test_all_train_configs_have_time_based_termination()
    
    print("\nAll config tests complete")
