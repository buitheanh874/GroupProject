"""
Collect normalization stats for final design (3 demand levels, no downstream).

Quick collection: 10 episodes per demand level = 30 total episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import build_env_from_config, load_yaml_config


def collect_stats_for_routes(config_path: Path, route_files: list[str], num_episodes: int = 10):
    """Collect normalization stats from multiple routes."""
    config = load_yaml_config(config_path)
    
    # Disable normalization for collection
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = True
    
    all_states = []
    
    for i, route_file in enumerate(route_files[:num_episodes]):
        config["env"]["sumo"]["route_file"] = route_file
        env = build_env_from_config(config)
        
        state = env.reset()
        episode_states = [state]
        
        done = False
        step = 0
        while not done and step < 30:  # Max 30 steps
            # Random action
            action = {tls: np.random.randint(0, 3) for tls in env._tls_ids}
            states_dict, _, done, _ = env.step(action)
            
            # Collect all TLS states
            for tls_state in states_dict.values():
                episode_states.append(tls_state)
            step += 1
        
        all_states.extend(episode_states)
        env.close()
        print(f"[{i+1}/{num_episodes}] {Path(route_file).name} - {len(episode_states)} states")
    
    return np.array(all_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_final_design.yaml")
    parser.add_argument("--output", default="configs/norm_final_design.json")
    parser.add_argument("--episodes-per-level", type=int, default=10)
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    
    # Load manifests
    easy_manifest = project_root / "networks/variants/train_final/manifest_easy.txt"
    medium_manifest = project_root / "networks/variants/train_final/manifest_medium.txt"
    hard_manifest = project_root / "networks/variants/train_final/manifest_hard.txt"
    
    easy_routes = [str(project_root / "networks/variants/train_final" / line.strip().replace("easy/", "")) 
                   for line in easy_manifest.read_text().splitlines() if line.strip()]
    medium_routes = [str(project_root / "networks/variants/train_final" / line.strip().replace("medium/", "")) 
                     for line in medium_manifest.read_text().splitlines() if line.strip()]
    hard_routes = [str(project_root / "networks/variants/train_final" / line.strip().replace("hard/", "")) 
                   for line in hard_manifest.read_text().splitlines() if line.strip()]
    
    print(f"Collecting from {args.episodes_per_level} episodes per demand level...")
    
    print("\n=== Easy (350 veh/hr/lane) ===")
    states_easy = collect_stats_for_routes(config_path, easy_routes, args.episodes_per_level)
    
    print("\n=== Medium (500 veh/hr/lane) ===")
    states_medium = collect_stats_for_routes(config_path, medium_routes, args.episodes_per_level)
    
    print("\n=== Hard (650 veh/hr/lane) ===")
    states_hard = collect_stats_for_routes(config_path, hard_routes, args.episodes_per_level)
    
    # Combine all states
    all_states = np.vstack([states_easy, states_medium, states_hard])
    
    print(f"\nTotal states collected: {len(all_states)}")
    print(f"State shape: {all_states.shape}")
    
    # Compute stats
    mean = np.mean(all_states, axis=0).tolist()
    std = np.std(all_states, axis=0).tolist()
    std = [max(s, 1e-6) for s in std]  # Avoid division by zero
    
    norm_stats = {
        "mean": mean,
        "std": std,
        "state_dim": len(mean),
        "num_samples": len(all_states),
        "demand_levels": ["easy_350", "medium_500", "hard_650"],
        "episodes_per_level": args.episodes_per_level,
    }
    
    output_path = project_root / args.output
    output_path.write_text(json.dumps(norm_stats, indent=2))
    print(f"\nNormalization stats saved to: {output_path}")
    print(f"State dim: {norm_stats['state_dim']}")


if __name__ == "__main__":
    main()
