"""Collect normalization stats for Easy demand (350 veh/hr/lane)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import build_env, load_config_with_inheritance


def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = str(project_root / "configs/train_final_design.yaml")
    
    config = load_config_with_inheritance(config_path)
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = True
    
    # Load easy routes
    manifest = project_root / "networks/variants/train_final/manifest_easy.txt"
    routes = [str(project_root / "networks/variants/train_final" / line.strip())
              for line in manifest.read_text().splitlines() if line.strip()]
    
    all_states = []
    num_episodes = 20  # Reduced for faster collection
    
    # Shuffle routes for random selection
    random.shuffle(routes)
    
    for i, route_file in enumerate(routes[:num_episodes]):
        config["env"]["sumo"]["route_file"] = route_file
        env = build_env(config)
        
        state = env.reset()
        episode_states = [state]
        
        done = False
        step = 0
        while not done and step < 30:
            action = {tls: np.random.randint(0, 3) for tls in env._tls_ids}
            states_dict, _, done, _ = env.step(action)
            for tls_state in states_dict.values():
                episode_states.append(tls_state)
            step += 1
        
        all_states.extend(episode_states)
        env.close()
        print(f"[{i+1}/{num_episodes}] {Path(route_file).name} - {len(episode_states)} states")
    
    states_array = np.array(all_states)
    
    output = {
        "mean": np.mean(states_array, axis=0).tolist(),
        "std": np.std(states_array, axis=0).tolist(),
        "num_samples": len(states_array),
        "num_episodes": num_episodes,
        "demand_level": "easy_350",
    }
    
    output_path = project_root / "configs/norm_easy.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {output_path}")
    print(f"Samples: {output['num_samples']}, State dim: {len(output['mean'])}")


if __name__ == "__main__":
    main()
