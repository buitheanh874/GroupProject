"""Collect norm for Medium demand - Part 2 (5 episodes)."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import build_env, load_config_with_inheritance

def main():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config_with_inheritance(str(project_root / "configs/train_final_design.yaml"))
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = True
    
    manifest = project_root / "networks/variants/train_final/manifest_medium.txt"
    routes = [str(project_root / "networks/variants/train_final" / line.strip())
              for line in manifest.read_text().splitlines() if line.strip()]
    
    random.seed(4004)
    random.shuffle(routes)
    
    all_states = []
    for i, route_file in enumerate(routes[:5]):
        config["env"]["sumo"]["route_file"] = route_file
        env = build_env(config)
        reset_result = env.reset()
        
        episode_states = []
        if isinstance(reset_result, dict):
            for s in reset_result.values():
                if isinstance(s, np.ndarray):
                    episode_states.append(s)
        elif isinstance(reset_result, np.ndarray):
            episode_states.append(reset_result)
        
        done, step = False, 0
        while not done and step < 30:
            action = {tls: np.random.randint(0, 3) for tls in env._tls_ids}
            states_dict, _, done, _ = env.step(action)
            for s in states_dict.values():
                if isinstance(s, np.ndarray):
                    episode_states.append(s)
            step += 1
        all_states.extend(episode_states)
        env.close()
        print(f"[med2] [{i+1}/5] {Path(route_file).name} - {len(episode_states)} states")
    
    states_array = np.vstack(all_states)
    output = {"mean": np.mean(states_array, axis=0).tolist(), "std": np.std(states_array, axis=0).tolist(),
              "num_samples": len(states_array), "part": "medium2"}
    (project_root / "configs/norm_medium2.json").write_text(json.dumps(output, indent=2))
    print(f"Saved: configs/norm_medium2.json ({len(all_states)} samples)")

if __name__ == "__main__": main()
