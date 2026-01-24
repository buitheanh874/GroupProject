"""Merge normalization stats from 3 demand levels with curriculum weighting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Load individual norm files
    easy_path = project_root / "configs/norm_easy.json"
    medium_path = project_root / "configs/norm_medium.json"
    hard_path = project_root / "configs/norm_hard.json"
    
    easy = json.loads(easy_path.read_text())
    medium = json.loads(medium_path.read_text())
    hard = json.loads(hard_path.read_text())
    
    # Weighted average based on curriculum (60/30/10)
    total_samples = easy["num_samples"] + medium["num_samples"] + hard["num_samples"]
    
    # Combine all samples for proper statistics
    # Weight by number of samples (which already reflects 60/30/10 ratio)
    mean_combined = (
        np.array(easy["mean"]) * easy["num_samples"] +
        np.array(medium["mean"]) * medium["num_samples"] +
        np.array(hard["mean"]) * hard["num_samples"]
    ) / total_samples
    
    # For std, we need to combine variances properly
    var_easy = np.array(easy["std"]) ** 2
    var_medium = np.array(medium["std"]) ** 2
    var_hard = np.array(hard["std"]) ** 2
    
    var_combined = (
        var_easy * easy["num_samples"] +
        var_medium * medium["num_samples"] +
        var_hard * hard["num_samples"]
    ) / total_samples
    
    std_combined = np.sqrt(var_combined)
    std_combined = np.maximum(std_combined, 1e-6)  # Avoid division by zero
    
    merged = {
        "mean": mean_combined.tolist(),
        "std": std_combined.tolist(),
        "state_dim": len(mean_combined),
        "total_samples": int(total_samples),
        "curriculum_ratio": "60/30/10",
        "sources": {
            "easy": {"samples": easy["num_samples"], "episodes": easy["num_episodes"]},
            "medium": {"samples": medium["num_samples"], "episodes": medium["num_episodes"]},
            "hard": {"samples": hard["num_samples"], "episodes": hard["num_episodes"]},
        }
    }
    
    output_path = project_root / "configs/norm_final_design.json"
    output_path.write_text(json.dumps(merged, indent=2))
    
    print(f"Merged normalization stats:")
    print(f"  Easy: {easy['num_samples']} samples ({easy['num_episodes']} epi)")
    print(f"  Medium: {medium['num_samples']} samples ({medium['num_episodes']} epi)")
    print(f"  Hard: {hard['num_samples']} samples ({hard['num_episodes']} epi)")
    print(f"  Total: {merged['total_samples']} samples")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
