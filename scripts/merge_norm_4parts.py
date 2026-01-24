"""Merge 4 norm parts into final norm file."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    parts = ["norm_easy1.json", "norm_easy2.json", "norm_medium1.json", "norm_medium2.json"]
    all_data = []
    total_samples = 0
    
    for p in parts:
        path = project_root / "configs" / p
        if not path.exists():
            print(f"Missing: {p}")
            continue
        data = json.loads(path.read_text())
        all_data.append(data)
        total_samples += data["num_samples"]
        print(f"Loaded {p}: {data['num_samples']} samples")
    
    if len(all_data) < 4:
        print("Warning: Not all parts found!")
    
    # Weighted average
    mean_combined = np.zeros(len(all_data[0]["mean"]))
    var_combined = np.zeros(len(all_data[0]["std"]))
    
    for data in all_data:
        w = data["num_samples"] / total_samples
        mean_combined += np.array(data["mean"]) * w
        var_combined += (np.array(data["std"]) ** 2) * w
    
    std_combined = np.sqrt(np.maximum(var_combined, 1e-12))
    
    merged = {
        "mean": mean_combined.tolist(),
        "std": std_combined.tolist(),
        "state_dim": len(mean_combined),
        "total_samples": total_samples,
        "parts": [d.get("part", "unknown") for d in all_data]
    }
    
    output_path = project_root / "configs/norm_final_design.json"
    output_path.write_text(json.dumps(merged, indent=2))
    print(f"\nMerged! Total: {total_samples} samples -> {output_path}")

if __name__ == "__main__": main()
