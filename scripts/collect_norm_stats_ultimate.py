from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def main() -> None:
    print("=" * 80)
    print("Collecting Normalization Stats for Ultimate Scenario")
    print("=" * 80)

    config_path = "configs/train_ultimate_pure.yaml"
    output_path = "configs/norm_stats_ultimate_clean.json"

    output_file = Path(output_path)
    
    if not output_file.exists():
        print(f"\nCreating dummy {output_path}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_stats = {
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "episodes": 0,
            "seed": 0,
            "num_samples": 0,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dummy_stats, f, indent=2)

    print("\nRunning normalization collection...")
    cmd = [
        sys.executable,
        "scripts/collect_norm_stats.py",
        "--config", config_path,
        "--episodes", "5",
        "--seed", "0",
        "--out", output_path,
    ]
    
    result = subprocess.run(cmd, cwd=repo_root)
    
    if result.returncode != 0:
        sys.exit(f"Normalization collection failed with exit code {result.returncode}")

    print("\n" + "=" * 80)
    print(f"Normalization stats saved to: {output_path}")
    print("=" * 80)
    
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()