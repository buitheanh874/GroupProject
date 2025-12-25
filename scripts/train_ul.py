from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def run_step(description: str, cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=repo_root)
    
    if result.returncode != 0:
        sys.exit(f"Step '{description}' failed with exit code {result.returncode}")
    
    print(f"\n[SUCCESS] {description}")


def main() -> None:
    print("=" * 80)
    print("Training Ultimate Scenario Models")
    print("=" * 80)

    run_step(
        description="Step 1: Collect normalization stats",
        cmd=[sys.executable, "scripts/collect_norm_stats_ultimate.py"],
    )

    run_step(
        description="Step 2: Train Pure Model (λ=0.0)",
        cmd=[
            sys.executable,
            "scripts/train.py",
            "--config", "configs/train_ultimate_pure.yaml",
            "--episodes", "250",
        ],
    )

    run_step(
        description="Step 3: Train Fair Model (λ=0.12)",
        cmd=[
            sys.executable,
            "scripts/train.py",
            "--config", "configs/train_ultimate_fair.yaml",
            "--episodes", "250",
        ],
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("Pure model: models/ultimate_pure/")
    print("Fair model: models/ultimate_fair/")
    print("=" * 80)


if __name__ == "__main__":
    main()