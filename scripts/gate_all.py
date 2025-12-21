from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"START {step_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        print(f"FAIL {step_name} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"STEP OK {step_name}")


def main() -> None:
    steps = [
        ("doctor", [sys.executable, str(repo_root / "scripts" / "doctor.py")]),
        (
            "smoke_baseline",
            [
                sys.executable,
                str(repo_root / "scripts" / "smoke_baseline.py"),
                "--config",
                "configs/train_sumo.yaml",
                "--cycles",
                "3",
                "--seed",
                "0",
            ],
        ),
        (
            "collect_norm_stats",
            [
                sys.executable,
                str(repo_root / "scripts" / "collect_norm_stats.py"),
                "--config",
                "configs/train_sumo.yaml",
                "--episodes",
                "2",
                "--seed",
                "0",
            ],
        ),
        (
            "run_sumo_episode",
            [
                sys.executable,
                str(repo_root / "scripts" / "run_sumo_episode.py"),
                "--config",
                "configs/train_sumo.yaml",
                "--seed",
                "0",
                "--cycles",
                "3",
            ],
        ),
        (
            "gate_train_5eps",
            [
                sys.executable,
                str(repo_root / "scripts" / "gate_train_5eps.py"),
                "--config",
                "configs/train_sumo.yaml",
                "--seed",
                "0",
            ],
        ),
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    print("ALL STEPS OK")


if __name__ == "__main__":
    main()
