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


def newest_model(patterns: list[str]) -> Path:
    candidates = []
    for pattern in patterns:
        candidates.extend(Path("models").glob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_rl_model() -> Path:
    preferred = newest_model(["gate5_*_best.pt"])
    if preferred:
        return preferred
    fallback = newest_model(["*_best.pt"])
    if fallback:
        return fallback
    raise SystemExit("No RL checkpoint found. Expected models/gate5_*_best.pt or models/*_best.pt")


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
        (
            "eval_fixed",
            [
                sys.executable,
                str(repo_root / "scripts" / "eval_sumo.py"),
                "--config",
                "configs/train_sumo.yaml",
                "--controller",
                "fixed",
                "--episodes",
                "2",
                "--seed",
                "0",
            ],
        ),
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    rl_model = find_rl_model()
    run_step(
        "eval_rl",
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_sumo.py"),
            "--config",
            "configs/train_sumo.yaml",
            "--controller",
            "rl",
            "--model-path",
            str(rl_model),
            "--episodes",
            "2",
            "--seed",
            "0",
        ],
    )

    run_step(
        "eval_fixed_kpi",
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_sumo.py"),
            "--config",
            "configs/eval_sumo_kpi.yaml",
            "--controller",
            "fixed",
            "--episodes",
            "2",
            "--seed",
            "0",
        ],
    )

    run_step(
        "eval_rl_kpi",
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_sumo.py"),
            "--config",
            "configs/eval_sumo_kpi.yaml",
            "--controller",
            "rl",
            "--model-path",
            str(rl_model),
            "--episodes",
            "2",
            "--seed",
            "0",
        ],
    )

    print("ALL STEPS OK")


if __name__ == "__main__":
    main()
