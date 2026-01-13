from __future__ import annotations

import subprocess
import sys


def test_train_parallel_dry_run_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_parallel.py",
            "--config",
            "configs/train_parallel_smoke_1.yaml",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Exit code was {result.returncode}, stderr: {result.stderr}"


def test_train_parallel_enabled_config():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_parallel.py",
            "--config",
            "configs/train_1.yaml",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Dry run complete" in result.stdout or "Parallel Training Plan" in result.stdout

