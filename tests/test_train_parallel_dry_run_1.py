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


def test_train_parallel_disabled_config():
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
    assert "parallel.enabled is false" in result.stdout or "Use scripts/train.py" in result.stdout
