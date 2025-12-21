from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config


def ensure_norm_stats(config_path: str) -> None:
    config = load_yaml_config(config_path)
    norm_cfg = config.get("normalization", {})
    norm_file = norm_cfg.get("file")
    sumo_cfg = config.get("env", {}).get("sumo", {})
    normalize_state = bool(sumo_cfg.get("normalize_state", True))

    if not normalize_state or not norm_file:
        return

    norm_path = Path(norm_file)
    if norm_path.is_file():
        return

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "collect_norm_stats.py"),
        "--config",
        config_path,
        "--episodes",
        "2",
        "--seed",
        "0",
    ]
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError("Failed to generate normalization stats")

    if not norm_path.is_file():
        raise RuntimeError(f"Normalization file was not created: {norm_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_sumo.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = str(Path(args.config))
    ensure_norm_stats(config_path)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--config",
        config_path,
        "--episodes",
        "5",
        "--seed",
        str(args.seed),
        "--run-name",
        "gate5",
    ]

    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        sys.exit(result.returncode)

    config = load_yaml_config(config_path)
    logging_cfg: Dict[str, str] = config.get("logging", {})
    log_dir = logging_cfg.get("log_dir", "logs")
    print(f"Gate run complete. Check logs in {Path(log_dir).resolve()}")


if __name__ == "__main__":
    main()
