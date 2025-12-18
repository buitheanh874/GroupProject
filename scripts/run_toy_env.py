from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root / "configs" / "train_toy.yaml"))
    args = parser.parse_args()

    sys.argv = [sys.argv[0], "--config", str(args.config)]
    train_main()


if __name__ == "__main__":
    main()
