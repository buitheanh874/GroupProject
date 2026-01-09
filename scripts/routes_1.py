from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.route_pool_loader import validate_route_file_nonempty


def _collect_routes(directory: Path, target: int) -> list[str]:
    files = sorted(directory.glob("*.rou.xml"))
    selected = []
    seen = set()
    for path in files:
        try:
            validate_route_file_nonempty(path)
        except Exception:
            continue
        name = path.name
        if name in seen:
            continue
        seen.add(name)
        selected.append(str(name))
        if len(selected) >= target:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-target", type=int, default=500)
    parser.add_argument("--eval-target", type=int, default=100)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "networks" / "variants" / "train"
    eval_dir = project_root / "networks" / "variants" / "eval"

    train_routes = _collect_routes(train_dir, max(0, int(args.train_target)))
    eval_routes = _collect_routes(eval_dir, max(0, int(args.eval_target)))
    overlap = set(train_routes).intersection(set(eval_routes))

    if len(train_routes) == 0:
        raise RuntimeError(f"No train routes found in {train_dir}")
    if len(eval_routes) == 0:
        raise RuntimeError(f"No eval routes found in {eval_dir}")
    if len(overlap) > 0:
        raise RuntimeError(f"Train and eval route lists overlap: {sorted(overlap)}")

    train_manifest = train_dir / "manifest_1.txt"
    eval_manifest = eval_dir / "manifest_1.txt"

    train_manifest.write_text("\n".join(train_routes), encoding="utf-8")
    eval_manifest.write_text("\n".join(eval_routes), encoding="utf-8")

    print(f"Wrote {len(train_routes)} train routes to {train_manifest}")
    print(f"Wrote {len(eval_routes)} eval routes to {eval_manifest}")


if __name__ == "__main__":
    main()
