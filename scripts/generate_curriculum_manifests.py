from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def generate_curriculum_manifests(train_dir: Path, scales: List[int], num_files: int = 100) -> None:
    for scale in scales:
        folder = train_dir / f"scaled_{scale}"
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue
        
        files = sorted(folder.glob("*.rou.xml"))[:num_files]
        if len(files) == 0:
            print(f"[WARN] No route files in {folder}")
            continue
        
        manifest_path = train_dir / f"manifest_curriculum_scale{scale}.txt"
        lines = [f"scaled_{scale}/{f.name}" for f in files]
        manifest_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Created {manifest_path.name}: {len(lines)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="networks/variants/train")
    parser.add_argument("--scales", nargs="+", type=int, default=[20, 30, 40, 50, 60])
    parser.add_argument("--num-files", type=int, default=100)
    args = parser.parse_args()
    
    train_dir = Path(args.train_dir)
    generate_curriculum_manifests(train_dir, args.scales, args.num_files)
    print("Done!")


if __name__ == "__main__":
    main()
