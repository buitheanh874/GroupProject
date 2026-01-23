"""
Batch generate 100 route files at base demand for curriculum training.
Usage: python scripts/generate_route_batch.py --count 100 --seed-start 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate batch of route files")
    parser.add_argument("--count", type=int, default=100, help="Number of routes to generate")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting seed")
    parser.add_argument("--base-flow", type=float, default=1000.0, help="Base flow per lane")
    parser.add_argument("--output-dir", type=str, default="networks/variants/train", help="Output directory")
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml", help="Network file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    net_file = project_root / args.net_file
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not net_file.exists():
        sys.exit(f"Network file not found: {net_file}")

    print(f"Generating {args.count} routes with base-flow={args.base_flow} veh/hr/lane")
    print(f"Seeds: {args.seed_start} to {args.seed_start + args.count - 1}")

    generated = []
    for i in range(args.count):
        seed = args.seed_start + i
        output_file = output_dir / f"bignet_curriculum_seed{seed:05d}.rou.xml"
        
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "generate_jtr_data.py"),
            "--net-file", str(net_file),
            "--output-route", str(output_file),
            "--seed", str(seed),
            "--base-flow", str(args.base_flow),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(project_root))
            if output_file.exists():
                generated.append(output_file.name)
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{args.count}")
            else:
                print(f"  [WARN] Failed to generate seed {seed}")
        except Exception as e:
            print(f"  [ERROR] Seed {seed}: {e}")

    manifest_path = output_dir / "manifest_curriculum_base.txt"
    manifest_path.write_text("\n".join(generated), encoding="utf-8")
    print(f"\nGenerated {len(generated)} routes")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
