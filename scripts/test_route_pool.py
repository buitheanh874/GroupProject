#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)

    return result


def extract_route_selections(log_text: str) -> list[str]:
    pattern = r"Using route '([^']+\.rou\.xml)'"
    return re.findall(pattern, log_text)


def main() -> None:
    print("=" * 80)
    print("ROUTE POOL FUNCTIONALITY TEST")
    print("=" * 80)
    print()

    calib_path = repo_root / "configs" / "scenario_hanoi_calibration.yaml"
    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        print()
        print("Generate it first:")
        print("  python scripts/inspect_net_boundaries.py \\")
        print("    --net networks/BIGMAP.net.xml \\")
        print(f"    --out {calib_path}")
        sys.exit(1)

    print("STEP 1: Generate test route variants")
    variants_dir = repo_root / "networks" / "test_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    for old_file in variants_dir.glob("*.rou.xml"):
        old_file.unlink()

    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_hanoi_route_variants.py"),
            "--calib",
            str(calib_path),
            "--out-dir",
            str(variants_dir),
            "--split",
            "train",
            "--n",
            "3",
            "--seed",
            "42",
            "--level",
            "med",
        ]
    )

    variant_files = sorted(list(variants_dir.glob("*.rou.xml")))
    if len(variant_files) < 3:
        print(f"\n[ERROR] Expected 3 variants, found {len(variant_files)}")
        sys.exit(1)

    print(f"\nGenerated {len(variant_files)} route variants")

    print()
    print("STEP 2: Create test config with route pool")

    from rl.utils import load_yaml_config, save_yaml_config

    test_config = repo_root / "configs" / "test_route_pool.yaml"
    base_config = load_yaml_config(str(repo_root / "configs" / "train_sumo.yaml"))

    base_config["run"]["run_name"] = "test_route_pool"
    base_config["train"]["episodes"] = 10
    base_config["train"]["print_every_episodes"] = 1
    base_config["train"]["route_pool"] = [str(f.relative_to(repo_root)).replace("\\", "/") for f in variant_files]
    base_config["env"]["sumo"]["max_sim_seconds"] = 600

    save_yaml_config(base_config, str(test_config))
    print(f"Created test config: {test_config}")
    print(f"Route pool size: {len(base_config['train']['route_pool'])}")

    print()
    print("STEP 3: Run training for 10 episodes")
    result = run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--config",
            str(test_config),
            "--episodes",
            "10",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("\n[WARNING] Training encountered errors")
        print("This may be due to SUMO not being installed or network issues")

    print()
    print("STEP 4: Verify route selection diversity")

    routes_used = extract_route_selections(result.stdout)

    if len(routes_used) == 0:
        print("\n[ERROR] No route selection logs found")
        print("Check if SUMOEnv.reset() is printing route selection")
        sys.exit(1)

    print(f"\nFound {len(routes_used)} route selections:")
    route_counts = Counter(routes_used)
    for route, count in route_counts.most_common():
        print(f"  - {Path(route).name}: {count} time(s)")

    unique_routes = len(route_counts)

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if unique_routes < 2:
        print("\n[FAIL] Only 1 unique route was used")
        print("Route pool randomization is not working correctly")
        sys.exit(1)

    print(f"\n[PASS] {unique_routes} different routes were used")
    print(f"Total selections: {len(routes_used)}")

    print("\nCleaning up test files...")
    test_config.unlink()
    print(f"Removed {test_config}")


if __name__ == "__main__":
    main()
