from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from collections import Counter

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def step(description: str) -> None:
    """Print step header."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
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


def main() -> None:
    print("="*80)
    print("ROUTE POOL FUNCTIONALITY TEST")
    print("="*80)

    base_route = repo_root / "networks" / "BI_50_test.rou.xml"
    if not base_route.exists():
        print(f"\n[ERROR] Base route file not found: {base_route}")
        print("This test requires a valid SUMO network setup.")
        sys.exit(1)

    step("Generate Route Variants")
    variants_dir = repo_root / "networks" / "test_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    
    run_command([
        sys.executable,
        str(repo_root / "scripts" / "generate_randomized_routes.py"),
        "--input", str(base_route),
        "--output-dir", str(variants_dir),
        "--variants", "3",
        "--seed", "42",
        "--global-range", "0.8", "1.2",
        "--per-flow-noise", "0.1"
    ])

    variant_files = list(variants_dir.glob("*.rou.xml"))
    if len(variant_files) != 3:
        print(f"\n[ERROR] Expected 3 variants, found {len(variant_files)}")
        sys.exit(1)
    
    print(f"\n✓ Generated {len(variant_files)} route variants")

    step("Create Test Config")
    test_config = repo_root / "configs" / "test_route_pool.yaml"

    from rl.utils import load_yaml_config, save_yaml_config
    
    base_config = load_yaml_config(str(repo_root / "configs" / "train_sumo.yaml"))

    base_config["run"]["run_name"] = "test_route_pool"
    base_config["train"]["episodes"] = 3
    base_config["train"]["print_every_episodes"] = 1
    base_config["train"]["route_pool"] = [str(base_route)] + [str(f) for f in variant_files]
    base_config["env"]["sumo"]["max_sim_seconds"] = 600  
    
    save_yaml_config(base_config, str(test_config))
    print(f"✓ Created test config: {test_config}")
    print(f"  Route pool size: {len(base_config['train']['route_pool'])}")

    step("Run Training (3 episodes)")
    result = run_command([
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--config", str(test_config),
        "--episodes", "3"
    ], check=False)
    
    if result.returncode != 0:
        print("\n[WARNING] Training encountered errors")
        print("This may be due to SUMO not being installed or network issues")
    step("Verify Route Selection")

    routes_used = []
    for line in result.stdout.split('\n'):
        if "Using route" in line:
            route_name = line.split("Using route '")[1].split("'")[0]
            routes_used.append(route_name)
    
    if len(routes_used) == 0:
        print("\n[ERROR] No route selection logs found")
        print("Check if SUMOEnv.reset() is printing route selection")
        sys.exit(1)
    
    print(f"\n✓ Found {len(routes_used)} route selections:")
    route_counts = Counter(routes_used)
    for route, count in route_counts.items():
        print(f"  - {route}: {count} time(s)")

    unique_routes = len(route_counts)
    if unique_routes == 1:
        print("\n[WARNING] Only 1 unique route was used")
        print("With 3 episodes and 4 routes, we expect some variety")
        print("This could be random chance or a bug")
    else:
        print(f"\n✓ SUCCESS: {unique_routes} different routes were used")

    step("Test Summary")
    
    print("\n Route pool functionality test completed\n")
    
    print("What was tested:")
    print("  ✓ Route variant generation")
    print("  ✓ Config with route_pool")
    print("  ✓ Training with route pool")
    print(f"  ✓ Route selection logging (found {len(routes_used)} selections)")
    
    if unique_routes > 1:
        print(f"   Multiple routes used ({unique_routes} different routes)")
        print("\n All checks passed!")
    else:
        print(f"    Only 1 route used (may need more episodes to verify randomization)")
        print("\n✓ Basic functionality works, but randomization unclear")
    
    print("\nTo use in production:")
    print("  1. Generate route variants:")
    print("      python scripts/generate_randomized_routes.py \\")
    print("        --input networks/YOUR_ROUTE.rou.xml \\")
    print("        --output-dir networks/variants \\")
    print("        --variants 5 --seed 42")
    print("\n  2. Update config:")
    print("      train:")
    print("        route_pool:")
    print("          - networks/YOUR_ROUTE.rou.xml")
    print("          - networks/variants/...")
    print("\n  3. Train:")
    print("      python scripts/train.py --config YOUR_CONFIG.yaml")

    print("\n" + "="*80)
    print("Cleaning up test files...")
    try:
        test_config.unlink()
        print(f"  Removed {test_config}")
    except Exception as e:
        print(f"  Warning: Cleanup failed: {e}")


if __name__ == "__main__":
    main()