#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def step(description: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)


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


def main() -> None:
    print("=" * 80)
    print("HANOI TRAINING SETUP - AUTOMATED PIPELINE")
    print("=" * 80)
    print()

    net_file = repo_root / "networks" / "BIGMAP.net.xml"
    calib_path = repo_root / "configs" / "scenario_hanoi_calibration.yaml"
    train_variants_dir = repo_root / "networks" / "variants" / "train"
    eval_variants_dir = repo_root / "networks" / "variants" / "eval"
    config_path = repo_root / "configs" / "train_sumo.yaml"

    if not net_file.exists():
        print(f"[ERROR] Network file not found: {net_file}")
        print()
        print("Please ensure BIGMAP.net.xml is in networks/ directory")
        sys.exit(1)

    if not calib_path.exists():
        step("Generate calibration file (auto-detect boundaries)")
        run_command(
            [
                sys.executable,
                str(repo_root / "scripts" / "inspect_net_boundaries.py"),
                "--net",
                str(net_file),
                "--out",
                str(calib_path),
            ]
        )

        print()
        print("[INFO] Please review and adjust calibration file:")
        print(f"  {calib_path}")
        print()
        print("Especially:")
        print("  - entry_edges / exit_edges (verify they match your network)")
        print("  - demand.total_pcu_per_hour (adjust based on real data)")
        print()
        response = input("Continue with generated calibration? [y/N]: ")
        if response.lower() != "y":
            print("Aborting. Please edit calibration file and run again.")
            sys.exit(0)
    else:
        print(f"[INFO] Using existing calibration: {calib_path}")

    step("Clean old route variants")
    for variant_dir in [train_variants_dir, eval_variants_dir]:
        if variant_dir.exists():
            import shutil

            shutil.rmtree(variant_dir)
            print(f"  Cleaned: {variant_dir}")
        variant_dir.mkdir(parents=True, exist_ok=True)

    step("Generate TRAINING variants (100 seeds)")
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_hanoi_route_variants.py"),
            "--calib",
            str(calib_path),
            "--out-dir",
            str(train_variants_dir),
            "--split",
            "train",
            "--n",
            "100",
            "--seed",
            "42",
            "--level",
            "med",
        ]
    )

    step("Generate EVALUATION variants (30 seeds)")
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_hanoi_route_variants.py"),
            "--calib",
            str(calib_path),
            "--out-dir",
            str(eval_variants_dir),
            "--split",
            "eval",
            "--n",
            "30",
            "--seed",
            "42",
            "--level",
            "med",
        ]
    )

    step("Update training config with route_pool")

    train_routes = sorted(list(train_variants_dir.glob("BIGMAP_variant_seed*.rou.xml")))
    if len(train_routes) == 0:
        print("[ERROR] No training routes generated")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    route_pool = [str(Path(rf).relative_to(repo_root)).replace("\\", "/") for rf in train_routes]

    config.setdefault("train", {})
    config["train"]["route_pool"] = route_pool

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"  Updated route_pool with {len(route_pool)} variants")

    step("Run validation tests")

    print("\nTEST 1: Scenario generator validation")
    result = run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "test_scenario_generator.py"),
        ],
        check=False,
    )

    if result.returncode != 0:
        print("\n[WARNING] Scenario validation failed")
        print("Review errors above, but setup will continue")

    print("\nTEST 2: Route pool functionality")
    result = run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "test_route_pool.py"),
        ],
        check=False,
    )

    if result.returncode != 0:
        print("\n[WARNING] Route pool test failed")
        print("This might be due to SUMO not being installed")
        print("You can still proceed with training if SUMO is available")

    step("SETUP COMPLETE")

    print()
    print("Training variants: " + str(train_variants_dir.relative_to(repo_root)))
    print(f"  Count: {len(train_routes)}")
    print()
    print("Evaluation variants: " + str(eval_variants_dir.relative_to(repo_root)))
    eval_routes = list(eval_variants_dir.glob("BIGMAP_variant_seed*.rou.xml"))
    print(f"  Count: {len(eval_routes)}")
    print()
    print("Training config updated: " + str(config_path.relative_to(repo_root)))
    print(f"  route_pool size: {len(route_pool)}")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Start training with random routes:")
    print(f"   python scripts/train.py --config {config_path.relative_to(repo_root)} --episodes 300")
    print()
    print("2. Monitor training logs to verify route selection:")
    print("   tail -f logs/*.csv")
    print("   grep 'Using route' logs/*.log")
    print()
    print("3. Evaluate trained model on held-out scenarios:")
    print("   python scripts/eval.py \\")
    print("     --config configs/eval_fixed_scenario.yaml \\")
    print("     --controller all \\")
    print("     --model-path models/YOURMODEL.pt \\")
    print("     --runs 30")
    print()
    print("Note: The eval script will need to be configured to use")
    print("      the generated eval variants from networks/variants/eval/")
    print()


if __name__ == "__main__":
    main()
