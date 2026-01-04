from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    
    calib_path = repo_root / "configs" / "scenario_hanoi_calibration.yaml"
    train_variants_dir = repo_root / "networks" / "variants" / "train"
    eval_variants_dir = repo_root / "networks" / "variants" / "eval"
    config_path = repo_root / "configs" / "train_sumo_random.yaml"

    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        print()
        print("Create it first:")
        print(f"  python scripts/inspect_net_boundaries.py \\")
        print(f"    --net networks/BIGMAP.net.xml \\")
        print(f"    --out {calib_path}")
        sys.exit(1)

    print("=" * 80)
    print("HANOI TRAINING SETUP - RANDOM ROUTE VARIANTS")
    print("=" * 80)
    print()

    print("[STEP 1] Clean old variants")
    for variant_dir in [train_variants_dir, eval_variants_dir]:
        if variant_dir.exists():
            import shutil
            shutil.rmtree(variant_dir)
            print(f"  Cleaned: {variant_dir}")
    
    print()
    print("[STEP 2] Generate TRAINING variants (100 seeds)")
    cmd_train = [
        sys.executable,
        str(repo_root / "scripts" / "generate_hanoi_route_variants.py"),
        "--calib", str(calib_path),
        "--out-dir", str(train_variants_dir),
        "--split", "train",
        "--n", "100",
        "--seed", "42",
        "--level", "med",
    ]
    
    result = subprocess.run(cmd_train, cwd=repo_root)
    if result.returncode != 0:
        sys.exit("Failed to generate training variants")
    
    print()
    print("[STEP 3] Generate EVALUATION variants (30 seeds)")
    cmd_eval = [
        sys.executable,
        str(repo_root / "scripts" / "generate_hanoi_route_variants.py"),
        "--calib", str(calib_path),
        "--out-dir", str(eval_variants_dir),
        "--split", "eval",
        "--n", "30",
        "--seed", "42",
        "--level", "med",
    ]
    
    result = subprocess.run(cmd_eval, cwd=repo_root)
    if result.returncode != 0:
        sys.exit("Failed to generate evaluation variants")
    
    print()
    print("[STEP 4] Update training config with route pool")
    
    train_routes = sorted(list(train_variants_dir.glob("BIGMAP_variant_seed*.rou.xml")))
    
    if len(train_routes) == 0:
        sys.exit("No training routes generated")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    route_pool = [str(Path(rf).relative_to(repo_root)).replace("\\", "/") for rf in train_routes]
    
    config.setdefault("train", {})
    config["train"]["route_pool"] = route_pool

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    
    print(f"  Updated route_pool with {len(route_pool)} variants")
    
    print()
    print("=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print()
    print("Training variants: " + str(train_variants_dir.relative_to(repo_root)))
    print(f"  Count: {len(train_routes)}")
    print()
    print("Evaluation variants: " + str(eval_variants_dir.relative_to(repo_root)))
    print(f"  Count: {len(list(eval_variants_dir.glob('BIGMAP_variant_seed*.rou.xml')))}")
    print()
    print("Next: Run training with random routes")
    print(f"  python scripts/train.py --config {config_path.relative_to(repo_root)} --episodes 300")
    print()
    print("Evaluate with fixed scenario:")
    print(f"  python scripts/eval.py --config configs/eval_fixed_scenario.yaml \\")
    print(f"    --controller all \\")
    print(f"    --model-path models/train_hanoi_random/train_sumo_hanoi_random_best.pt \\")
    print(f"    --runs 10")
    print()


if __name__ == "__main__":
    main()