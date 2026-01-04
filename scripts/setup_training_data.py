from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config, save_yaml_config


def main() -> None:
    print("=" * 80)
    print("SETTING UP HANOI-REALISTIC TRAINING DATA")
    print("=" * 80)

    net_file = repo_root / "networks" / "BIGMAP.net.xml"
    variants_dir = repo_root / "networks" / "train_variants"
    config_path = repo_root / "configs" / "train_sumo.yaml"

    if not net_file.exists():
        sys.exit(1)

    variants_dir.mkdir(parents=True, exist_ok=True)

    for f in variants_dir.glob("*.rou.xml"):
        f.unlink()

    num_variants = 50

    print(f"Generating {num_variants} scenarios based on Hanoi traffic density...")
    print("  - Base Flow: ~2000 vehicles/hour/lane (Peak saturation)")
    print("  - Mix: 85% Moto / 12% Car / 3% Bus")

    for i in range(num_variants):
        volume_scale = 0.4 + (float(i) / float(num_variants)) * 0.7
        output_file = variants_dir / f"hanoi_route_seed{i}_load{volume_scale:.2f}.rou.xml"

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "generate_jtr_data.py"),
            "--net-file",
            str(net_file),
            "--output-route",
            str(output_file),
            "--seed",
            str(int(i)),
            "--volume-scale",
            str(float(volume_scale)),
        ]

        try:
            subprocess.run(cmd, check=True)
            if i % 5 == 0:
                print(f"  > Generated scenario {i}: Load factor {volume_scale:.2f}")
        except subprocess.CalledProcessError:
            print(f"  [FAIL] Variant {i}")

    generated_files = sorted(list(variants_dir.glob("*.rou.xml")))

    config = load_yaml_config(str(config_path))

    if "train" not in config:
        config["train"] = {}

    route_pool_rel = [str(p.relative_to(repo_root)).replace("\\", "/") for p in generated_files]
    config["train"]["route_pool"] = route_pool_rel

    if "env" in config and "sumo" in config["env"]:
        config["env"]["sumo"]["vehicle_weights"] = {
            "motorcycle": 0.25,
            "passenger": 1.0,
            "bus": 3.0,
        }
        config["env"]["sumo"]["use_pcu_weighted_wait"] = True

    save_yaml_config(config, str(config_path))
    print("Config updated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
