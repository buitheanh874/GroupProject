from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.repo_root import find_repo_root


def find_binary(name: str, sumo_home: Optional[str]) -> Optional[str]:
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return str(found)
    return None


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"{label} not found: {path}")


def validate_route_file_extension(route_file_str: str, repo_root: Path) -> None:
    """Validate that route_file is a valid XML route file, not a manifest."""
    route_lower = route_file_str.lower().strip()
    
    if route_lower.endswith(".txt"):
        print(f"ERROR: sumo.route_file points to a .txt manifest: {route_file_str}")
        print("SUMO requires a .rou.xml or .xml route file, not a text manifest.")
        print("If you want to use a route pool, set 'train.route_pool_manifest' or 'eval.route_pool_manifest'")
        print("and let the loader resolve it to a valid route file.")
        sys.exit(1)
    
    if route_lower == "." or route_lower == "":
        print(f"ERROR: sumo.route_file is invalid: '{route_file_str}'")
        print("Please provide a valid .rou.xml or .xml route file path.")
        sys.exit(1)
    
    if not (route_lower.endswith(".rou.xml") or route_lower.endswith(".xml")):
        print(f"WARNING: sumo.route_file has unexpected extension: {route_file_str}")
        print("Expected .rou.xml or .xml extension for SUMO route files.")


def validate_config(config: Dict[str, Any], repo_root: Path) -> bool:
    """Validate config paths and settings. Returns True if all checks pass."""
    sumo_cfg = config.get("env", {}).get("sumo", {})
    errors = []
    warnings = []
    
    net_file = sumo_cfg.get("net_file", "")
    if net_file:
        net_path = Path(net_file)
        if not net_path.is_absolute():
            net_path = repo_root / net_file
        if not net_path.exists():
            errors.append(f"Network file not found: {net_path}")
        else:
            print(f"[OK] Network file exists: {net_path}")
    else:
        errors.append("env.sumo.net_file not specified")
    
    route_file = sumo_cfg.get("route_file", "")
    if route_file:
        route_lower = str(route_file).lower().strip()
        
        if route_lower.endswith(".txt"):
            errors.append(
                f"sumo.route_file points to a .txt manifest: {route_file}\n"
                "  SUMO requires .rou.xml or .xml, not text manifest.\n"
                "  Use train.route_pool_manifest or eval.route_pool_manifest for route pools."
            )
        elif route_lower == "." or route_lower == "":
            errors.append(f"sumo.route_file is invalid: '{route_file}'")
        else:
            route_path = Path(route_file)
            if not route_path.is_absolute():
                route_path = repo_root / route_file
            if not route_path.exists():
                errors.append(f"Route file not found: {route_path}")
            else:
                if route_lower.endswith(".rou.xml") or route_lower.endswith(".xml"):
                    print(f"[OK] Route file exists (XML): {route_path}")
                else:
                    warnings.append(f"Route file has unexpected extension: {route_file}")
                    print(f"[OK] Route file exists: {route_path}")
    else:
        errors.append("env.sumo.route_file not specified")
    
    for split in ["train", "eval"]:
        manifest = config.get(split, {}).get("route_pool_manifest")
        if manifest:
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = repo_root / manifest
            if manifest_path.exists():
                print(f"[OK] {split}.route_pool_manifest exists: {manifest_path}")
            else:
                warnings.append(f"{split}.route_pool_manifest not found: {manifest_path}")
    
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")
    
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  [ERR] {e}")
        return False
    
    return True


def main() -> None:
    repo_root = find_repo_root(__file__)
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Doctor check for SUMO/TraCI environment and config validation.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file for validation")
    parser.add_argument("--net-file", default="networks/BIGNET.net.xml")
    parser.add_argument("--route-file", default="networks/variants/train/bignet_train_seed00042.rou.xml")
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-traci", action="store_true", help="Skip TraCI connection test")
    args = parser.parse_args()

    sumo_home = os.getenv("SUMO_HOME")

    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"SUMO_HOME: {sumo_home if sumo_home else '(not set)'}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Repo root: {repo_root}")

    sumo_bin = find_binary("sumo", sumo_home)
    sumo_gui_bin = find_binary("sumo-gui", sumo_home)

    if not sumo_bin:
        sys.exit("Missing sumo binary. Install SUMO or set SUMO_HOME.")

    print(f"sumo: {sumo_bin}")
    print(f"sumo-gui: {sumo_gui_bin if sumo_gui_bin else '(not found)'}")

    try:
        import traci
        print("traci: import OK")
    except Exception as exc:
        sys.exit(f"Failed to import traci: {exc}")

    if args.config:
        print(f"\n--- Config Validation: {args.config} ---")
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / args.config
        
        if not config_path.exists():
            sys.exit(f"Config file not found: {config_path}")
        
        from rl.utils import load_yaml_config
        config = load_yaml_config(str(config_path))
        
        if not validate_config(config, repo_root):
            sys.exit(1)
        
        sumo_cfg = config.get("env", {}).get("sumo", {})
        net_file = sumo_cfg.get("net_file", args.net_file)
        route_file = sumo_cfg.get("route_file", args.route_file)
        net_path = Path(net_file) if Path(net_file).is_absolute() else repo_root / net_file
        route_path = Path(route_file) if Path(route_file).is_absolute() else repo_root / route_file
        
        print(f"\nConfig validation: PASSED")
    else:
        net_path = repo_root / args.net_file
        route_path = repo_root / args.route_file
        require_path(net_path, "Network file")
        require_path(route_path, "Route file")

    if args.skip_traci:
        print("\nTraCI test: SKIPPED (--skip-traci)")
        print("Status: OK (config only)")
        return

    command = [
        sumo_bin,
        "-n",
        str(net_path),
        "-r",
        str(route_path),
        "--step-length",
        str(float(args.step_length)),
        "--seed",
        str(int(args.seed)),
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
    ]

    try:
        traci.start(command)
        traci.simulationStep()
    except Exception as exc:
        try:
            traci.close(False)
        except Exception:
            pass
        sys.exit(f"TraCI start failed: {exc}")
    else:
        print("TraCI: connected")
    finally:
        try:
            traci.close(False)
        except Exception:
            pass

    print("Status: OK")


if __name__ == "__main__":
    main()
