from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", default="networks/BI.net.xml")
    parser.add_argument("--route-file", default="networks/BI_50_test.rou.xml")
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sumo_home = os.getenv("SUMO_HOME")

    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"SUMO_HOME: {sumo_home if sumo_home else '(not set)'}")

    sumo_bin = find_binary("sumo", sumo_home)
    sumo_gui_bin = find_binary("sumo-gui", sumo_home)

    if not sumo_bin:
        sys.exit("Missing sumo binary. Install SUMO or set SUMO_HOME.")

    print(f"sumo: {sumo_bin}")
    print(f"sumo-gui: {sumo_gui_bin if sumo_gui_bin else '(not found)'}")

    try:
        import traci
    except Exception as exc:
        sys.exit(f"Failed to import traci: {exc}")

    net_path = repo_root / args.net_file
    route_path = repo_root / args.route_file

    require_path(net_path, "Network file")
    require_path(route_path, "Route file")

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
