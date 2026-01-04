from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET


def _parse_flows(path: Path) -> List[Tuple[float, float, str]]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    flows = []
    for elem in root.iter():
        if elem.tag != "flow":
            continue
        begin = float(elem.get("begin", 0.0))
        end = float(elem.get("end", begin))
        flows.append((begin, end, elem.get("id", "")))
    return flows


def _verify(flows: List[Tuple[float, float, str]], stages: List[Tuple[float, float]]) -> Tuple[List[str], List[str]]:
    issues = []
    warns = []
    for idx, (start, stop) in enumerate(stages):
        overlapping = [fid for begin, end, fid in flows if begin < stop and end > start]
        if len(overlapping) == 0:
            issues.append(f"Stage {idx} [{start},{stop}] has no flows overlapping the interval.")
        exact = [fid for begin, end, fid in flows if begin == start and end == stop]
        if len(exact) == 0:
            warns.append(f"[WARN] Stage {idx} [{start},{stop}] has no flow exactly matching bounds.")
    return issues, warns


def main(argv: List[str] | None = None) -> None:
    arg_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Verify ultimate scenario route stages exist in a SUMO .rou.xml file.")
    parser.add_argument("--route-file", default="networks/BI_ultimate_clean.rou.xml", help="Route file to verify.")
    parser.add_argument("--stages", nargs="+", type=float, default=[0, 1920, 3840, 5760, 7680], help="Stage boundaries (inclusive, increasing).")
    args = parser.parse_args(arg_list)

    route_path = Path(args.route_file)
    user_provided_route = any(opt.startswith("--route-file") or opt == "--route-file" for opt in arg_list)
    if not route_path.exists():
        if user_provided_route:
            sys.exit(f"Route file not found: {route_path}")
        print(f"[WARN] Default route file not found: {route_path}. Skipping verification.")
        return

    if len(args.stages) < 2:
        sys.exit("Must provide at least two stage boundaries.")
    boundaries = args.stages
    stages = list(zip(boundaries[:-1], boundaries[1:]))

    try:
        flows = _parse_flows(route_path)
    except Exception as exc:
        sys.exit(f"Failed to parse route file: {exc}")

    issues, warns = _verify(flows, stages)
    for msg in warns:
        print(msg)
    if len(issues) == 0:
        print("PASS: All stages contain flows.")
        return

    print("FAIL:")
    for msg in issues:
        print(f" - {msg}")
    sys.exit(1)


if __name__ == "__main__":
    main()
