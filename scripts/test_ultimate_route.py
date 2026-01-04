from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def parse_route_file(route_path: Path) -> Tuple[List[Tuple[float, float, str]], int]:
    if not route_path.exists():
        raise FileNotFoundError(f"Route file not found: {route_path}")
    
    tree = ET.parse(route_path)
    root = tree.getroot()
    
    flows: List[Tuple[float, float, str]] = []
    
    for flow_elem in root.findall("flow"):
        flow_id = flow_elem.get("id", "unknown")
        begin = float(flow_elem.get("begin", 0))
        end = float(flow_elem.get("end", 0))
        flows.append((begin, end, flow_id))
    
    total_flows = len(flows)
    return flows, total_flows


def check_stage_coverage(flows: List[Tuple[float, float, str]]) -> Dict[str, int]:
    stages = {
        "stage_1_low": (0, 1920),
        "stage_2_high": (1920, 3840),
        "stage_3_low": (3840, 5760),
        "stage_4_high": (5760, 7680),
    }
    
    coverage: Dict[str, int] = {stage: 0 for stage in stages}
    
    for begin, end, flow_id in flows:
        for stage_name, (stage_begin, stage_end) in stages.items():
            if begin < stage_end and end > stage_begin:
                overlap = min(end, stage_end) - max(begin, stage_begin)
                if overlap > 0:
                    coverage[stage_name] += 1
    
    return coverage


def main() -> None:
    route_file = Path("networks/BI_ultimate_clean.rou.xml")
    
    print("=" * 80)
    print("ULTIMATE SCENARIO ROUTE VALIDATION")
    print("=" * 80)
    
    if not route_file.exists():
        print(f"\nERROR: Route file not found: {route_file}")
        sys.exit(1)
    
    print(f"\nValidating: {route_file}")
    
    try:
        flows, total_count = parse_route_file(route_file)
    except Exception as exc:
        print(f"\nERROR: Failed to parse route file: {exc}")
        sys.exit(1)
    
    print(f"Total flows defined: {total_count}")
    
    coverage = check_stage_coverage(flows)
    
    print("\nStage Coverage:")
    print(f"  Stage 1 (0-1920s, Low):       {coverage['stage_1_low']} flows")
    print(f"  Stage 2 (1920-3840s, High):   {coverage['stage_2_high']} flows")
    print(f"  Stage 3 (3840-5760s, Low):    {coverage['stage_3_low']} flows")
    print(f"  Stage 4 (5760-7680s, High):   {coverage['stage_4_high']} flows")
    
    all_stages_covered = all(count > 0 for count in coverage.values())
    
    if all_stages_covered:
        print("\nSTATUS: PASS - All stages have flow definitions")
        print("\n" + "=" * 80)
        sys.exit(0)
    else:
        missing = [stage for stage, count in coverage.items() if count == 0]
        print(f"\nSTATUS: FAIL - Missing flows in stages: {missing}")
        print("\n" + "=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()