from __future__ import annotations

import argparse
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

HANOI_BASE_FLOW_PER_LANE = 2000.0


def get_source_edges_info(net_file: Path) -> Tuple[Dict[str, int], List[str]]:
    tree = ET.parse(net_file)
    root = tree.getroot()

    all_edges: Dict[str, int] = {}
    incoming = set()
    outgoing = set()

    for edge in root.findall("edge"):
        eid = edge.get("id")
        if not eid:
            continue
        if eid.startswith(":"):
            continue

        lanes = edge.findall("lane")
        if lanes:
            all_edges[eid] = len(lanes)

    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")

        if from_edge in all_edges:
            outgoing.add(from_edge)
        if to_edge in all_edges:
            incoming.add(to_edge)

    source_ids = [e for e in all_edges if e not in incoming]
    sink_ids = [e for e in all_edges if e not in outgoing]
    sources_info = {eid: all_edges[eid] for eid in source_ids}

    return sources_info, sorted(sink_ids)


def generate_flows_xml(output_path: Path, sources_info: Dict[str, int], duration: int, global_scale: float) -> None:
    root = ET.Element("routes")

    vtypes = [
        {
            "id": "motorcycle",
            "vClass": "motorcycle",
            "length": "2.0",
            "width": "0.8",
            "maxSpeed": "13.89",
            "accel": "3.5",
            "decel": "4.0",
            "latAlignment": "right",
            "sigma": "0.8",
            "minGap": "0.5",
        },
        {
            "id": "passenger",
            "vClass": "passenger",
            "length": "4.5",
            "width": "1.8",
            "maxSpeed": "13.89",
            "accel": "2.5",
            "decel": "4.5",
            "sigma": "0.3",
            "minGap": "2.0",
        },
        {
            "id": "bus",
            "vClass": "bus",
            "length": "12.0",
            "width": "2.5",
            "maxSpeed": "10.0",
            "accel": "1.2",
            "decel": "2.5",
            "sigma": "0.1",
            "minGap": "2.5",
        },
    ]

    for vt in vtypes:
        ET.SubElement(root, "vType", **vt)

    veh_distribution = {
        "motorcycle": 0.85,
        "passenger": 0.12,
        "bus": 0.03,
    }

    for edge_id, num_lanes in sources_info.items():
        base_edge_flow = float(num_lanes) * float(HANOI_BASE_FLOW_PER_LANE)
        edge_noise = random.uniform(0.6, 1.1)
        total_edge_flow = base_edge_flow * float(global_scale) * float(edge_noise)

        for v_type, ratio in veh_distribution.items():
            flow_rate = float(total_edge_flow) * float(ratio)

            if flow_rate > 1.0:
                flow = ET.SubElement(root, "flow")
                flow.set("id", f"f_{edge_id}_{v_type}")
                flow.set("from", str(edge_id))
                flow.set("begin", "0")
                flow.set("end", str(int(duration)))
                flow.set("vehsPerHour", f"{flow_rate:.2f}")
                flow.set("type", str(v_type))
                flow.set("departLane", "best")
                flow.set("departSpeed", "max")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def generate_turnfile_xml(output_path: Path, sinks: List[str], duration: int) -> None:
    root = ET.Element("turns")
    interval = ET.SubElement(root, "interval")
    interval.set("begin", "0")
    interval.set("end", str(int(duration)))

    for sink_id in sinks:
        from_edge = ET.SubElement(interval, "fromEdge")
        from_edge.set("id", str(sink_id))
        
        to_edge = ET.SubElement(from_edge, "toEdge")
        to_edge.set("id", str(sink_id))
        to_edge.set("probability", "1.0")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hanoi-realistic route variants for SUMO network"
    )
    parser.add_argument("--net-file", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--output-route", required=True, help="Output route file (.rou.xml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--volume-scale", type=float, default=1.0, help="Demand scaling factor (0.0-2.0)")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration (seconds)")
    args = parser.parse_args()

    random.seed(int(args.seed))

    net_path = Path(args.net_file)
    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")

    out_path = Path(args.output_route)
    temp_dir = out_path.parent / "temp_jtr"
    temp_dir.mkdir(parents=True, exist_ok=True)

    flow_file = temp_dir / f"flows_{int(args.seed)}.xml"
    turn_file = temp_dir / f"turns_{int(args.seed)}.xml"

    sources_info, sinks = get_source_edges_info(net_path)

    if len(sources_info) == 0:
        sys.exit(f"No source edges found in network: {net_path}")

    if len(sinks) == 0:
        sys.exit(f"No sink edges found in network: {net_path}")

    volume_scale = max(0.1, min(2.0, float(args.volume_scale)))
    duration = max(60, int(args.duration))

    print(f"Generating route file: {out_path.name}")
    print(f"  Network: {net_path.name}")
    print(f"  Sources: {len(sources_info)}")
    print(f"  Sinks: {len(sinks)}")
    print(f"  Volume scale: {volume_scale:.2f}")
    print(f"  Duration: {duration}s")

    generate_flows_xml(flow_file, sources_info, duration, volume_scale)
    generate_turnfile_xml(turn_file, sinks, duration)

    cmd = [
        "python",
        "-m",
        "sumolib.tools.generateRoutes",
        "--net-file",
        str(net_path),
        "--flow-file",
        str(flow_file),
        "--turn-file",
        str(turn_file),
        "--output",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"Warning: Route generation had issues: {result.stderr}")
    except Exception as exc:
        print(f"Warning: Could not run SUMO route generation: {exc}")
        print("Creating minimal fallback route file...")
        generate_flows_xml(out_path, sources_info, duration, volume_scale)

    if out_path.exists():
        file_size_kb = out_path.stat().st_size / 1024.0
        print(f"Route file created: {out_path} ({file_size_kb:.1f} KB)")
    else:
        sys.exit(f"Failed to create route file: {out_path}")

    print("Done")


if __name__ == "__main__":
    main()