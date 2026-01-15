"""
Generate SUMO flow files with updated vehicle distribution (80% motorcycle, 18% car, 2% bus)
This creates flow-only route files that SUMO can use directly.
"""
from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

def get_source_edges_info(net_file: Path) -> Dict[str, int]:
    """Extract source edges and their lane counts from network file."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    all_edges: Dict[str, int] = {}
    incoming = set()
    
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if not eid or eid.startswith(":"):
            continue
        lanes = edge.findall("lane")
        if lanes:
            all_edges[eid] = len(lanes)
    
    for conn in root.findall("connection"):
        to_edge = conn.get("to")
        if to_edge in all_edges:
            incoming.add(to_edge)
    
    source_ids = [e for e in all_edges if e not in incoming]
    sources_info = {eid: all_edges[eid] for eid in source_ids}
    
    return sources_info


def generate_flows_xml(
    output_path: Path,
    sources_info: Dict[str, int],
    duration: int,
    global_scale: float,
    base_flow: float,
    seed: int
) -> None:
    """Generate flow-based route file with updated vehicle distribution."""
    random.seed(seed)
    
    root = ET.Element("routes")
    
    # Vehicle types
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
    
    # Updated vehicle distribution
    veh_distribution = {
        "motorcycle": 0.80,   # 80% xe máy
        "passenger": 0.18,    # 18% ô tô con
        "bus": 0.02,          # 2% bus
    }
    
    for edge_id, num_lanes in sources_info.items():
        base_edge_flow = float(num_lanes) * float(base_flow)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SUMO flow files with updated vehicle distribution"
    )
    parser.add_argument("--net-file", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--output-route", required=True, help="Output route file (.rou.xml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--volume-scale", type=float, default=1.0, help="Demand scaling factor")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration (seconds)")
    parser.add_argument("--base-flow", type=float, default=2000, help="Base flow per lane in veh/hr")
    args = parser.parse_args()
    
    net_path = Path(args.net_file)
    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")
    
    out_path = Path(args.output_route)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sources_info = get_source_edges_info(net_path)
    
    if len(sources_info) == 0:
        sys.exit(f"No source edges found in network: {net_path}")
    
    volume_scale = max(0.1, min(2.0, float(args.volume_scale)))
    duration = max(60, int(args.duration))
    base_flow = max(100.0, float(args.base_flow))
    
    print(f"Generating route file: {out_path.name}")
    print(f"  Sources: {len(sources_info)}")
    print(f"  Base flow: {base_flow:.0f} veh/hr/lane")
    print(f"  Vehicle mix: 80% motorcycle, 18% car, 2% bus")
    
    generate_flows_xml(out_path, sources_info, duration, volume_scale, base_flow, args.seed)
    
    if out_path.exists():
        file_size_kb = out_path.stat().st_size / 1024.0
        print(f"OK Created: {out_path} ({file_size_kb:.1f} KB)")
    else:
        sys.exit(f"Failed to create route file: {out_path}")


if __name__ == "__main__":
    main()
