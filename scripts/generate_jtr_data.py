from __future__ import annotations

import argparse
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

def get_edges(net_file: Path) -> Tuple[List[str], List[str]]:
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    all_edges = set()
    incoming = set()
    outgoing = set()
    
    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        
        if from_edge and not from_edge.startswith(":"):
            all_edges.add(from_edge)
            outgoing.add(from_edge)
            
        if to_edge and not to_edge.startswith(":"):
            all_edges.add(to_edge)
            incoming.add(to_edge)
            
    sources = sorted(list(all_edges - incoming))
    sinks = sorted(list(all_edges - outgoing))
    
    return sources, sinks

def generate_flows_xml(output_path: Path, sources: List[str], duration: int, base_volume: float) -> None:
    root = ET.Element("routes")
    
    vtypes = [
        {
            'id': 'motorcycle', 
            'vClass': 'motorcycle', 
            'length': '2.0', 
            'width': '0.8', 
            'maxSpeed': '13.89', 
            'accel': '3.0', 
            'decel': '4.0', 
            'latAlignment': 'right', 
            'sigma': '0.7',
            'minGap': '1.0'
        },
        {
            'id': 'passenger', 
            'vClass': 'passenger', 
            'length': '4.5', 
            'width': '1.8', 
            'maxSpeed': '13.89', 
            'accel': '2.5', 
            'decel': '4.5', 
            'sigma': '0.5',
            'minGap': '2.5'
        },
        {
            'id': 'bus', 
            'vClass': 'bus', 
            'length': '12.0', 
            'width': '2.5', 
            'maxSpeed': '10.0', 
            'accel': '1.2', 
            'decel': '2.5', 
            'sigma': '0.3',
            'minGap': '3.0'
        }
    ]
    
    for vt in vtypes:
        ET.SubElement(root, "vType", **vt)
        
    veh_distribution = {
        "motorcycle": 0.85,
        "passenger": 0.12,
        "bus": 0.03
    }
    
    for edge in sources:
        edge_noise = random.uniform(0.8, 1.2)
        total_edge_flow = base_volume * edge_noise
        
        for v_type, ratio in veh_distribution.items():
            flow_rate = total_edge_flow * ratio
            
            flow = ET.SubElement(root, "flow")
            flow.set("id", f"flow_{edge}_{v_type}")
            flow.set("from", edge)
            flow.set("begin", "0")
            flow.set("end", str(duration))
            flow.set("vehsPerHour", f"{flow_rate:.2f}")
            flow.set("type", v_type)
            flow.set("departLane", "best")
            flow.set("departSpeed", "max")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

def generate_sinks_xml(output_path: Path, sinks: List[str], duration: int) -> None:
    root = ET.Element("turns")
    interval = ET.SubElement(root, "interval")
    interval.set("begin", "0")
    interval.set("end", str(duration))
    
    for s in sinks:
        sink_elem = ET.SubElement(interval, "fromEdge")
        sink_elem.set("id", s)
        to_elem = ET.SubElement(sink_elem, "toEdge")
        to_elem.set("id", s)
        to_elem.set("probability", "1.0")
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--output-route", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume-scale", type=float, default=1.0)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    net_path = Path(args.net_file)
    out_path = Path(args.output_route)
    temp_dir = out_path.parent / "temp_jtr"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    flow_file = temp_dir / f"flows_{args.seed}.xml"
    sink_file = temp_dir / f"sinks_{args.seed}.xml"
    
    sources, sinks = get_edges(net_path)
    
    if not sources or not sinks:
        sys.exit(1)

    base_volume = 800 * args.volume_scale 
    generate_flows_xml(flow_file, sources, 3600, base_volume)
    generate_sinks_xml(sink_file, sinks, 3600)
    
    r_right = max(0.1, min(0.3, random.gauss(0.20, 0.05)))
    r_left = max(0.05, min(0.25, random.gauss(0.15, 0.05)))
    r_straight = max(0.0, 1.0 - r_right - r_left)
    
    turn_defaults = f"{r_right:.2f},{r_straight:.2f},{r_left:.2f},0"
    
    cmd = [
        "jtrrouter",
        "--net-file", str(net_path),
        "--route-files", str(flow_file),
        "--turn-ratio-files", str(sink_file),
        "--output-file", str(out_path),
        "--turn-defaults", turn_defaults,
        "--accept-all-destinations", "true",
        "--seed", str(args.seed),
        "--no-step-log", "true"
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()