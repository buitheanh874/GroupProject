"""
Generate training routes for final design with 3 demand levels and 3 imbalance ratios.

Demand Levels:
- Easy: 350 veh/hr/lane (60 routes)
- Medium: 500 veh/hr/lane (30 routes)
- Hard: 650 veh/hr/lane (10 routes)

Imbalance Ratios (slightly offset from action splits 70:30, 50:50, 30:70):
- ns_heavy: 65:35 (NS flow higher)
- balanced: 50:50
- ew_heavy: 35:65 (EW flow higher)

Turning ratio: 80% straight, 10% left, 10% right
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.generate_jtr_data import (
    TURN_LEFT_WEIGHT,
    TURN_RIGHT_WEIGHT,
    TURN_STRAIGHT_WEIGHT,
    build_connection_map,
    build_transition_probabilities,
    enumerate_paths_from_source,
    get_source_edges_info,
)

# Final design constants
DEMAND_LEVELS = {
    "easy": {"avg": 350, "routes": 200},
    "medium": {"avg": 500, "routes": 100},
    "hard": {"avg": 650, "routes": 50},
}

# 3 imbalance levels (slightly offset from 70:30, 50:50, 30:70 actions)
IMBALANCE_RATIOS = [
    {"name": "ns_heavy", "ns_ratio": 0.65, "ew_ratio": 0.35},
    {"name": "balanced", "ns_ratio": 0.50, "ew_ratio": 0.50},
    {"name": "ew_heavy", "ns_ratio": 0.35, "ew_ratio": 0.65},
]

# Vehicle types
VTYPES = [
    {"id": "motorcycle", "vClass": "motorcycle", "length": "2.0", "width": "0.8", "maxSpeed": "13.89", "accel": "3.5", "decel": "4.0", "sigma": "0.8", "minGap": "0.5"},
    {"id": "passenger", "vClass": "passenger", "length": "4.5", "width": "1.8", "maxSpeed": "13.89", "accel": "2.5", "decel": "4.5", "sigma": "0.3", "minGap": "2.0"},
    {"id": "bus", "vClass": "bus", "length": "12.0", "width": "2.5", "maxSpeed": "10.0", "accel": "1.2", "decel": "2.5", "sigma": "0.1", "minGap": "2.5"},
]

VEHICLE_DISTRIBUTION = {"motorcycle": 0.86, "passenger": 0.12, "bus": 0.02}


def _lane_vector_from_shape(shape_str: str) -> Tuple[float, float]:
    pts = []
    for token in shape_str.split():
        try:
            x_str, y_str = token.split(",")
            pts.append((float(x_str), float(y_str)))
        except Exception:
            continue
    if len(pts) < 2:
        return (0.0, 0.0)
    return (pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])


def _edge_axis_map(net_path: Path, source_edges: Dict[str, int]) -> Dict[str, str]:
    """Infer axis ('NS' or 'EW') for each source edge."""
    tree = ET.parse(net_path)
    root = tree.getroot()
    axis_map: Dict[str, str] = {}
    
    lane_by_edge: Dict[str, List[Tuple[float, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id not in source_edges:
            continue
        vectors = []
        for lane in edge.findall("lane"):
            shape = lane.get("shape")
            if shape:
                vectors.append(_lane_vector_from_shape(shape))
        if vectors:
            lane_by_edge[edge_id] = vectors
    
    for edge_id in source_edges:
        vecs = lane_by_edge.get(edge_id, [])
        if not vecs:
            axis_map[edge_id] = "NS"
            continue
        vx = np.mean([v[0] for v in vecs])
        vy = np.mean([v[1] for v in vecs])
        axis_map[edge_id] = "NS" if abs(vy) >= abs(vx) else "EW"
    
    return axis_map


def generate_route_file(
    net_file: Path,
    output_file: Path,
    base_demand: float,
    ns_ratio: float,
    ew_ratio: float,
    duration: int = 1800,
) -> int:
    """Generate a single route file with given demand and imbalance."""
    sources_info, sinks = get_source_edges_info(net_file)
    conn_map = build_connection_map(net_file)
    transitions = build_transition_probabilities(
        conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT
    )
    axis_map = _edge_axis_map(net_file, sources_info)
    
    # Calculate per-axis flows
    ns_flow = base_demand * 2 * ns_ratio  # 2x because we split total across 2 axes
    ew_flow = base_demand * 2 * ew_ratio
    
    root = ET.Element("routes")
    
    # vTypes
    for vt in VTYPES:
        ET.SubElement(root, "vType", **vt)
    
    flows_written = 0
    for edge_id, lanes in sources_info.items():
        axis = axis_map.get(edge_id, "NS")
        per_lane = ns_flow if axis == "NS" else ew_flow
        total_edge_flow = per_lane * float(lanes)
        
        path_probs = enumerate_paths_from_source(
            source_edge=edge_id,
            transitions=transitions,
            sink_edges=sinks,
        )
        if not path_probs:
            continue
        
        for idx, (path_prob, path_edges) in enumerate(path_probs):
            route_flow = float(total_edge_flow) * float(path_prob)
            route_str = " ".join(path_edges)
            last_edge = path_edges[-1]
            
            for v_type, ratio in VEHICLE_DISTRIBUTION.items():
                flow_rate = route_flow * float(ratio)
                if flow_rate <= 1.0:
                    continue
                
                flow = ET.SubElement(root, "flow")
                flow.set("id", f"f_{edge_id}_{idx}_{v_type}")
                # Don't use from/to when using embedded route
                flow.set("begin", "0")
                flow.set("end", str(duration))
                flow.set("vehsPerHour", f"{flow_rate:.2f}")
                flow.set("type", str(v_type))
                flow.set("departLane", "best")
                flow.set("departSpeed", "max")
                
                route = ET.SubElement(flow, "route")
                route.set("edges", route_str)
                flows_written += 1
    
    ET.indent(ET.ElementTree(root), space="    ")
    ET.ElementTree(root).write(output_file, encoding="UTF-8", xml_declaration=True)
    return flows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training routes for final design")
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml")
    parser.add_argument("--output-dir", type=str, default="networks/variants/train_final")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--verify", action="store_true", help="Verify existing routes")
    args = parser.parse_args()
    
    net_file = project_root / args.net_file
    output_dir = project_root / args.output_dir
    
    if not net_file.exists():
        sys.exit(f"Network file not found: {net_file}")
    
    seed = args.seed_start
    
    for level_name, level_config in DEMAND_LEVELS.items():
        level_dir = output_dir / level_name
        level_dir.mkdir(parents=True, exist_ok=True)
        
        routes_per_imbalance = level_config["routes"] // 3
        remainder = level_config["routes"] % 3
        
        generated = []
        for i, imbal in enumerate(IMBALANCE_RATIOS):
            n_routes = routes_per_imbalance + (1 if i < remainder else 0)
            
            for j in range(n_routes):
                filename = f"bignet_d{level_config['avg']}_{imbal['name']}_seed{seed:05d}.rou.xml"
                output_file = level_dir / filename
                
                flows = generate_route_file(
                    net_file=net_file,
                    output_file=output_file,
                    base_demand=float(level_config["avg"]),
                    ns_ratio=imbal["ns_ratio"],
                    ew_ratio=imbal["ew_ratio"],
                    duration=args.duration,
                )
                
                generated.append(f"{level_name}/{filename}")
                seed += 1
                print(f"[{level_name}] {filename} | flows={flows}")
        
        # Write manifest for this level
        manifest_path = output_dir / f"manifest_{level_name}.txt"
        manifest_path.write_text("\n".join(generated), encoding="utf-8")
        print(f"Manifest: {manifest_path} ({len(generated)} routes)")
    
    print(f"\nDone! Total routes: {seed - args.seed_start}")


if __name__ == "__main__":
    main()
