"""
Generate evaluation routes for fair comparison with RL.

This script generates routes for evaluation with proper imbalance patterns:
1. Unseen routes at d500 (same demand as training, different seeds)
2. Unseen demand d750 (never seen during training)
3. Mixed-demand routes (350→500→650 over 1500s)

All routes are 1500s duration to match eval horizon.
Imbalance ratios: ns_heavy (65:35), balanced (50:50), ew_heavy (35:65)
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

# Imbalance ratios (same as training)
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


def generate_constant_demand_route(
    net_file: Path,
    output_file: Path,
    base_demand: float,
    ns_ratio: float,
    ew_ratio: float,
    duration: int = 1500,
) -> int:
    """Generate a route file with constant demand throughout."""
    sources_info, sinks = get_source_edges_info(net_file)
    conn_map = build_connection_map(net_file)
    transitions = build_transition_probabilities(
        conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT
    )
    axis_map = _edge_axis_map(net_file, sources_info)
    
    # Calculate per-axis flows
    ns_flow = base_demand * 2 * ns_ratio
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
            
            for v_type, ratio in VEHICLE_DISTRIBUTION.items():
                flow_rate = route_flow * float(ratio)
                if flow_rate <= 1.0:
                    continue
                
                flow = ET.SubElement(root, "flow")
                flow.set("id", f"f_{edge_id}_{idx}_{v_type}")
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


def generate_mixed_demand_route(
    net_file: Path,
    output_file: Path,
    demands: List[Tuple[int, int, float]],  # (begin, end, demand)
    ns_ratio: float,
    ew_ratio: float,
) -> int:
    """Generate a route file with time-varying demand levels.
    
    demands: list of (begin_sec, end_sec, demand_veh_hr_lane)
    """
    sources_info, sinks = get_source_edges_info(net_file)
    conn_map = build_connection_map(net_file)
    transitions = build_transition_probabilities(
        conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT
    )
    axis_map = _edge_axis_map(net_file, sources_info)
    
    root = ET.Element("routes")
    
    # vTypes
    for vt in VTYPES:
        ET.SubElement(root, "vType", **vt)
    
    flows_written = 0
    
    for period_idx, (begin, end, base_demand) in enumerate(demands):
        # Calculate per-axis flows for this period
        ns_flow = base_demand * 2 * ns_ratio
        ew_flow = base_demand * 2 * ew_ratio
        
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
                
                for v_type, ratio in VEHICLE_DISTRIBUTION.items():
                    flow_rate = route_flow * float(ratio)
                    if flow_rate <= 1.0:
                        continue
                    
                    flow = ET.SubElement(root, "flow")
                    flow.set("id", f"f_{edge_id}_{idx}_{v_type}_p{period_idx}")
                    flow.set("begin", str(begin))
                    flow.set("end", str(end))
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


def generate_unseen_routes(
    net_file: Path,
    output_dir: Path,
    demand: int,
    seed_start: int,
    count_per_imbalance: int = 10,
    duration: int = 1500,
) -> List[str]:
    """Generate unseen routes at specified demand level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    seed = seed_start
    
    for imbal in IMBALANCE_RATIOS:
        for _ in range(count_per_imbalance):
            filename = f"bignet_d{demand}_{imbal['name']}_seed{seed:05d}.rou.xml"
            output_file = output_dir / filename
            
            flows = generate_constant_demand_route(
                net_file=net_file,
                output_file=output_file,
                base_demand=float(demand),
                ns_ratio=imbal["ns_ratio"],
                ew_ratio=imbal["ew_ratio"],
                duration=duration,
            )
            
            generated.append(filename)
            print(f"[d{demand}] {filename} | flows={flows}")
            seed += 1
    
    # Write manifest
    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(generated), encoding="utf-8")
    print(f"Manifest: {manifest_path} ({len(generated)} routes)")
    
    return generated


def generate_mixed_demand_routes(
    net_file: Path,
    output_dir: Path,
    seed_start: int,
    count_per_imbalance: int = 10,
    duration: int = 1500,
) -> List[str]:
    """Generate mixed-demand routes (350→500→650 over duration)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    seed = seed_start
    
    # 3 equal periods: 1/3 each
    period_len = duration // 3
    demands = [
        (0, period_len, 350),                    # First 1/3: 350 veh/hr/lane
        (period_len, 2 * period_len, 500),       # Second 1/3: 500 veh/hr/lane
        (2 * period_len, duration, 650),         # Final 1/3: 650 veh/hr/lane
    ]
    
    for imbal in IMBALANCE_RATIOS:
        for _ in range(count_per_imbalance):
            filename = f"bignet_mixed_{imbal['name']}_seed{seed:05d}.rou.xml"
            output_file = output_dir / filename
            
            flows = generate_mixed_demand_route(
                net_file=net_file,
                output_file=output_file,
                demands=demands,
                ns_ratio=imbal["ns_ratio"],
                ew_ratio=imbal["ew_ratio"],
            )
            
            generated.append(filename)
            print(f"[mixed] {filename} | flows={flows}")
            seed += 1
    
    # Write manifest
    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(generated), encoding="utf-8")
    print(f"Manifest: {manifest_path} ({len(generated)} routes)")
    
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation routes")
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml")
    parser.add_argument("--output-base", type=str, default="networks/variants/eval_final")
    parser.add_argument("--count-per-imbalance", type=int, default=10)
    parser.add_argument("--duration", type=int, default=1500, help="Route duration (s)")
    args = parser.parse_args()
    
    net_file = project_root / args.net_file
    output_base = project_root / args.output_base
    
    if not net_file.exists():
        sys.exit(f"Network file not found: {net_file}")
    
    print("=" * 60)
    print("Generating Evaluation Routes")
    print("=" * 60)
    
    # 1. Unseen d500 routes (seeds 10001+)
    print("\n[1/3] Generating unseen d500 routes...")
    generate_unseen_routes(
        net_file=net_file,
        output_dir=output_base / "d500_unseen",
        demand=500,
        seed_start=10001,
        count_per_imbalance=args.count_per_imbalance,
        duration=args.duration,
    )
    
    # 2. Unseen d750 routes (seeds 20001+)
    print("\n[2/3] Generating unseen d750 routes...")
    generate_unseen_routes(
        net_file=net_file,
        output_dir=output_base / "d750_unseen",
        demand=750,
        seed_start=20001,
        count_per_imbalance=args.count_per_imbalance,
        duration=args.duration,
    )
    
    # 3. Mixed-demand routes (seeds 30001+)
    print("\n[3/3] Generating mixed-demand routes...")
    generate_mixed_demand_routes(
        net_file=net_file,
        output_dir=output_base / "mixed_demand",
        seed_start=30001,
        count_per_imbalance=args.count_per_imbalance,
        duration=args.duration,
    )
    
    print("\n" + "=" * 60)
    print("Done! Generated 90 evaluation routes total.")
    print("=" * 60)


if __name__ == "__main__":
    main()
