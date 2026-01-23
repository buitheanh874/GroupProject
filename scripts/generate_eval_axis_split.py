#!/usr/bin/env python
"""
Generate evaluation routes with axis-imbalanced demand:
- One axis (NS or EW) high demand (veh/hr/lane)
- The other axis low demand

Usage example:
python scripts/generate_eval_axis_split.py --axis-high ns --high-flow 1000 --low-flow 500 --num-routes 10 --output-dir networks/variants/eval_axis_split
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.generate_jtr_data import (  # type: ignore
    TURN_LEFT_WEIGHT,
    TURN_RIGHT_WEIGHT,
    TURN_STRAIGHT_WEIGHT,
    build_connection_map,
    build_transition_probabilities,
    enumerate_paths_from_source,
    get_source_edges_info,
)

# Vehicle types and distribution (matching training/eval defaults)
VTYPES = [
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

VEHICLE_DISTRIBUTION = {
    "motorcycle": 0.86,
    "passenger": 0.12,
    "bus": 0.02,
}


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
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return (x1 - x0, y1 - y0)


def _edge_axis_map(net_path: Path, source_edges: Dict[str, int]) -> Dict[str, str]:
    """
    Infer axis ('NS' or 'EW') for each source edge using lane geometry.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()
    axis_map: Dict[str, str] = {}

    lane_by_edge: Dict[str, List[Tuple[float, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id not in source_edges:
            continue
        lanes = edge.findall("lane")
        vectors: List[Tuple[float, float]] = []
        for lane in lanes:
            shape = lane.get("shape")
            if shape:
                vectors.append(_lane_vector_from_shape(shape))
        if vectors:
            lane_by_edge[edge_id] = vectors

    for edge_id in source_edges:
        vecs = lane_by_edge.get(edge_id, [])
        if not vecs:
            axis_map[edge_id] = "NS"  # fallback
            continue
        vx = np.mean([v[0] for v in vecs])
        vy = np.mean([v[1] for v in vecs])
        axis_map[edge_id] = "NS" if abs(vy) >= abs(vx) else "EW"

    return axis_map


def _generate_routes(
    net_file: Path,
    output_dir: Path,
    num_routes: int,
    seed_start: int,
    duration: int,
    axis_high: str,
    high_flow: float,
    low_flow: float,
) -> List[Path]:
    sources_info, sinks = get_source_edges_info(net_file)
    conn_map = build_connection_map(net_file)
    transitions = build_transition_probabilities(
        conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT
    )
    axis_map = _edge_axis_map(net_file, sources_info)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    for i in range(num_routes):
        seed = seed_start + i
        route_file = output_dir / f"bignet_axis_{axis_high}_seed{seed:05d}_d{int(high_flow)}_{int(low_flow)}.rou.xml"

        root = ET.Element("routes")

        # vTypes
        for vt in VTYPES:
            ET.SubElement(root, "vType", **vt)

        flows_written = 0
        for edge_id, lanes in sources_info.items():
            axis = axis_map.get(edge_id, "NS")
            per_lane = high_flow if axis.lower() == axis_high.lower() else low_flow
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
                    flow.set("from", str(edge_id))
                    flow.set("to", str(last_edge))
                    flow.set("begin", "0")
                    flow.set("end", str(int(duration)))
                    flow.set("perHour", f"{flow_rate:.2f}")
                    flow.set("type", str(v_type))
                    flow.set("departLane", "best")
                    flow.set("departSpeed", "max")

                    route = ET.SubElement(flow, "route")
                    route.set("edges", route_str)

                    flows_written += 1

        ET.indent(ET.ElementTree(root), space="    ")
        ET.ElementTree(root).write(route_file, encoding="UTF-8", xml_declaration=True)
        generated.append(route_file)
        print(
            f"[{i+1}/{num_routes}] {route_file.name} | flows={flows_written} | axis_high={axis_high} per_lane high/low={high_flow}/{low_flow}"
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate axis-imbalanced eval routes")
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml")
    parser.add_argument("--output-dir", type=str, default="networks/variants/eval_axis_split")
    parser.add_argument("--num-routes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--axis-high", choices=["ns", "ew"], default="ns")
    parser.add_argument("--high-flow", type=float, default=1000.0, help="veh/hr/lane for high axis")
    parser.add_argument("--low-flow", type=float, default=500.0, help="veh/hr/lane for low axis")
    parser.add_argument("--manifest-out", type=str,
                        default="networks/variants/train_turn801010/manifest_eval_axis_split.txt")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    net_file = project_root / args.net_file
    output_dir = project_root / args.output_dir

    generated = _generate_routes(
        net_file=net_file,
        output_dir=output_dir,
        num_routes=args.num_routes,
        seed_start=args.seed_start,
        duration=args.duration,
        axis_high=args.axis_high,
        high_flow=args.high_flow,
        low_flow=args.low_flow,
    )

    manifest_path = project_root / args.manifest_out
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import os
    lines = []
    for r in generated:
        rel = os.path.relpath(r, manifest_path.parent)
        lines.append(rel.replace("\\", "/"))
    header = [
        "# Axis-imbalanced routes",
        f"# axis_high={args.axis_high}, per_lane high/low={args.high_flow}/{args.low_flow}, duration={args.duration}s",
        "# Generated by scripts/generate_eval_axis_split.py",
        "",
    ]
    manifest_path.write_text("\n".join(header + lines), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
