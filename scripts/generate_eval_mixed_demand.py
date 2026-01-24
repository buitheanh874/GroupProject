#!/usr/bin/env python
"""
Generate evaluation routes with time-varying demand (e.g., off-peak -> normal -> peak).
Uses the same turn ratios and vehicle distribution as training (86/12/2, 10/80/10).

Default phases (seconds): [1200, 1200, 1200]
Default per-lane demands (veh/hr/lane): [500, 750, 1000]
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

# Vehicle types and distribution (matching main configs)
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


def _ensure_lengths(phases: Sequence[int], demands: Sequence[float]) -> None:
    if len(phases) != len(demands):
        raise ValueError(f"phases and demands must have same length, got {len(phases)} vs {len(demands)}")
    if any(t <= 0 for t in phases):
        raise ValueError("phase durations must be > 0")
    if any(d <= 0 for d in demands):
        raise ValueError("demands must be > 0")


def _generate_single_route(
    net_file: Path,
    out_path: Path,
    phases_sec: Sequence[int],
    demands: Sequence[float],
    seed: int,
) -> int:
    sources_info, sinks = get_source_edges_info(net_file)
    conn_map = build_connection_map(net_file)
    transitions = build_transition_probabilities(
        conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT
    )

    root = ET.Element("routes")
    for vt in VTYPES:
        ET.SubElement(root, "vType", **vt)

    flows_written = 0
    phase_starts = [0]
    for t in phases_sec[:-1]:
        phase_starts.append(phase_starts[-1] + t)

    for phase_idx, (begin, demand_per_lane) in enumerate(zip(phase_starts, demands)):
        end = begin + phases_sec[phase_idx]
        for edge_id, lanes in sources_info.items():
            total_edge_flow = float(lanes) * float(demand_per_lane)
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
                    flow.set("id", f"f_{phase_idx}_{edge_id}_{idx}_{v_type}")
                    flow.set("from", str(edge_id))
                    flow.set("to", str(last_edge))
                    flow.set("begin", str(int(begin)))
                    flow.set("end", str(int(end)))
                    flow.set("perHour", f"{flow_rate:.2f}")
                    flow.set("type", str(v_type))
                    flow.set("departLane", "best")
                    flow.set("departSpeed", "max")
                    route = ET.SubElement(flow, "route")
                    route.set("edges", route_str)
                    flows_written += 1

    ET.indent(ET.ElementTree(root), space="    ")
    ET.ElementTree(root).write(out_path, encoding="UTF-8", xml_declaration=True)
    return flows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed-demand (time-varying) eval routes")
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml")
    parser.add_argument("--output-dir", type=str, default="networks/variants/eval_mixed_demand")
    parser.add_argument("--num-routes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=800)
    parser.add_argument("--phases", type=str, default="1200,1200,1200", help="Comma-separated seconds per phase")
    parser.add_argument("--demands", type=str, default="500,750,1000", help="Comma-separated per-lane demands per phase")
    parser.add_argument("--manifest-out", type=str,
                        default="networks/variants/train_turn801010/manifest_eval_mixed_demand.txt")
    args = parser.parse_args()

    phases_sec = [int(x) for x in args.phases.split(",") if x.strip()]
    demands = [float(x) for x in args.demands.split(",") if x.strip()]
    _ensure_lengths(phases_sec, demands)

    net_file = project_root / args.net_file
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for i in range(args.num_routes):
        seed = args.seed_start + i
        out_path = output_dir / f"bignet_mixed_seed{seed:05d}_phases{'-'.join(str(int(d)) for d in demands)}.rou.xml"
        flows = _generate_single_route(
            net_file=net_file,
            out_path=out_path,
            phases_sec=phases_sec,
            demands=demands,
            seed=seed,
        )
        generated.append(out_path)
        print(f"[{i+1}/{args.num_routes}] {out_path.name} | flows={flows} | phases={phases_sec} demands={demands}")

    manifest_path = project_root / args.manifest_out
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in generated:
        rel = os.path.relpath(r, manifest_path.parent)
        lines.append(rel.replace("\\", "/"))
    header = [
        "# Mixed demand routes (time-varying)",
        f"# phases_sec={phases_sec}, demands_per_lane={demands}",
        "# Generated by scripts/generate_eval_mixed_demand.py",
        "",
    ]
    manifest_path.write_text("\n".join(header + lines), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
