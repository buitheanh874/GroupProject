#!/usr/bin/env python
"""
Generate evaluation route files where each entry direction receives a random
per-lane demand in the range [flow_min, flow_max]. Demands differ across
directions (unbalanced) but stay within the user-provided bounds.

Outputs:
    - Route files under <output_dir>/<demand>/..._d<demand>.rou.xml
    - Per-demand manifest.txt files
    - Optional aggregate manifest for eval (`--train-manifest-out`)
    - Metadata JSON per route listing the sampled per-direction flows
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
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

VEHICLE_DISTRIBUTION = {
    "motorcycle": 0.86,
    "passenger": 0.12,
    "bus": 0.02,
}

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


def _compute_flow_range(target: int | None, flow_min: float | None, flow_max: float | None) -> Tuple[float, float]:
    """
    Determine the per-lane flow sampling range.

    If custom bounds are provided, they take precedence. Otherwise, derive a
    narrow band around the target demand while clamping to [500, 1000].
    """
    base_min = 500.0 if flow_min is None else float(flow_min)
    base_max = 1000.0 if flow_max is None else float(flow_max)

    if target is not None and flow_min is None and flow_max is None:
        half_span = 200.0
        base_min = max(500.0, float(target) - half_span)
        base_max = min(1000.0, float(target) + half_span)

    if base_min <= 0 or base_max <= base_min:
        raise ValueError(f"Invalid flow range: [{base_min}, {base_max}]")

    return base_min, base_max


def _add_vtypes(root: ET.Element) -> None:
    for vt in VTYPES:
        ET.SubElement(root, "vType", **vt)


def _sample_edge_flows(
    sources_info: Dict[str, int],
    rng: random.Random,
    flow_range: Tuple[float, float],
) -> Dict[str, Dict[str, float]]:
    """
    Sample per-direction (edge) flows in veh/hr/lane within the flow_range.
    """
    lower, upper = flow_range
    sampled: Dict[str, Dict[str, float]] = {}
    for edge_id, num_lanes in sources_info.items():
        per_lane = rng.uniform(lower, upper)
        sampled[edge_id] = {
            "per_lane": per_lane,
            "lanes": float(num_lanes),
            "total_per_hour": per_lane * float(num_lanes),
        }
    return sampled


def _write_route_file(
    output_path: Path,
    sources_info: Dict[str, int],
    sinks: List[str],
    transitions: Dict[str, List[Tuple[str, float]]],
    duration: int,
    edge_flows: Dict[str, Dict[str, float]],
) -> int:
    """
    Write the SUMO route file with randomized per-direction flows.

    Returns the number of flow elements written.
    """
    root = ET.Element("routes")
    _add_vtypes(root)

    flows_written = 0

    for edge_id, num_lanes in sources_info.items():
        total_edge_flow = edge_flows[edge_id]["total_per_hour"]

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

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    return flows_written


def _write_manifest(manifest_path: Path, routes: Iterable[Path]) -> None:
    lines = [r.name for r in sorted(routes)]
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate eval routes with randomized per-direction demand in [500, 1000]",
    )
    parser.add_argument("--net-file", type=str, default="networks/BIGNET.net.xml", help="SUMO network file")
    parser.add_argument("--output-dir", type=str, default="networks/variants/eval_random_dirflow",
                        help="Output root directory for generated routes")
    parser.add_argument("--demands", type=str, default="500,750,1000",
                        help="Comma-separated target demand labels (used for folder/file naming)")
    parser.add_argument("--num-routes", type=int, default=10, help="Number of route files per demand")
    parser.add_argument("--seed-start", type=int, default=400, help="Base seed for sampling flows")
    parser.add_argument("--duration", type=int, default=3600, help="Injection duration in seconds")
    parser.add_argument("--flow-min", type=float, default=None, help="Optional lower bound (veh/hr/lane)")
    parser.add_argument("--flow-max", type=float, default=None, help="Optional upper bound (veh/hr/lane)")
    parser.add_argument("--train-manifest-out", type=str,
                        default="networks/variants/train_turn801010/manifest_eval_random_dirflow.txt",
                        help="Path for aggregate manifest usable by eval.py --route-manifest")
    parser.add_argument("--metadata", action="store_true", default=True,
                        help="Write per-route metadata JSON (on by default)")
    parser.add_argument("--no-metadata", dest="metadata", action="store_false",
                        help="Disable writing metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    net_path = project_root / args.net_file
    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")

    demands: List[int] = []
    for part in args.demands.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            demands.append(int(part))
        except ValueError:
            sys.exit(f"Invalid demand value: {part}")

    sources_info, sinks = get_source_edges_info(net_path)
    conn_map = build_connection_map(net_path)
    transitions = build_transition_probabilities(
        conn_map,
        TURN_LEFT_WEIGHT,
        TURN_STRAIGHT_WEIGHT,
        TURN_RIGHT_WEIGHT,
    )

    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_routes: List[Path] = []

    print(f"Sources: {len(sources_info)}, Sinks: {len(sinks)}")
    print(f"Output dir: {output_root}")

    for demand in demands:
        flow_range = _compute_flow_range(demand, args.flow_min, args.flow_max)
        print(f"\nDemand label {demand}: sampling per-lane flow in [{flow_range[0]:.1f}, {flow_range[1]:.1f}] veh/hr")

        demand_dir = output_root / f"{demand}"
        demand_dir.mkdir(parents=True, exist_ok=True)
        demand_routes: List[Path] = []

        for idx in range(args.num_routes):
            seed = args.seed_start + idx + demand * 1000
            rng = random.Random(seed)

            edge_flows = _sample_edge_flows(sources_info, rng, flow_range)

            outfile = demand_dir / f"bignet_dirflow_seed{seed:05d}_d{demand}_rand_dirflow.rou.xml"
            flows_written = _write_route_file(
                output_path=outfile,
                sources_info=sources_info,
                sinks=sinks,
                transitions=transitions,
                duration=args.duration,
                edge_flows=edge_flows,
            )

            demand_routes.append(outfile)
            aggregate_routes.append(outfile)

            per_lane_vals = [v["per_lane"] for v in edge_flows.values()]
            print(
                f"  [{idx + 1}/{args.num_routes}] seed={seed} -> {outfile.name} "
                f"(flows={flows_written}, min_per_lane={min(per_lane_vals):.1f}, max_per_lane={max(per_lane_vals):.1f})"
            )

            if args.metadata:
                meta = {
                    "seed": seed,
                    "demand_label": demand,
                    "flow_range": {"min": flow_range[0], "max": flow_range[1]},
                    "edges": edge_flows,
                }
                meta_path = outfile.with_suffix(".meta.json")
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        manifest_path = demand_dir / "manifest.txt"
        _write_manifest(manifest_path, demand_routes)
        print(f"  Manifest written: {manifest_path}")

    # Aggregate manifest (relative paths from its own directory)
    if args.train_manifest_out:
        manifest_out_path = project_root / args.train_manifest_out
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)

        rel_lines: List[str] = []
        for route in sorted(aggregate_routes):
            rel_path = os.path.relpath(route, manifest_out_path.parent)
            rel_lines.append(rel_path.replace("\\", "/"))

        header = [
            "# Randomized per-direction demand routes",
            "# Each entry edge samples veh/hr/lane in [500,1000] (or provided bounds)",
            "# File generated by scripts/generate_eval_random_dirflow.py",
            "",
        ]
        manifest_out_path.write_text("\n".join(header + rel_lines), encoding="utf-8")
        print(f"\nAggregate manifest written: {manifest_out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
