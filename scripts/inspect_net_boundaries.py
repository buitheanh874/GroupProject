#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_network_topology(net_file: Path) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    tree = ET.parse(str(net_file))
    root = tree.getroot()

    edge_incoming: Dict[str, int] = {}
    edge_outgoing: Dict[str, int] = {}
    all_edges: List[str] = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id:
            continue
        if edge_id.startswith(":"):
            continue

        lanes = edge.findall("lane")
        if not lanes:
            continue

        all_edges.append(edge_id)
        edge_incoming[edge_id] = 0
        edge_outgoing[edge_id] = 0

    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")

        if from_edge in edge_outgoing:
            edge_outgoing[from_edge] += 1
        if to_edge in edge_incoming:
            edge_incoming[to_edge] += 1

    return edge_incoming, edge_outgoing, all_edges


def identify_boundaries(
    incoming: Dict[str, int],
    outgoing: Dict[str, int],
    all_edges: List[str],
) -> Tuple[List[str], List[str]]:
    entry_edges: List[str] = []
    exit_edges: List[str] = []

    for edge_id in all_edges:
        in_degree = int(incoming.get(edge_id, 0))
        out_degree = int(outgoing.get(edge_id, 0))

        if in_degree == 0 and out_degree > 0:
            entry_edges.append(edge_id)

        if in_degree > 0 and out_degree == 0:
            exit_edges.append(edge_id)

    return sorted(entry_edges), sorted(exit_edges)


def generate_calibration_yaml(
    net_file: Path,
    entry_edges: List[str],
    exit_edges: List[str],
    output_path: Path,
) -> None:
    yaml_content = {
        "scenario": {
            "net_file": str(net_file.name),
            "entry_edges": entry_edges,
            "exit_edges": exit_edges,
            "vehicle_mix_mean": {
                "motorcycle": 0.84,
                "passenger": 0.12,
                "bus": 0.03,
                "other": 0.01,
            },
            "vehicle_mix_kappa": 50,
            "pcu_weights": {
                "motorcycle": 0.25,
                "passenger": 1.0,
                "bus": 3.0,
                "other": 1.0,
            },
            "demand": {
                "total_pcu_per_hour": {
                    "low": 3000,
                    "med": 5000,
                    "high": 7000,
                },
                "entry_dirichlet_alpha": 2.0,
            },
            "turning": {
                "mean_LSR": [0.15, 0.70, 0.15],
                "kappa": 30,
            },
            "turning_overrides": {},
            "stages": {
                "enabled": False,
                "intervals": [],
            },
            "simulation": {
                "step_length_sec": 1.0,
                "duration_sec": 3600,
                "min_total_vehicles": 100,
            },
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect entry/exit edges from SUMO network for Hanoi scenario calibration"
    )
    parser.add_argument("--net", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument(
        "--out",
        default="configs/scenario_hanoi_calibration.yaml",
        help="Output calibration YAML file",
    )
    args = parser.parse_args()

    net_path = Path(args.net)
    if not net_path.exists():
        sys.exit(f"ERROR: Network file not found: {net_path}")

    print("=" * 80)
    print("HANOI SCENARIO - NETWORK BOUNDARY DETECTION")
    print("=" * 80)
    print(f"Network: {net_path.name}\n")

    try:
        incoming, outgoing, all_edges = parse_network_topology(net_path)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to parse network: {exc}")

    entry_edges, exit_edges = identify_boundaries(incoming, outgoing, all_edges)

    print(f"Total edges in network: {len(all_edges)}\n")

    print(f"ENTRY EDGES (source - no incoming): {len(entry_edges)}")
    for edge in entry_edges:
        out_degree = outgoing.get(edge, 0)
        print(f"  - {edge} (out_degree={out_degree})")

    print()

    print(f"EXIT EDGES (sink - no outgoing): {len(exit_edges)}")
    for edge in exit_edges:
        in_degree = incoming.get(edge, 0)
        print(f"  - {edge} (in_degree={in_degree})")

    print()

    if len(entry_edges) == 0:
        print("WARNING: No entry edges detected!")
        print("Possible reasons:")
        print("  - Network has no clear boundaries (all edges interconnected)")
        print("  - Need manual specification in calibration file")
        print()

    if len(exit_edges) == 0:
        print("WARNING: No exit edges detected!")
        print("Possible reasons:")
        print("  - Network has no clear boundaries")
        print("  - Need manual specification in calibration file")
        print()

    out_path = Path(args.out)
    generate_calibration_yaml(net_path, entry_edges, exit_edges, out_path)

    print("=" * 80)
    print(f"Calibration file generated: {out_path}")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print(f"1. Review {out_path}")
    print("   - Verify entry/exit edges match your network topology")
    print("   - Adjust demand levels based on real data if available")
    print()
    print("2. Generate route variants:")
    print("   python scripts/generate_hanoi_route_variants.py \\")
    print(f"     --calib {out_path} \\")
    print("     --out-dir networks/variants \\")
    print("     --split train --n 50 --seed 42 --level med")
    print()


if __name__ == "__main__":
    main()
