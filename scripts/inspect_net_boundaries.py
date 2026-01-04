from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set


def parse_network_topology(net_file: Path) -> tuple[Dict[str, int], Dict[str, int], List[str]]:
    tree = ET.parse(str(net_file))
    root = tree.getroot()

    edge_incoming: Dict[str, int] = {}
    edge_outgoing: Dict[str, int] = {}
    all_edges: List[str] = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
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
) -> tuple[List[str], List[str]]:
    entry_edges = []
    exit_edges = []

    for edge_id in all_edges:
        in_degree = incoming.get(edge_id, 0)
        out_degree = outgoing.get(edge_id, 0)

        if in_degree == 0 and out_degree > 0:
            entry_edges.append(edge_id)

        if in_degree > 0 and out_degree == 0:
            exit_edges.append(edge_id)

    return sorted(entry_edges), sorted(exit_edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect network boundaries and suggest entry/exit edges")
    parser.add_argument("--net", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--out", default="boundary_suggestion.yaml", help="Output YAML file")
    args = parser.parse_args()

    net_path = Path(args.net)
    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")

    print("=" * 80)
    print("NETWORK BOUNDARY INSPECTION")
    print("=" * 80)
    print(f"Network: {net_path.name}")
    print()

    try:
        incoming, outgoing, all_edges = parse_network_topology(net_path)
    except Exception as exc:
        sys.exit(f"Failed to parse network: {exc}")

    entry_edges, exit_edges = identify_boundaries(incoming, outgoing, all_edges)

    print(f"Total edges: {len(all_edges)}")
    print()
    print("ENTRY EDGES (source nodes - no incoming connections):")
    for edge in entry_edges:
        print(f"  - {edge}")
    print()
    print("EXIT EDGES (sink nodes - no outgoing connections):")
    for edge in exit_edges:
        print(f"  - {edge}")
    print()

    yaml_content = f"""scenario:
  net_file: {net_path.name}

  entry_edges:
"""
    for edge in entry_edges:
        yaml_content += f"    - {edge}\n"

    yaml_content += """
  exit_edges:
"""
    for edge in exit_edges:
        yaml_content += f"    - {edge}\n"

    yaml_content += """
  vehicle_mix_mean:
    motorcycle: 0.84
    passenger: 0.12
    bus: 0.03
    other: 0.01

  vehicle_mix_kappa: 50

  pcu_weights:
    motorcycle: 0.25
    passenger: 1.0
    bus: 3.0
    other: 1.0

  demand:
    total_pcu_per_hour:
      low: 3000
      med: 5000
      high: 7000
    entry_dirichlet_alpha: 2.0

  turning:
    mean_LSR: [0.15, 0.70, 0.15]
    kappa: 30

  turning_overrides: {}

  simulation:
    step_length_sec: 1.0
    duration_sec: 3600
    min_total_vehicles: 100
"""

    out_path = Path(args.out)
    out_path.write_text(yaml_content, encoding="utf-8")

    print("=" * 80)
    print(f"Configuration template saved: {out_path}")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Review {out_path} - adjust entry/exit edges if needed")
    print("2. Tune vehicle_mix_mean and demand levels based on local data")
    print("3. Run route generation:")
    print(f"   python scripts/generate_hanoi_route_variants.py \\")
    print(f"     --calib {out_path} \\")
    print(f"     --out-dir networks/variants \\")
    print(f"     --split train --n 100 --seed 42")
    print()


if __name__ == "__main__":
    main()