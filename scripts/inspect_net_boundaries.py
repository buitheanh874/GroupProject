#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_net(net_path: Path) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    node_types: Dict[str, str] = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        if not node_id:
            continue
        node_types[node_id] = node.get("type", "")

    edges: List[Tuple[str, str, str]] = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        from_node = edge.get("from")
        to_node = edge.get("to")
        if from_node is None or to_node is None:
            continue
        edges.append((edge_id, from_node, to_node))

    return node_types, edges


def _compute_degrees(edges: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}
    for _, u, v in edges:
        outdeg[u] = outdeg.get(u, 0) + 1
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, 0)
        outdeg.setdefault(v, 0)
    return indeg, outdeg


def inspect_boundaries(net_path: Path) -> Tuple[List[str], List[str]]:
    node_types, edges = _parse_net(net_path)
    indeg, outdeg = _compute_degrees(edges)

    entry_edges: List[str] = []
    exit_edges: List[str] = []

    for edge_id, from_node, to_node in edges:
        from_type = node_types.get(from_node, "")
        to_type = node_types.get(to_node, "")

        if indeg.get(from_node, 0) == 0 or from_type == "dead_end":
            entry_edges.append(edge_id)
        if outdeg.get(to_node, 0) == 0 or to_type == "dead_end":
            exit_edges.append(edge_id)

    return sorted(set(entry_edges)), sorted(set(exit_edges))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heuristic entry/exit edge detection for Hanoi scenario calibration"
    )
    parser.add_argument("--net", required=True, help="Path to SUMO .net.xml file")
    parser.add_argument("--out", required=True, help="Output JSON path for detected boundaries")
    args = parser.parse_args()

    net_path = Path(args.net)
    if not net_path.exists():
        sys.exit(f"ERROR: Network file not found: {net_path}")

    try:
        entry_edges, exit_edges = inspect_boundaries(net_path)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to parse network: {exc}")

    result = {
        "net": str(net_path),
        "entry_edges": entry_edges,
        "exit_edges": exit_edges,
        "notes": [],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
