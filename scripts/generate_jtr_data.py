from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

HANOI_BASE_FLOW_PER_LANE = 1000.0
TURN_LEFT_WEIGHT = 0.10
TURN_STRAIGHT_WEIGHT = 0.80
TURN_RIGHT_WEIGHT = 0.10
MIN_PATH_PROB = 1e-4
MAX_PATH_STEPS = 25


def get_source_edges_info(net_file: Path) -> Tuple[Dict[str, int], List[str]]:
    tree = ET.parse(net_file)
    root = tree.getroot()

    all_edges: Dict[str, int] = {}
    incoming = set()
    outgoing = set()

    for edge in root.findall("edge"):
        eid = edge.get("id")
        if not eid:
            continue
        if eid.startswith(":"):
            continue

        lanes = edge.findall("lane")
        if lanes:
            all_edges[eid] = len(lanes)

    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")

        if from_edge in all_edges:
            outgoing.add(from_edge)
        if to_edge in all_edges:
            incoming.add(to_edge)

    source_ids = [e for e in all_edges if e not in incoming]
    sink_ids = [e for e in all_edges if e not in outgoing]
    sources_info = {eid: all_edges[eid] for eid in source_ids}

    return sources_info, sorted(sink_ids)


def build_connection_map(net_file: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns mapping: from_edge -> list of (to_edge, dir) where dir in {'l','s','r'}.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    valid_edges = {e.get("id") for e in root.findall("edge") if e.get("id") and not e.get("id").startswith(":")}
    raw: Dict[str, Dict[str, set]] = {}
    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        direction = conn.get("dir", "")
        if not from_edge or not to_edge:
            continue
        if from_edge not in valid_edges or to_edge not in valid_edges:
            continue
        if direction not in {"l", "s", "r"}:
            continue
        dir_map = raw.setdefault(from_edge, {"l": set(), "s": set(), "r": set()})
        dir_map.setdefault(direction, set()).add(to_edge)

    conn_map: Dict[str, List[Tuple[str, str]]] = {}
    for from_edge, dir_map in raw.items():
        entries: List[Tuple[str, str]] = []
        for direction, to_edges in dir_map.items():
            for to_edge in sorted(to_edges):
                entries.append((to_edge, direction))
        conn_map[from_edge] = entries
    return conn_map


def _compute_turn_probabilities_for_edge(
    conns: List[Tuple[str, str]],
    left_weight: float,
    straight_weight: float,
    right_weight: float,
) -> List[Tuple[str, float]]:
    """
    Given outgoing connections for an edge, compute probability per connection
    using target weights (80% straight, 10% left, 10% right by default) and
    renormalizing to available directions.
    """
    by_dir: Dict[str, List[str]] = {"l": [], "s": [], "r": []}
    for to_edge, d in conns:
        by_dir.setdefault(d, []).append(to_edge)

    base_weight = {"l": left_weight, "s": straight_weight, "r": right_weight}
    present_dirs = [d for d in ["l", "s", "r"] if len(by_dir.get(d, [])) > 0 and base_weight[d] > 0.0]
    if len(present_dirs) == 0:
        # No valid connections; return empty and caller will treat as sink.
        return []

    total_present_weight = sum(base_weight[d] for d in present_dirs)
    probabilities: List[Tuple[str, float]] = []
    for d in present_dirs:
        share = base_weight[d] / total_present_weight
        choices = by_dir[d]
        per_conn = share / float(len(choices))
        for to_edge in choices:
            probabilities.append((to_edge, per_conn))
    # Normalize to ensure numerical stability
    total = sum(p for _, p in probabilities)
    if total > 0:
        probabilities = [(edge, p / total) for edge, p in probabilities]
    return probabilities


def build_transition_probabilities(
    conn_map: Dict[str, List[Tuple[str, str]]],
    left_weight: float = TURN_LEFT_WEIGHT,
    straight_weight: float = TURN_STRAIGHT_WEIGHT,
    right_weight: float = TURN_RIGHT_WEIGHT,
) -> Dict[str, List[Tuple[str, float]]]:
    transitions: Dict[str, List[Tuple[str, float]]] = {}
    for from_edge, conns in conn_map.items():
        probs = _compute_turn_probabilities_for_edge(conns, left_weight, straight_weight, right_weight)
        if probs:
            transitions[from_edge] = probs
    return transitions


def enumerate_paths_from_source(
    source_edge: str,
    transitions: Dict[str, List[Tuple[str, float]]],
    sink_edges: List[str],
    max_steps: int = MAX_PATH_STEPS,
    min_prob: float = MIN_PATH_PROB,
) -> List[Tuple[float, List[str]]]:
    sink_set = set(sink_edges)
    paths: List[Tuple[float, List[str]]] = []
    stack: List[Tuple[float, List[str]]] = [(1.0, [source_edge])]

    while stack:
        prob, path = stack.pop()
        current = path[-1]

        if prob < min_prob:
            continue
        if current in sink_set or current not in transitions or len(path) >= max_steps:
            paths.append((prob, path))
            continue

        for to_edge, p_conn in transitions[current]:
            if to_edge in path:
                continue  # avoid cycles
            stack.append((prob * p_conn, path + [to_edge]))

    total_prob = sum(p for p, _ in paths)
    if total_prob == 0:
        return []
    normalized = [(p / total_prob, path) for p, path in paths]
    return normalized


def generate_flows_xml(
    output_path: Path,
    sources_info: Dict[str, int],
    sinks: List[str],
    transitions: Dict[str, List[Tuple[str, float]]],
    duration: int,
    global_scale: float,
    base_flow: float = HANOI_BASE_FLOW_PER_LANE,
) -> None:
    root = ET.Element("routes")

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

    veh_distribution = {
        "motorcycle": 0.86,   # 86% xe may
        "passenger": 0.12,    # 12% o to
        "bus": 0.02,          # 2% bus
        # Removed 'other' type
    }

    for edge_id, num_lanes in sources_info.items():
        base_edge_flow = float(num_lanes) * float(base_flow)
        edge_noise = random.uniform(0.6, 1.1)
        total_edge_flow = base_edge_flow * float(global_scale) * float(edge_noise)

        path_probs = enumerate_paths_from_source(
            source_edge=edge_id,
            transitions=transitions,
            sink_edges=sinks,
            max_steps=MAX_PATH_STEPS,
            min_prob=MIN_PATH_PROB,
        )

        if not path_probs:
            continue

        for idx, (path_prob, path_edges) in enumerate(path_probs):
            route_flow = float(total_edge_flow) * float(path_prob)
            route_str = " ".join(path_edges)
            last_edge = path_edges[-1]
            for v_type, ratio in veh_distribution.items():
                flow_rate = route_flow * float(ratio)

                if flow_rate > 1.0:
                    flow = ET.SubElement(root, "flow")
                    flow.set("id", f"f_{edge_id}_{idx}_{v_type}")
                    flow.set("from", str(edge_id))
                    flow.set("to", str(last_edge))
                    flow.set("begin", "0")
                    flow.set("end", str(int(duration)))
                    # Use perHour for compatibility across SUMO GUI versions (vehsPerHour is newer alias).
                    flow.set("perHour", f"{flow_rate:.2f}")
                    flow.set("type", str(v_type))
                    flow.set("departLane", "best")
                    flow.set("departSpeed", "max")

                    route = ET.SubElement(flow, "route")
                    route.set("edges", route_str)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def generate_turnfile_xml(output_path: Path, sinks: List[str], duration: int) -> None:
    # Deprecated in the new duarouter-based pipeline; kept for compatibility.
    root = ET.Element("turns")
    interval = ET.SubElement(root, "interval")
    interval.set("begin", "0")
    interval.set("end", str(int(duration)))

    for sink_id in sinks:
        from_edge = ET.SubElement(interval, "fromEdge")
        from_edge.set("id", str(sink_id))
        
        to_edge = ET.SubElement(from_edge, "toEdge")
        to_edge.set("id", str(sink_id))
        to_edge.set("probability", "1.0")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hanoi-realistic route variants for SUMO network"
    )
    parser.add_argument("--net-file", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--output-route", required=True, help="Output route file (.rou.xml)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--volume-scale", type=float, default=1.0, help="Demand scaling factor (0.0-2.0)")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration (seconds)")
    parser.add_argument("--base-flow", type=float, default=HANOI_BASE_FLOW_PER_LANE,
                        help="Base flow per lane in veh/hr (train=500, eval=2000 for Hanoi)")
    args = parser.parse_args()

    random.seed(int(args.seed))

    net_path = Path(args.net_file)
    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")

    out_path = Path(args.output_route)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources_info, sinks = get_source_edges_info(net_path)

    if len(sources_info) == 0:
        sys.exit(f"No source edges found in network: {net_path}")

    if len(sinks) == 0:
        sys.exit(f"No sink edges found in network: {net_path}")

    volume_scale = max(0.1, min(2.0, float(args.volume_scale)))
    duration = max(60, int(args.duration))

    print(f"Generating route file: {out_path.name}")
    print(f"  Network: {net_path.name}")
    print(f"  Sources: {len(sources_info)}")
    print(f"  Sinks: {len(sinks)}")
    print(f"  Volume scale: {volume_scale:.2f}")
    print(f"  Duration: {duration}s")

    base_flow = max(100.0, float(args.base_flow))
    print(f"  Base flow: {base_flow:.0f} veh/hr/lane")
    print(f"  Turning ratios per intersection: straight={TURN_STRAIGHT_WEIGHT:.2f}, left={TURN_LEFT_WEIGHT:.2f}, right={TURN_RIGHT_WEIGHT:.2f}")

    conn_map = build_connection_map(net_path)
    transitions = build_transition_probabilities(conn_map, TURN_LEFT_WEIGHT, TURN_STRAIGHT_WEIGHT, TURN_RIGHT_WEIGHT)

    generate_flows_xml(
        output_path=out_path,
        sources_info=sources_info,
        sinks=sinks,
        transitions=transitions,
        duration=duration,
        global_scale=volume_scale,
        base_flow=base_flow,
    )

    if out_path.exists():
        file_size_kb = out_path.stat().st_size / 1024.0
        print(f"Route file created: {out_path} ({file_size_kb:.1f} KB)")
    else:
        sys.exit(f"Failed to create route file: {out_path}")

    print("Done")


if __name__ == "__main__":
    main()
