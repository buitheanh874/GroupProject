from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def extract_tls_ids(net_file: str) -> list[str]:
    tree = ET.parse(net_file)
    root = tree.getroot()

    tls_ids = []
    for tl in root.findall(".//tlLogic"):
        tls_id = tl.get("id")
        if tls_id:
            tls_ids.append(tls_id)

    return sorted(set(tls_ids))


def extract_junction_info(net_file: str, tls_id: str) -> dict:
    tree = ET.parse(net_file)
    root = tree.getroot()

    junction = root.find(f".//junction[@id='{tls_id}']")
    if junction is None:
        return None

    incoming_lanes = []
    for connection in root.findall(".//connection"):
        to_edge = connection.get("to")
        if isinstance(to_edge, str) and to_edge.startswith(tls_id):
            from_edge = connection.get("from")
            from_lane = connection.get("fromLane")
            if from_edge is not None and from_lane is not None:
                incoming_lanes.append(f"{from_edge}_{from_lane}")

    return {
        "lanes_ns_ctrl": [],
        "lanes_ew_ctrl": [],
        "lanes_right_turn_slip_ns": [],
        "lanes_right_turn_slip_ew": [],
    }


def generate_config(
    net_file: str,
    route_file: str,
    output_file: str,
    center_tls: str = None,
):
    tls_ids = extract_tls_ids(net_file)

    if len(tls_ids) == 0:
        raise ValueError(f"No traffic lights found in {net_file}")

    print(f"Found {len(tls_ids)} traffic lights: {tls_ids}")

    if center_tls is None:
        center_tls = tls_ids[0]
    elif center_tls not in tls_ids:
        raise ValueError(f"Center TLS '{center_tls}' not found in network")

    config = {
        "run": {
            "run_name": "train_5junction",
            "seed": 42,
            "device": "cpu",
        },
        "env": {
            "type": "sumo",
            "sumo": {
                "sumo_binary": "sumo",
                "net_file": net_file,
                "route_file": route_file,
                "additional_files": [],
                "tls_ids": tls_ids,
                "center_tls_id": center_tls,
                "downstream_links": {
                    "N": "EDGE_TODO",
                    "E": "EDGE_TODO",
                    "S": "EDGE_TODO",
                    "W": "EDGE_TODO",
                },
                "vehicle_weights": {
                    "motorcycle": 0.25,
                    "passenger": 1.0,
                    "bus": 3.0,
                },
                "step_length_sec": 1.0,
                "green_cycle_sec": 60,
                "yellow_sec": 2,
                "all_red_sec": 0,
                "max_cycles": 0,
                "max_sim_seconds": 3600,
                "terminate_on_empty": False,
                "rho_min": 0.1,
                "lambda_fairness": 0.0,
                "action_splits": [
                    [0.30, 0.70],
                    [0.40, 0.60],
                    [0.50, 0.50],
                    [0.60, 0.40],
                    [0.70, 0.30],
                ],
                "action_table": [],
                "include_transition_in_waiting": True,
                "normalize_state": True,
                "return_raw_state": False,
                "enable_kpi_tracker": True,
                "sumo_extra_args": [],
                "state_dim": 12,
                "enable_downstream_occupancy": True,
                "lane_groups": {
                    "lanes_ns_ctrl": [],
                    "lanes_ew_ctrl": [],
                    "lanes_right_turn_slip_ns": [],
                    "lanes_right_turn_slip_ew": [],
                },
                "phase_program": {
                    "ns_green": 0,
                    "ew_green": 4,
                    "ns_yellow": 1,
                    "ew_yellow": 5,
                    "all_red": None,
                },
            },
        },
        "normalization": {
            "mean": [0.0] * 12,
            "std": [1.0] * 12,
            "file": "configs/norm_stats_5junction.json",
        },
        "agent": {
            "hidden_dims": [128, 128],
            "gamma": 0.98,
            "learning_rate": 0.0001,
            "batch_size": 64,
            "replay_buffer_size": 100000,
            "target_update_freq": 1000,
        },
        "exploration": {
            "eps_start": 1.0,
            "eps_end": 0.05,
            "eps_decay_steps": 20000,
        },
        "train": {
            "episodes": 300,
            "save_every_episodes": 50,
            "print_every_episodes": 10,
            "route_pool": [route_file],
        },
        "baseline": {
            "fixed_action_id": 7,
        },
        "logging": {
            "log_dir": "logs/5junction",
            "model_dir": "models/5junction",
            "results_dir": "results/5junction",
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated config: {output_file}")
    print("\nTODO items:")
    print("   1. Map downstream_links (N/E/S/W edges)")
    print(f"   2. Map lane_groups for center TLS: {center_tls}")
    print("   3. Collect normalization stats:")
    print("      python scripts/collect_norm_stats.py \\")
    print(f"        --config {output_file} \\")
    print("        --episodes 5 \\")
    print("        --out configs/norm_stats_5junction.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True, help="SUMO network file")
    parser.add_argument("--route", required=True, help="SUMO route file")
    parser.add_argument("--output", default="configs/train_5junction_auto.yaml")
    parser.add_argument("--center-tls", default=None, help="Center TLS ID")
    args = parser.parse_args()

    generate_config(
        net_file=args.net,
        route_file=args.route,
        output_file=args.output,
        center_tls=args.center_tls,
    )


if __name__ == "__main__":
    main()
