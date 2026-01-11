"""
Extract lane directions from SUMO network geometry.
This script determines which direction (N/E/S/W) each lane approaches a junction from,
based on the lane's geometric shape/angle rather than lane name heuristics.
"""
import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

def parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    """Parse SUMO shape string into list of (x, y) coordinates."""
    points = []
    for part in shape_str.strip().split():
        coords = part.split(',')
        if len(coords) >= 2:
            points.append((float(coords[0]), float(coords[1])))
    return points

def compute_approach_angle(shape: List[Tuple[float, float]]) -> float:
    """
    Compute the angle at which a lane approaches its endpoint (junction).
    Returns angle in degrees: 0=East, 90=North, 180=West, 270=South.
    """
    if len(shape) < 2:
        return 0.0
    
    x1, y1 = shape[-2]
    x2, y2 = shape[-1]
    
    dx = x2 - x1
    dy = y2 - y1
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

def angle_to_direction(angle: float) -> str:
    """
    Convert angle to cardinal direction (N/E/S/W).
    The direction represents WHERE the traffic is COMING FROM.
    
    If a lane points East (angle ~0), traffic comes from the West → W
    If a lane points North (angle ~90), traffic comes from the South → S
    If a lane points West (angle ~180), traffic comes from the East → E
    If a lane points South (angle ~270), traffic comes from the North → N
    """
    angle = angle % 360
    
    if 315 <= angle or angle < 45:
        return "W"
    elif 45 <= angle < 135:
        return "S"
    elif 135 <= angle < 225:
        return "E"
    else:
        return "N"

def extract_lane_directions(net_file: str) -> Dict[str, Dict[str, str]]:
    """
    Extract lane directions from network file.
    Returns: {lane_id: {"direction": "N/E/S/W", "angle": degrees, "to_junction": junction_id}}
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    lane_info: Dict[str, Dict[str, str]] = {}
    
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id', '')
        to_junction = edge.get('to', '')
        
        if edge_id.startswith(':'):
            continue
            
        for lane in edge.findall('lane'):
            lane_id = lane.get('id', '')
            shape_str = lane.get('shape', '')
            
            if not shape_str:
                continue
                
            shape = parse_shape(shape_str)
            if len(shape) < 2:
                continue
                
            angle = compute_approach_angle(shape)
            direction = angle_to_direction(angle)
            
            lane_info[lane_id] = {
                "direction": direction,
                "angle": round(angle, 1),
                "to_junction": to_junction
            }
    
    return lane_info

def generate_approach_lanes_config(
    net_file: str,
    lane_groups_by_tls: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate approach_lanes configuration for each TLS based on geometry.
    
    Args:
        net_file: Path to SUMO network file
        lane_groups_by_tls: Current config with lanes_ns_ctrl and lanes_ew_ctrl
        
    Returns:
        approach_lanes config: {tls_id: {"N": [...], "E": [...], "S": [...], "W": [...]}}
    """
    lane_directions = extract_lane_directions(net_file)
    
    result: Dict[str, Dict[str, List[str]]] = {}
    
    for tls_id, groups in lane_groups_by_tls.items():
        approach = {"N": [], "E": [], "S": [], "W": []}
        
        all_lanes = groups.get("lanes_ns_ctrl", []) + groups.get("lanes_ew_ctrl", [])
        
        for lane_id in all_lanes:
            if lane_id in lane_directions:
                direction = lane_directions[lane_id]["direction"]
                approach[direction].append(lane_id)
            else:
                print(f"[WARN] Lane '{lane_id}' not found in network, skipping")
        
        result[tls_id] = approach
    
    return result

def main():
    net_file = "networks/BIGNET.net.xml"
    
    lane_groups_by_tls = {
        "J0": {
            "lanes_ns_ctrl": ["-E3_0", "-E3_1", "-E2_0", "-E2_1"],
            "lanes_ew_ctrl": ["-E1_0", "-E1_1", "-E1_2", "-E0_0", "-E0_1", "-E0_2"]
        },
        "J3": {
            "lanes_ns_ctrl": ["-E25_0", "-E25_1", "E2_0", "E2_1"],
            "lanes_ew_ctrl": ["-E26_0", "E24_0"]
        },
        "J1": {
            "lanes_ns_ctrl": ["-E6_0", "-E6_1", "-E5_0", "-E5_1"],
            "lanes_ew_ctrl": ["E0_0", "E0_1", "E0_2", "-E4_0", "-E4_1", "-E4_2"]
        },
        "J4": {
            "lanes_ns_ctrl": ["E3_0", "E3_1", "-E16_0", "-E16_1"],
            "lanes_ew_ctrl": ["E14_0", "-E15_0"]
        },
        "J2": {
            "lanes_ns_ctrl": ["-E19_0", "-E19_1", "-E18_0", "-E18_1"],
            "lanes_ew_ctrl": ["E1_0", "E1_1", "E1_2", "-E17_0", "-E17_1", "-E17_2"]
        },
        "J7": {
            "lanes_ns_ctrl": ["-E27_0", "-E27_1", "E6_0", "E6_1"],
            "lanes_ew_ctrl": ["-E28_0", "E26_0"]
        },
        "J6": {
            "lanes_ns_ctrl": ["E5_0", "E5_1", "-E13_0", "-E13_1"],
            "lanes_ew_ctrl": ["-E12_0", "-E14_0"]
        },
        "J14": {
            "lanes_ns_ctrl": ["E18_0", "E18_1", "-E21_0", "-E21_1"],
            "lanes_ew_ctrl": ["E15_0", "-E20_0"]
        },
        "J17": {
            "lanes_ns_ctrl": ["-E22_0", "-E22_1", "E19_0", "E19_1"],
            "lanes_ew_ctrl": ["-E24_0", "-E23_0"]
        }
    }
    
    print("=" * 70)
    print("EXTRACTING LANE DIRECTIONS FROM NETWORK GEOMETRY")
    print("=" * 70)
    
    lane_directions = extract_lane_directions(net_file)
    
    print(f"\nTotal lanes in network: {len(lane_directions)}")
    print("\nSample lane directions:")
    sample_lanes = ["-E3_0", "-E2_0", "-E1_0", "-E0_0", "E0_0", "E1_0"]
    for lane in sample_lanes:
        if lane in lane_directions:
            info = lane_directions[lane]
            print(f"  {lane}: direction={info['direction']}, angle={info['angle']}°, to={info['to_junction']}")
    
    print("\n" + "=" * 70)
    print("GENERATED APPROACH_LANES CONFIG")
    print("=" * 70)
    
    approach_lanes = generate_approach_lanes_config(net_file, lane_groups_by_tls)
    
    for tls_id, directions in approach_lanes.items():
        print(f"\n  {tls_id}:")
        for dir_key in ["N", "E", "S", "W"]:
            lanes = directions[dir_key]
            print(f"    {dir_key}: {lanes}")
    
    output_file = "configs/approach_lanes_generated.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(approach_lanes, f, indent=2)
    print(f"\n[OK] Saved to {output_file}")
    
    print("\n" + "=" * 70)
    print("YAML CONFIG SNIPPET (copy to train_bignet_short.yaml)")
    print("=" * 70)
    for tls_id, directions in approach_lanes.items():
        print(f"      {tls_id}:")
        for dir_key in ["N", "E", "S", "W"]:
            lanes = directions[dir_key]
            if lanes:
                lanes_str = ", ".join(f'"{l}"' for l in lanes)
                print(f"        {dir_key}: [{lanes_str}]")
            else:
                print(f"        {dir_key}: []")

if __name__ == "__main__":
    main()
