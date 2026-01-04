from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


class HanoiRouteGenerator:
    def __init__(self, calib_path: str, out_dir: str, seed: int):
        self.calib_path = Path(calib_path)
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.rng = np.random.default_rng(int(seed))
        
        self.load_calibration()
        
    def load_calibration(self) -> None:
        if not self.calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")
        
        with open(self.calib_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        scenario_cfg = self.config.get("scenario", {})
        
        self.net_file = scenario_cfg.get("net_file", "networks/BIGMAP.net.xml")
        self.entry_edges = scenario_cfg.get("entry_edges", [])
        self.exit_edges = scenario_cfg.get("exit_edges", [])
        
        if not self.entry_edges:
            raise ValueError("entry_edges not configured in calibration file")
        if not self.exit_edges:
            raise ValueError("exit_edges not configured in calibration file")
        
        self.vehicle_mix_mean = scenario_cfg.get("vehicle_mix_mean", {
            "motorcycle": 0.84,
            "passenger": 0.12,
            "bus": 0.03,
            "other": 0.01,
        })
        self.vehicle_mix_kappa = scenario_cfg.get("vehicle_mix_kappa", 50)
        
        self.pcu_weights = scenario_cfg.get("pcu_weights", {
            "motorcycle": 0.25,
            "passenger": 1.0,
            "bus": 3.0,
            "other": 1.0,
        })
        
        demand_cfg = scenario_cfg.get("demand", {})
        self.demand_levels = demand_cfg.get("total_pcu_per_hour", {
            "low": 3000,
            "med": 5000,
            "high": 7000,
        })
        self.entry_dirichlet_alpha = demand_cfg.get("entry_dirichlet_alpha", 2.0)
        
        turning_cfg = scenario_cfg.get("turning", {})
        self.turning_mean_LSR = turning_cfg.get("mean_LSR", [0.15, 0.70, 0.15])
        self.turning_kappa = turning_cfg.get("kappa", 30)
        self.turning_overrides = scenario_cfg.get("turning_overrides", {})
        
        sim_cfg = scenario_cfg.get("simulation", {})
        self.step_length_sec = sim_cfg.get("step_length_sec", 1.0)
        self.duration_sec = sim_cfg.get("duration_sec", 3600)
        self.min_total_vehicles = sim_cfg.get("min_total_vehicles", 100)
    
    def sample_scenario_params(self, level: str = "med") -> Dict[str, Any]:
        if level not in self.demand_levels:
            raise ValueError(f"Unknown demand level: {level}")
        
        total_pcu = float(self.demand_levels[level])
        
        entry_alpha = float(self.entry_dirichlet_alpha)
        K = len(self.entry_edges)
        pi = self.rng.dirichlet([entry_alpha] * K)
        Q_entry = {edge: float(total_pcu * pi[i]) for i, edge in enumerate(self.entry_edges)}
        
        v_mean = [
            self.vehicle_mix_mean.get("motorcycle", 0.84),
            self.vehicle_mix_mean.get("passenger", 0.12),
            self.vehicle_mix_mean.get("bus", 0.03),
            self.vehicle_mix_mean.get("other", 0.01),
        ]
        v_mix = self.rng.dirichlet([self.vehicle_mix_kappa * x for x in v_mean])
        
        turning_ratios = {}
        mu_LSR = self.turning_mean_LSR
        
        for entry_edge in self.entry_edges:
            if entry_edge in self.turning_overrides:
                mu = self.turning_overrides[entry_edge]
            else:
                mu = mu_LSR
            
            theta = self.rng.dirichlet([self.turning_kappa * x for x in mu])
            turning_ratios[entry_edge] = {
                "left": float(theta[0]),
                "straight": float(theta[1]),
                "right": float(theta[2]),
            }
        
        return {
            "level": level,
            "total_pcu": total_pcu,
            "entry_split": Q_entry,
            "vehicle_mix": {
                "motorcycle": float(v_mix[0]),
                "passenger": float(v_mix[1]),
                "bus": float(v_mix[2]),
                "other": float(v_mix[3]),
            },
            "turning_ratios": turning_ratios,
        }
    
    def convert_pcu_to_vehicles(self, Q_pcu: float, v_mix: Dict[str, float]) -> Dict[str, float]:
        w = self.pcu_weights
        result = {}
        
        for vtype in ["motorcycle", "passenger", "bus", "other"]:
            pcu_share = Q_pcu * v_mix.get(vtype, 0.0)
            w_vtype = w.get(vtype, 1.0)
            veh_per_hour = pcu_share / w_vtype
            result[vtype] = float(veh_per_hour)
        
        return result
    
    def generate_flows_xml(self, scenario: Dict[str, Any], output_path: Path) -> None:
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
        
        for vtype_cfg in vtypes:
            ET.SubElement(root, "vType", **vtype_cfg)
        
        entry_split = scenario.get("entry_split", {})
        v_mix = scenario.get("vehicle_mix", {})
        
        for entry_edge, Q_pcu in entry_split.items():
            veh_per_type = self.convert_pcu_to_vehicles(Q_pcu, v_mix)
            
            for vtype, veh_per_hour in veh_per_type.items():
                if veh_per_hour > 0.1:
                    flow = ET.SubElement(root, "flow")
                    flow.set("id", f"flow_{entry_edge}_{vtype}")
                    flow.set("from", str(entry_edge))
                    flow.set("begin", "0")
                    flow.set("end", str(int(self.duration_sec)))
                    flow.set("vehsPerHour", f"{veh_per_hour:.2f}")
                    flow.set("type", vtype)
                    flow.set("departLane", "best")
                    flow.set("departSpeed", "max")
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    
    def generate_turns_xml(self, scenario: Dict[str, Any], output_path: Path) -> None:
        root = ET.Element("turns")
        interval = ET.SubElement(root, "interval")
        interval.set("begin", "0")
        interval.set("end", str(int(self.duration_sec)))
        
        turning_ratios = scenario.get("turning_ratios", {})
        
        for entry_edge, ratios in turning_ratios.items():
            from_edge = ET.SubElement(interval, "fromEdge")
            from_edge.set("id", str(entry_edge))
            
            for exit_edge in self.exit_edges:
                to_edge = ET.SubElement(from_edge, "toEdge")
                to_edge.set("id", str(exit_edge))
                prob = 1.0 / len(self.exit_edges)
                to_edge.set("probability", f"{prob:.4f}")
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    
    def run_jtrrouter(self, flows_path: Path, turns_path: Path, route_path: Path) -> bool:
        net_path = Path(self.net_file)
        
        if not net_path.exists():
            print(f"Warning: Network file not found: {net_path}")
            return False
        
        cmd = [
            "jtrrouter",
            "-n", str(net_path),
            "-f", str(flows_path),
            "-t", str(turns_path),
            "-o", str(route_path),
            "--ignore-errors",
            "--accept-all-destinations",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.returncode == 0
        except FileNotFoundError:
            print("Warning: jtrrouter not found in PATH. Using fallback generation.")
            return False
    
    def fallback_route_generation(self, flows_path: Path, route_path: Path) -> None:
        tree = ET.parse(str(flows_path))
        root = tree.getroot()
        
        routes_root = ET.Element("routes")
        
        for vtype in root.findall("vType"):
            routes_root.append(vtype)
        
        for flow in root.findall("flow"):
            routes_root.append(flow)
        
        routes_tree = ET.ElementTree(routes_root)
        ET.indent(routes_tree, space="    ")
        routes_tree.write(route_path, encoding="UTF-8", xml_declaration=True)
    
    def validate_route_file(self, route_path: Path) -> bool:
        try:
            tree = ET.parse(str(route_path))
            root = tree.getroot()
            
            flows = root.findall("flow")
            total_veh_per_hour = 0.0
            
            for flow in flows:
                veh_per_hour = float(flow.get("vehsPerHour", "0"))
                total_veh_per_hour += veh_per_hour
            
            total_vehicles = total_veh_per_hour * (self.duration_sec / 3600.0)
            
            if total_vehicles < self.min_total_vehicles:
                print(f"Warning: Total vehicles ({total_vehicles:.0f}) < minimum ({self.min_total_vehicles})")
                return False
            
            return True
        except Exception as exc:
            print(f"Error validating route file: {exc}")
            return False
    
    def generate_variant(self, seed_idx: int, level: str = "med") -> Optional[str]:
        variant_seed = self.seed + seed_idx
        self.rng = np.random.default_rng(int(variant_seed))
        
        scenario = self.sample_scenario_params(level=level)
        
        temp_dir = self.out_dir / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        flows_path = temp_dir / f"flows_{variant_seed}.xml"
        turns_path = temp_dir / f"turns_{variant_seed}.xml"
        
        self.generate_flows_xml(scenario, flows_path)
        self.generate_turns_xml(scenario, turns_path)
        
        route_path = self.out_dir / f"BIGMAP_variant_seed{variant_seed:06d}.rou.xml"
        
        success = self.run_jtrrouter(flows_path, turns_path, route_path)
        
        if not success or not route_path.exists():
            self.fallback_route_generation(flows_path, route_path)
        
        if not self.validate_route_file(route_path):
            return None
        
        meta_path = self.out_dir / f"meta_{variant_seed:06d}.json"
        meta = {
            "seed": variant_seed,
            "scenario": scenario,
            "duration_sec": self.duration_sec,
            "min_total_vehicles": self.min_total_vehicles,
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        return str(route_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hanoi-style random route variants for training/evaluation"
    )
    parser.add_argument("--calib", required=True, help="Path to calibration YAML file")
    parser.add_argument("--out-dir", required=True, help="Output directory for route files")
    parser.add_argument("--split", choices=["train", "eval"], required=True, help="train or eval split")
    parser.add_argument("--n", type=int, required=True, help="Number of variants to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--level", choices=["low", "med", "high"], default="med", help="Demand level")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        base_seed = int(args.seed)
    else:
        base_seed = int(args.seed) + 100000

    print("=" * 80)
    print("HANOI ROUTE VARIANT GENERATOR")
    print("=" * 80)
    print(f"Calibration: {args.calib}")
    print(f"Split: {args.split}")
    print(f"Variants: {args.n}")
    print(f"Base seed: {base_seed}")
    print(f"Demand level: {args.level}")
    print(f"Output: {out_dir}")
    print()

    try:
        generator = HanoiRouteGenerator(args.calib, str(out_dir), base_seed)
    except Exception as exc:
        sys.exit(f"Failed to initialize generator: {exc}")

    generated_count = 0
    generated_files = []

    for idx in range(int(args.n)):
        print(f"[{idx+1}/{args.n}] Generating variant seed={base_seed+idx:06d}...", end=" ", flush=True)
        
        try:
            route_file = generator.generate_variant(idx, level=args.level)
            
            if route_file:
                file_size_kb = Path(route_file).stat().st_size / 1024.0
                print(f"OK ({file_size_kb:.1f} KB)")
                generated_files.append(route_file)
                generated_count += 1
            else:
                print("SKIP (validation failed)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print()
    print("=" * 80)
    print(f"Generated {generated_count}/{args.n} variants")
    print("=" * 80)
    print()

    if len(generated_files) > 0:
        manifest_path = out_dir / f"{args.split}_manifest.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for route_file in generated_files:
                f.write(f"{route_file}\n")
        
        print("Generated files:")
        for route_file in generated_files[:5]:
            print(f"  {Path(route_file).name}")
        
        if len(generated_files) > 5:
            print(f"  ... and {len(generated_files)-5} more")
        
        print()
        print(f"Manifest: {manifest_path}")
        print()


if __name__ == "__main__":
    main()