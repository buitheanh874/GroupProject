from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import xml.etree.ElementTree as ET

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def test_route_file_validity(route_path: Path) -> bool:
    try:
        tree = ET.parse(str(route_path))
        root = tree.getroot()
        
        flows = root.findall("flow")
        if len(flows) == 0:
            print(f"  FAIL: No flows found in {route_path.name}")
            return False
        
        total_veh_per_hour = 0.0
        for flow in flows:
            try:
                veh_per_hour = float(flow.get("vehsPerHour", "0"))
                total_veh_per_hour += veh_per_hour
            except ValueError:
                print(f"  FAIL: Invalid vehsPerHour in flow {flow.get('id')}")
                return False
        
        if total_veh_per_hour <= 0:
            print(f"  FAIL: Total demand is zero")
            return False
        
        print(f"  PASS: XML valid, total demand {total_veh_per_hour:.1f} veh/h")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_metadata_consistency(meta_path: Path, route_path: Path) -> bool:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        assert "seed" in meta, "Missing 'seed' in metadata"
        assert "scenario" in meta, "Missing 'scenario' in metadata"
        assert "duration_sec" in meta, "Missing 'duration_sec' in metadata"
        
        scenario = meta["scenario"]
        assert "entry_split" in scenario, "Missing 'entry_split' in scenario"
        assert "vehicle_mix" in scenario, "Missing 'vehicle_mix' in scenario"
        assert "turning_ratios" in scenario, "Missing 'turning_ratios' in scenario"
        
        v_mix = scenario["vehicle_mix"]
        total_mix = sum(v_mix.values())
        assert abs(total_mix - 1.0) < 0.01, f"Vehicle mix doesn't sum to 1.0: {total_mix}"
        
        for edge, ratios in scenario["turning_ratios"].items():
            total_ratio = sum(ratios.values())
            assert abs(total_ratio - 1.0) < 0.01, f"Turning ratios for {edge} don't sum to 1.0"
        
        print(f"  PASS: Metadata consistent")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_demand_tolerance(meta_path: Path, tolerance: float = 0.10) -> bool:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        scenario = meta["scenario"]
        config_demand = scenario.get("total_pcu", 0.0)
        actual_demand = sum(scenario["entry_split"].values())
        
        error = abs(actual_demand - config_demand) / (config_demand + 1e-6)
        
        if error > tolerance:
            print(f"  FAIL: Demand mismatch. Expected {config_demand:.1f}, got {actual_demand:.1f} (error {error*100:.1f}%)")
            return False
        
        print(f"  PASS: Demand within tolerance ({error*100:.1f}%)")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def test_reproducibility(calib_path: str, out_dir: Path, seed: int) -> bool:
    try:
        from scripts.generate_hanoi_route_variants import HanoiRouteGenerator
        
        gen1 = HanoiRouteGenerator(calib_path, str(out_dir), seed)
        scenario1 = gen1.sample_scenario_params(level="med")
        
        gen2 = HanoiRouteGenerator(calib_path, str(out_dir), seed)
        scenario2 = gen2.sample_scenario_params(level="med")
        
        assert scenario1 == scenario2, "Scenarios don't match with same seed"
        
        print(f"  PASS: Reproducible with same seed")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main() -> None:
    print("=" * 80)
    print("SCENARIO GENERATOR TESTS")
    print("=" * 80)
    print()

    calib_path = "configs/scenario_hanoi_calibration.yaml"
    out_dir = Path("networks/test_variants")
    
    if not Path(calib_path).exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        sys.exit(1)
    
    print("[TEST 1] Generate test variants")
    try:
        from scripts.generate_hanoi_route_variants import HanoiRouteGenerator
        
        gen = HanoiRouteGenerator(calib_path, str(out_dir), 42)
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        route_file = gen.generate_variant(0, level="med")
        if not route_file:
            print("  FAIL: Could not generate variant")
            sys.exit(1)
        
        print(f"  PASS: Generated {Path(route_file).name}")
        
    except Exception as exc:
        print(f"  FAIL: {exc}")
        sys.exit(1)
    
    print()
    print("[TEST 2] Validate route file")
    route_path = list(out_dir.glob("BIGMAP_variant_seed*.rou.xml"))[0]
    if not test_route_file_validity(route_path):
        sys.exit(1)
    
    print()
    print("[TEST 3] Validate metadata")
    meta_path = list(out_dir.glob("meta_*.json"))[0]
    if not test_metadata_consistency(meta_path, route_path):
        sys.exit(1)
    
    print()
    print("[TEST 4] Check demand tolerance")
    if not test_demand_tolerance(meta_path, tolerance=0.15):
        sys.exit(1)
    
    print()
    print("[TEST 5] Reproducibility")
    if not test_reproducibility(calib_path, out_dir, 42):
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()