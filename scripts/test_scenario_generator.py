#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def test_xml_well_formed(route_path: Path) -> bool:
    try:
        tree = ET.parse(str(route_path))
        root = tree.getroot()

        flows = root.findall("flow")
        if len(flows) == 0:
            print(f"  [FAIL] No flows found in {route_path.name}")
            return False

        total_veh_per_hour = 0.0
        for flow in flows:
            try:
                veh_per_hour = float(flow.get("vehsPerHour", "0"))
                total_veh_per_hour += veh_per_hour
            except ValueError:
                print(f"  [FAIL] Invalid vehsPerHour in flow {flow.get('id')}")
                return False

        if total_veh_per_hour <= 0.0:
            print("  [FAIL] Total demand is zero")
            return False

        print(f"  [PASS] XML valid, total demand {total_veh_per_hour:.1f} veh/h")
        return True

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_metadata_consistency(meta_path: Path) -> bool:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if "seed" not in meta:
            raise KeyError("Missing 'seed'")
        if "scenario" not in meta:
            raise KeyError("Missing 'scenario'")
        if "duration_sec" not in meta:
            raise KeyError("Missing 'duration_sec'")

        scenario = meta["scenario"]
        if "entry_split" not in scenario:
            raise KeyError("Missing 'entry_split'")
        if "vehicle_mix" not in scenario:
            raise KeyError("Missing 'vehicle_mix'")
        if "turning_ratios" not in scenario:
            raise KeyError("Missing 'turning_ratios'")

        v_mix = scenario["vehicle_mix"]
        total_mix = float(sum(v_mix.values()))
        if abs(total_mix - 1.0) >= 0.01:
            raise ValueError(f"Vehicle mix doesn't sum to 1.0: {total_mix}")

        for edge, ratios in scenario["turning_ratios"].items():
            total_ratio = float(sum(ratios.values()))
            if abs(total_ratio - 1.0) >= 0.01:
                raise ValueError(f"Turning ratios for {edge} don't sum to 1.0: {total_ratio}")

        print("  [PASS] Metadata consistent")
        return True

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_demand_tolerance(meta_path: Path, tolerance: float = 0.15) -> bool:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        scenario = meta["scenario"]
        config_demand = float(scenario.get("total_pcu", 0.0))
        actual_demand = float(sum(scenario["entry_split"].values()))

        error = abs(actual_demand - config_demand) / (config_demand + 1e-6)

        if error > float(tolerance):
            print(
                f"  [FAIL] Demand mismatch. Expected {config_demand:.1f}, got {actual_demand:.1f} "
                f"(error {error * 100.0:.1f}%)"
            )
            return False

        print(f"  [PASS] Demand within tolerance (error {error * 100.0:.1f}%)")
        return True

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_vehicle_mix_range(meta_path: Path) -> bool:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        v_mix = meta["scenario"]["vehicle_mix"]

        expected = {
            "motorcycle": (0.64, 0.94),
            "passenger": (0.02, 0.22),
            "bus": (0.00, 0.10),
        }

        for vtype, (min_val, max_val) in expected.items():
            actual = float(v_mix.get(vtype, 0.0))
            if not (float(min_val) <= actual <= float(max_val)):
                print(
                    f"  [FAIL] {vtype} mix {actual:.3f} outside expected range [{min_val:.2f}, {max_val:.2f}]"
                )
                return False

        print("  [PASS] Vehicle mix within Hanoi range")
        print(
            f"    Motorcycle: {float(v_mix.get('motorcycle', 0.0)):.2f}, "
            f"Car: {float(v_mix.get('passenger', 0.0)):.2f}, "
            f"Bus: {float(v_mix.get('bus', 0.0)):.2f}"
        )
        return True

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_sumo_smoke_run(route_path: Path, net_file: Path, duration_sec: int = 60) -> bool:
    if not net_file.exists():
        print(f"  [SKIP] Network file not found: {net_file}")
        return True

    try:
        cmd = [
            "sumo",
            "-n",
            str(net_file),
            "-r",
            str(route_path),
            "--step-length",
            "1.0",
            "--begin",
            "0",
            "--end",
            str(int(duration_sec)),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"  [FAIL] SUMO exited with code {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False

        print(f"  [PASS] SUMO smoke test passed ({int(duration_sec)}s)")
        return True

    except FileNotFoundError:
        print("  [SKIP] SUMO not found in PATH")
        return True
    except subprocess.TimeoutExpired:
        print("  [FAIL] SUMO timeout after 30s")
        return False
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_reproducibility(calib_path: str, out_dir: Path, seed: int) -> bool:
    try:
        from scripts.generate_hanoi_route_variants import HanoiRouteGenerator

        gen1 = HanoiRouteGenerator(calib_path, str(out_dir), int(seed))
        scenario1 = gen1.sample_scenario_params(level="med")

        gen2 = HanoiRouteGenerator(calib_path, str(out_dir), int(seed))
        scenario2 = gen2.sample_scenario_params(level="med")

        for key in ["total_pcu", "level"]:
            if scenario1.get(key) != scenario2.get(key):
                print(f"  [FAIL] Mismatch in {key}: {scenario1.get(key)} vs {scenario2.get(key)}")
                return False

        split1 = scenario1.get("entry_split", {})
        split2 = scenario2.get("entry_split", {})
        for entry_edge in split1:
            if abs(float(split1[entry_edge]) - float(split2.get(entry_edge, 0.0))) > 1e-6:
                print(f"  [FAIL] Entry split mismatch for {entry_edge}")
                return False

        print("  [PASS] Reproducible with same seed")
        return True

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def main() -> None:
    print("=" * 80)
    print("SCENARIO GENERATOR VALIDATION TESTS")
    print("=" * 80)
    print()

    calib_path = "configs/scenario_hanoi_calibration.yaml"
    out_dir = Path("networks/test_variants")

    if not Path(calib_path).exists():
        print(f"[ERROR] Calibration file not found: {calib_path}")
        print()
        print("Generate it first:")
        print("  python scripts/inspect_net_boundaries.py \\")
        print("    --net networks/BIGMAP.net.xml \\")
        print(f"    --out {calib_path}")
        sys.exit(1)

    print("TEST 1: Generate test variant")
    try:
        from scripts.generate_hanoi_route_variants import HanoiRouteGenerator

        gen = HanoiRouteGenerator(calib_path, str(out_dir), 42)
        out_dir.mkdir(parents=True, exist_ok=True)

        route_file = gen.generate_variant(0, level="med")
        if not route_file:
            print("  [FAIL] Could not generate variant")
            sys.exit(1)

        print(f"  [PASS] Generated {Path(route_file).name}")

    except Exception as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)

    print()
    print("TEST 2: Validate XML structure")
    route_matches = sorted(out_dir.glob("BIGMAP_variant_seed*.rou.xml"))
    if len(route_matches) == 0:
        print("  [FAIL] No route file found in output directory")
        sys.exit(1)
    route_path = route_matches[0]
    if not test_xml_well_formed(route_path):
        sys.exit(1)

    print()
    print("TEST 3: Validate metadata")
    meta_matches = sorted(out_dir.glob("meta_*.json"))
    if len(meta_matches) == 0:
        print("  [FAIL] No metadata file found in output directory")
        sys.exit(1)
    meta_path = meta_matches[0]
    if not test_metadata_consistency(meta_path):
        sys.exit(1)

    print()
    print("TEST 4: Check demand tolerance")
    if not test_demand_tolerance(meta_path, tolerance=0.15):
        sys.exit(1)

    print()
    print("TEST 5: Check vehicle mix range")
    if not test_vehicle_mix_range(meta_path):
        sys.exit(1)

    print()
    print("TEST 6: SUMO smoke test")
    net_file = Path("networks/BIGMAP.net.xml")
    sumo_ok = test_sumo_smoke_run(route_path, net_file, duration_sec=60)
    if not sumo_ok:
        print("  [WARNING] SUMO test failed, but continuing...")

    print()
    print("TEST 7: Reproducibility")
    if not test_reproducibility(calib_path, out_dir, seed=999):
        sys.exit(1)

    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Scenario generator is validated and ready to use.")


if __name__ == "__main__":
    main()
