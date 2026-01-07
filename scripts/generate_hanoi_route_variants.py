from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from scripts.repo_root import find_repo_root

project_root = find_repo_root(__file__)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.hanoi_calibration import normalize_calibration_schema, validate_calibration
from scripts.hanoi_turns import build_turn_ratios_xml, resolve_turn_mapping

PROFILES = ["low", "high_balanced", "high_unbalanced", "ultimate", "auto"]


@dataclass
class FlowDef:
    flow_id: str
    from_edge: str
    to_edge: str
    veh_type: str
    vehs_per_hour: float
    begin: float
    end: float


def load_calibration(calib_path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
    normalized = normalize_calibration_schema(data)
    scenario = validate_calibration({"scenario": normalized})
    scenario["net_file"] = scenario.get("net_file", normalized.get("net_file", data.get("net_file", "")))
    scenario["map_prefix"] = scenario.get("map_prefix", normalized.get("map_prefix", "hanoi"))
    scenario.setdefault("horizon_sec", normalized.get("horizon_sec", 3600))
    return scenario


def select_seeds(split: str, n: int, base_seed: int, profile_offset: int = 0) -> List[int]:
    offset = 0 if split.lower() == "train" else 100000
    offset += profile_offset * 10000
    return [int(base_seed) + offset + i for i in range(int(n))]


def expand_entry_alpha(alpha: Any, num_entries: int) -> np.ndarray:
    if alpha is None:
        return np.ones(num_entries, dtype=np.float64)
    if isinstance(alpha, (int, float)):
        return np.full(num_entries, float(alpha), dtype=np.float64)
    values = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if values.size != num_entries:
        raise ValueError("entry_dirichlet_alpha length must match number of entry edges")
    return values


def sample_entry_split(entry_edges: List[str], alpha: np.ndarray, rng: np.random.Generator, enforce_unbalanced: bool) -> Dict[str, float]:
    sampled = rng.dirichlet(alpha)
    if enforce_unbalanced:
        idx = int(np.argmax(sampled))
        if sampled[idx] < 0.7:
            sampled[idx] = 0.7
            remaining = max(1e-9, 1.0 - sampled[idx])
            others = np.delete(sampled, idx)
            if others.size > 0:
                others = others / max(1e-9, float(np.sum(others)))
                others = others * remaining
                sampled = np.insert(others, idx, sampled[idx])
    return {edge: float(prob) for edge, prob in zip(entry_edges, sampled)}


def sample_vehicle_mix(mean: Dict[str, float], kappa: float, rng: np.random.Generator) -> Dict[str, float]:
    items = sorted(mean.items())
    base = np.asarray([max(float(v), 0.0) for _, v in items], dtype=np.float64)
    base = base / max(1e-9, base.sum())
    concentration = np.maximum(base * float(kappa), 1e-3)
    sampled = rng.dirichlet(concentration)
    return {name: float(prob) for (name, _), prob in zip(items, sampled)}


def sample_turning(mean_lsr: Sequence[float], kappa: float, rng: np.random.Generator, override: Optional[Sequence[float]] = None) -> Dict[str, float]:
    base = np.asarray(override if override is not None else mean_lsr, dtype=np.float64).reshape(-1)
    if base.size != 3:
        raise ValueError("turning ratios must have length 3")
    base = base / max(1e-9, base.sum())
    concentration = np.maximum(base * float(kappa), 1e-3)
    sampled = rng.dirichlet(concentration)
    labels = ["L", "S", "R"]
    return {lbl: float(val) for lbl, val in zip(labels, sampled)}


def build_level_plan(calib: Dict[str, Any], profile: str, rng: np.random.Generator) -> List[Dict[str, Any]]:
    demand_map = calib.get("demand", {}).get("total_pcu_per_hour", {})
    horizon = float(calib.get("horizon_sec", 3600))
    stage_cfg = calib.get("stages", {})
    enabled = bool(stage_cfg.get("enabled", False))
    intervals_cfg = stage_cfg.get("intervals", []) if enabled else []

    def level_from_profile() -> str:
        if profile == "low":
            return "low"
        if profile in {"high_balanced", "high_unbalanced", "ultimate"}:
            return "high"
        return "med" if "med" in demand_map else next(iter(demand_map.keys()))

    if enabled and len(intervals_cfg) > 0:
        plan: List[Dict[str, Any]] = []
        for item in intervals_cfg:
            begin = float(item.get("begin_sec", item.get("begin", 0.0)))
            end = float(item.get("end_sec", item.get("end", horizon)))
            lvl = str(item.get("level", level_from_profile()))
            total_pcu = float(demand_map.get(lvl, next(iter(demand_map.values()))))
            plan.append({"begin": begin, "end": end, "level": lvl, "pcu_per_hour": total_pcu})
        return plan

    level = level_from_profile()
    total_pcu = float(demand_map.get(level, next(iter(demand_map.values()))))
    return [{"begin": 0.0, "end": horizon, "level": level, "pcu_per_hour": total_pcu}]


def compute_vehicle_rates(total_pcu_per_hour: float, vehicle_mix: Dict[str, float], pcu_weights: Dict[str, float]) -> Dict[str, float]:
    weights = {k: float(pcu_weights.get(k, 1.0)) for k in vehicle_mix.keys()}
    denom = sum(vehicle_mix[k] * weights[k] for k in vehicle_mix.keys())
    if denom <= 0.0:
        raise ValueError("Vehicle mix/PCU weights produce zero denominator")
    veh_per_hour = float(total_pcu_per_hour) / float(denom)
    return {k: veh_per_hour * vehicle_mix[k] for k in vehicle_mix.keys()}


DEFAULT_VTYPES: List[Dict[str, str]] = [
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
    {
        "id": "other",
        "vClass": "passenger",
        "length": "4.5",
        "width": "1.8",
        "maxSpeed": "13.89",
        "accel": "2.5",
        "decel": "4.5",
        "sigma": "0.5",
        "minGap": "1.5",
    },
]


def build_flows(
    entry_edges: List[str],
    turn_map: Dict[str, Dict[str, List[str]]],
    entry_split: Dict[str, float],
    vehicle_rates: Dict[str, float],
    level_plan: List[Dict[str, Any]],
    turning_probs: Dict[str, Dict[str, float]],
) -> List[FlowDef]:
    flows: List[FlowDef] = []
    for interval_idx, interval in enumerate(level_plan):
        begin = float(interval["begin"])
        end = float(interval["end"])
        duration = max(0.0, end - begin)
        if duration <= 0.0:
            continue
        for entry_edge in entry_edges:
            entry_weight = float(entry_split.get(entry_edge, 0.0))
            if entry_weight <= 0.0:
                continue
            dir_probs = turning_probs.get(entry_edge, {})
            dir_map = turn_map.get(entry_edge, {})
            for dir_key, exits in dir_map.items():
                prob_dir = float(dir_probs.get(dir_key, 0.0))
                if prob_dir <= 0.0 or len(exits) == 0:
                    continue
                dest_weight = prob_dir / float(len(exits))
                for exit_edge in exits:
                    for veh_type, rate in vehicle_rates.items():
                        flow_rate = float(rate) * entry_weight * dest_weight
                        flow_id = f"{interval_idx}_{entry_edge}_{exit_edge}_{veh_type}"
                        flows.append(
                            FlowDef(
                                flow_id=flow_id,
                                from_edge=entry_edge,
                                to_edge=exit_edge,
                                veh_type=veh_type,
                                vehs_per_hour=flow_rate,
                                begin=begin,
                                end=end,
                            )
                        )
    return flows


def expected_vehicle_count(flows: Iterable[FlowDef]) -> float:
    total = 0.0
    for flow in flows:
        duration_hours = (float(flow.end) - float(flow.begin)) / 3600.0
        total += float(flow.vehs_per_hour) * duration_hours
    return total


def write_flows_xml(flow_defs: List[FlowDef], output_path: Path, vehicle_types: Optional[List[Dict[str, str]]] = None) -> None:
    import xml.etree.ElementTree as ET

    root = ET.Element("routes")
    vtypes = vehicle_types if vehicle_types is not None else DEFAULT_VTYPES
    seen_ids = set()
    for vt in sorted(vtypes, key=lambda item: str(item.get("id", ""))):
        vt_id = str(vt.get("id"))
        if vt_id in seen_ids:
            continue
        ET.SubElement(root, "vType", **{k: str(v) for k, v in vt.items()})
        seen_ids.add(vt_id)

    for flow in sorted(flow_defs, key=lambda f: str(f.flow_id)):
        attrs = {
            "id": str(flow.flow_id),
            "from": str(flow.from_edge),
            "to": str(flow.to_edge),
            "begin": f"{float(flow.begin):.1f}",
            "end": f"{float(flow.end):.1f}",
            "vehsPerHour": f"{float(flow.vehs_per_hour):.6f}",
            "type": str(flow.veh_type),
            "departLane": "best",
            "departSpeed": "max",
        }
        ET.SubElement(root, "flow", **attrs)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def find_router() -> Optional[str]:
    for binary in ["jtrrouter", "duarouter"]:
        path = shutil.which(binary)
        if path:
            return path
    return None


def run_router(router_bin: str, net_file: Path, trips_file: Path, output_route: Path, seed: int, step_length_sec: float = 1.0, turn_file: Optional[Path] = None) -> None:
    cmd = [
        router_bin,
        "--net-file",
        str(net_file),
    ]
    if "jtrrouter" in Path(router_bin).name:
        cmd.extend(["--flow-files", str(trips_file)])
        if turn_file:
            cmd.extend(["--turn-ratio-files", str(turn_file)])
    else:
        cmd.extend(["--route-files", str(trips_file)])
    cmd.extend(
        [
            "--output-file",
            str(output_route),
            "--seed",
            str(int(seed)),
            "--begin",
            "0",
            "--step-length",
            str(float(step_length_sec)),
            "--ignore-errors",
            "true",
        ]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Router failed (code {result.returncode}): {result.stderr}")


def write_manifest(manifest_path: Path, routes: List[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    manifest_dir = manifest_path.parent.resolve()
    for route in routes:
        route_path = Path(route).resolve()
        try:
            rel_path = route_path.relative_to(manifest_dir)
            lines.append(rel_path.as_posix())
        except ValueError:
            lines.append(route_path.as_posix())
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def generate_variant(
    calib: Dict[str, Any],
    profile: str,
    seed: int,
    turn_map: Dict[str, Dict[str, List[str]]],
    max_resamples: int = 5,
    enforce_min_total: bool = True,
) -> Tuple[List[FlowDef], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))

    entry_edges = [str(x) for x in calib.get("entry_edges", [])]
    exit_edges = [str(x) for x in calib.get("exit_edges", [])]
    pcu_weights = {str(k): float(v) for k, v in calib.get("pcu_weights", {}).items()}
    demand_map = calib.get("demand", {}).get("total_pcu_per_hour", {})
    entry_alpha_cfg = calib.get("demand", {}).get("entry_dirichlet_alpha", None)
    entry_alpha = expand_entry_alpha(entry_alpha_cfg, len(entry_edges))

    veh_mix_cfg = calib.get("vehicle_mix", {})
    veh_mean = {str(k): float(v) for k, v in veh_mix_cfg.get("mean", {}).items()}
    veh_kappa = float(veh_mix_cfg.get("kappa", 1.0))

    turning_cfg = calib.get("turning", {})
    mean_lsr = turning_cfg.get("mean_LSR", turning_cfg.get("mean_lsr", [0.15, 0.7, 0.15]))
    kappa_turn = float(turning_cfg.get("kappa", 10.0))
    overrides = {str(k): v for k, v in turning_cfg.get("turning_overrides", {}).items()}

    min_total = float(calib.get("min_total_vehicles", 0))

    for _ in range(max_resamples):
        entry_split = sample_entry_split(entry_edges, entry_alpha, rng, enforce_unbalanced=profile == "high_unbalanced")
        vehicle_mix = sample_vehicle_mix(veh_mean, veh_kappa, rng)
        level_plan = build_level_plan(calib, profile, rng)

        turning_probs: Dict[str, Dict[str, float]] = {}
        for edge in entry_edges:
            turning_probs[edge] = sample_turning(mean_lsr, kappa_turn, rng, overrides.get(edge))

        flows: List[FlowDef] = []
        for interval in level_plan:
            level = interval["level"]
            total_pcu = float(demand_map.get(level, next(iter(demand_map.values()))))
            veh_rates = compute_vehicle_rates(total_pcu, vehicle_mix, pcu_weights)
            flows.extend(build_flows(entry_edges, turn_map, entry_split, veh_rates, [interval], turning_probs=turning_probs))

        expected_total = expected_vehicle_count(flows)
        if not enforce_min_total or expected_total >= min_total:
            meta = {
                "seed": int(seed),
                "profile": profile,
                "entry_split": entry_split,
                "vehicle_mix": vehicle_mix,
                "turning": turning_probs,
                "level_plan": level_plan,
                "expected_total_vehicles": expected_total,
                "min_total_vehicles": min_total,
            }
            return flows, meta

    raise ValueError(f"Failed to satisfy min_total_vehicles={min_total} after {max_resamples} resamples")


def generate_routes(
    calib: Dict[str, Any],
    split: str,
    profile: str,
    seeds: List[int],
    out_dir: Path,
    skip_router: bool = False,
    router_bin: Optional[str] = None,
) -> List[Path]:
    map_prefix = str(calib.get("map_prefix", "hanoi"))
    net_file = Path(calib.get("net_file", "net.xml"))
    turn_map = resolve_turn_mapping(calib)
    routes: List[Path] = []

    for seed in seeds:
        flows, meta = generate_variant(calib, profile=profile, seed=seed, turn_map=turn_map)
        base_parts = [map_prefix]
        if profile and profile not in {"", "auto"}:
            base_parts.append(profile)
        base_parts.append(split)
        base_name = "_".join(base_parts) + f"_seed{int(seed):05d}"
        variant_dir = out_dir / split
        flows_path = variant_dir / f"flows_{base_name}.xml"
        turns_path = variant_dir / f"turns_{base_name}.xml"
        rou_path = variant_dir / f"{base_name}.rou.xml"
        meta_path = variant_dir / f"meta_{base_name}.json"

        write_flows_xml(flows, flows_path, vehicle_types=DEFAULT_VTYPES)

        turning_probs = meta.get("turning", {})
        end_time = max((interval.get("end", interval.get("end_sec", 0.0)) for interval in meta.get("level_plan", [])), default=calib.get("horizon_sec", 3600))
        turns_xml = build_turn_ratios_xml(turn_map, turning_probs, begin=0.0, end=float(end_time))
        turns_path.parent.mkdir(parents=True, exist_ok=True)
        turns_path.write_text(turns_xml, encoding="utf-8")

        meta["files"] = {
            "flows": str(flows_path),
            "turns": str(turns_path),
            "rou": str(rou_path),
        }
        meta["routed"] = False
        meta["router"] = None
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if not skip_router:
            router = router_bin or find_router()
            if not router:
                raise RuntimeError("SUMO router (jtrrouter/duarouter) not found. Install SUMO or rerun with --skip-router.")
            routed_path = rou_path
            run_router(
                router,
                net_file=net_file,
                trips_file=flows_path,
                output_route=routed_path,
                seed=seed,
                turn_file=turns_path,
            )
            meta["routed"] = True
            meta["router"] = Path(router).name
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            rou_path.write_text(flows_path.read_text(encoding="utf-8"), encoding="utf-8")

        routes.append(rou_path)
    return routes


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hanoi route variants (MDP style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--calib", required=True, help="Calibration YAML file")
    parser.add_argument("--out-dir", required=True, help="Output directory (e.g., networks/variants)")
    parser.add_argument("--split", choices=["train", "eval"], required=True, help="Data split")
    parser.add_argument("--n", type=int, default=10, help="Number of variants per profile")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--profile", choices=PROFILES, default=None, help="Traffic profile (omit for eval to generate all)")
    parser.add_argument("--router-binary", default=None, help="Optional router binary override")
    parser.add_argument("--skip-router", action="store_true", help="Skip router step (writes flows as .rou.xml)")
    args = parser.parse_args(argv)

    calib = load_calibration(Path(args.calib))
    out_dir = Path(args.out_dir)

    profiles: List[str]
    if args.profile:
        profiles = [args.profile]
    else:
        profiles = ["low", "high_balanced", "high_unbalanced", "ultimate"] if args.split == "eval" else ["auto"]

    manifest_paths: List[Path] = []
    for idx, profile in enumerate(profiles):
        seeds = select_seeds(args.split, args.n, args.seed, profile_offset=idx if len(profiles) > 1 else 0)
        routes = generate_routes(
            calib=calib,
            split=args.split,
            profile=profile,
            seeds=seeds,
            out_dir=out_dir,
            skip_router=bool(args.skip_router),
            router_bin=args.router_binary,
        )

        manifest_name = "manifest.txt" if len(profiles) == 1 else f"manifest_{profile}.txt"
        manifest_path = out_dir / args.split / manifest_name
        write_manifest(manifest_path, routes)
        manifest_paths.append(manifest_path)

    print(f"[INFO] Generated manifests: {[p.as_posix() for p in manifest_paths]}")


if __name__ == "__main__":
    main()
