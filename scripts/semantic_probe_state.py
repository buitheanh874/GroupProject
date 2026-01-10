from __future__ import annotations

import argparse
import csv
import math
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

script_dir = Path(__file__).resolve().parent
project_root_hint = script_dir.parent
if str(project_root_hint) not in sys.path:
    sys.path.insert(0, str(project_root_hint))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from scripts.repo_root import find_repo_root

project_root = find_repo_root(__file__)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.utils import ensure_dir, load_yaml_config, set_global_seed
from scripts.common import build_env
from scripts.config_normalization import normalize_action_table_schema
from scripts.route_pool_loader import load_route_pool_from_config, resolve_route_file_if_manifest
from scripts.scenario_config_bridge import apply_calibration_overrides


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic probe using MDP state response to forced NS/EW phases.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--warmup_steps", type=int, default=60)
    parser.add_argument("--baseline_steps", type=int, default=30)
    parser.add_argument("--hold_steps", type=int, default=40)
    parser.add_argument("--min_baseline_queue", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args(argv)


def _get_lane_groups(env: Any, tls_id: str) -> Tuple[List[str], List[str], str]:
    lanes_by_tls = getattr(env, "_lanes_by_tls", {})
    if tls_id in lanes_by_tls:
        group = lanes_by_tls[tls_id]
        ns_lanes = list(getattr(group, "lanes_ns_ctrl", []))
        ew_lanes = list(getattr(group, "lanes_ew_ctrl", []))
        if ns_lanes or ew_lanes:
            return ns_lanes, ew_lanes, "lanes_by_tls"

    direction_lanes = getattr(env, "_direction_lanes_by_tls", {}).get(tls_id, {})
    if direction_lanes:
        ns_lanes = list(direction_lanes.get("N", [])) + list(direction_lanes.get("S", []))
        ew_lanes = list(direction_lanes.get("E", [])) + list(direction_lanes.get("W", []))
        if ns_lanes or ew_lanes:
            return ns_lanes, ew_lanes, "direction_lanes"

    return [], [], "missing_lane_groups"


def _snapshot_from_lanes(traci: Any, ns_lanes: List[str], ew_lanes: List[str]) -> Dict[str, float]:
    halting_ns = 0.0
    halting_ew = 0.0
    waiting_ns = 0.0
    waiting_ew = 0.0
    veh_ns = 0.0
    veh_ew = 0.0

    for lane_id in ns_lanes:
        try:
            halting_ns += float(traci.lane.getLastStepHaltingNumber(str(lane_id)))
            waiting_ns += float(traci.lane.getWaitingTime(str(lane_id)))
            veh_ns += float(traci.lane.getLastStepVehicleNumber(str(lane_id)))
        except Exception:
            continue

    for lane_id in ew_lanes:
        try:
            halting_ew += float(traci.lane.getLastStepHaltingNumber(str(lane_id)))
            waiting_ew += float(traci.lane.getWaitingTime(str(lane_id)))
            veh_ew += float(traci.lane.getLastStepVehicleNumber(str(lane_id)))
        except Exception:
            continue

    try:
        vehicle_count_total = float(traci.vehicle.getIDCount())
    except Exception:
        vehicle_count_total = 0.0

    return {
        "halting_ns": halting_ns,
        "halting_ew": halting_ew,
        "waiting_ns": waiting_ns,
        "waiting_ew": waiting_ew,
        "veh_ns": veh_ns,
        "veh_ew": veh_ew,
        "vehicle_count": vehicle_count_total,
    }


def _set_phase(traci: Any, env: Any, tls_id: str, phase_index: int, hold_steps: int) -> None:
    if hasattr(env, "_set_phase"):
        env._set_phase(tls_id=tls_id, phase_index=int(phase_index), hold_steps=int(hold_steps))
    else:
        traci.trafficlight.setPhase(str(tls_id), int(phase_index))
        traci.trafficlight.setPhaseDuration(str(tls_id), float(hold_steps))


def _get_tls_ids(env: Any) -> List[str]:
    tls_ids = sorted(getattr(env, "_tls_ids", []))
    if len(tls_ids) == 0 and hasattr(env, "center_tls_id"):
        tls_ids = [str(getattr(env, "center_tls_id"))]
    return tls_ids


def _measure_window(
    traci: Any,
    ns_lanes: List[str],
    ew_lanes: List[str],
    steps: int,
    warmup: int = 0,
) -> Dict[str, float]:
    for _ in range(max(0, int(warmup))):
        traci.simulationStep()

    sums = {
        "halting_ns": 0.0,
        "halting_ew": 0.0,
        "waiting_ns": 0.0,
        "waiting_ew": 0.0,
        "veh_ns": 0.0,
        "veh_ew": 0.0,
        "vehicle_count": 0.0,
    }
    count = max(1, int(steps))

    for _ in range(count):
        traci.simulationStep()
        snapshot = _snapshot_from_lanes(traci, ns_lanes, ew_lanes)
        for key in sums:
            sums[key] += snapshot.get(key, 0.0)

    return {k: v / float(count) for k, v in sums.items()}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def run_probe(
    config_path: str,
    out_dir: Path,
    warmup_steps: int,
    baseline_steps: int,
    hold_steps: int,
    min_baseline_queue: float,
    threshold: float,
    repeats: int,
) -> Dict[str, Any]:
    config = load_yaml_config(config_path)
    config = apply_calibration_overrides(config, project_root=project_root)
    config = normalize_action_table_schema(config)
    config.setdefault("env", {}).setdefault("sumo", {})
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = True
    load_route_pool_from_config(config, split="train", project_root=project_root)
    load_route_pool_from_config(config, split="eval", project_root=project_root)
    resolve_route_file_if_manifest(config, project_root)
    set_global_seed(int(config.get("run", {}).get("seed", 0)))

    tls_ids_config = sorted(config.get("env", {}).get("sumo", {}).get("tls_ids", []))
    rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    tls_seen: set[str] = set()
    vehicle_counts_all: List[float] = []

    for repeat in range(int(repeats)):
        env = None
        tls_ids_current = list(tls_ids_config)
        try:
            env = build_env(config)
            env.reset()
            traci = getattr(env, "_traci", None)
            if traci is None:
                raise RuntimeError("traci_missing")
            tls_ids_env = _get_tls_ids(env)
            if tls_ids_env:
                tls_ids_current = tls_ids_env
            tls_seen.update(tls_ids_current)
            phases = getattr(env, "_phases", None)
            if phases is None:
                for tls_id in tls_ids_current:
                    rows.append(_make_error_row(repeat, tls_id, "skipped", "missing_phase_mapping", ""))
                    status_counts["skipped"] = status_counts.get("skipped", 0) + 1
                continue
            


            for _ in range(max(0, int(warmup_steps))):
                traci.simulationStep()

            for tls_id in tls_ids_current:
                try:
                    ns_idx, ew_idx = env.get_ns_ew_phase_indices(tls_id)
                except AttributeError:
                    ns_idx = getattr(phases, "ns_green", None)
                    ew_idx = getattr(phases, "ew_green", None)

                if ns_idx is None or ew_idx is None:
                    rows.append(_make_error_row(repeat, tls_id, "skipped", "missing_phase_mapping", ""))
                    status_counts["skipped"] = status_counts.get("skipped", 0) + 1
                    continue

                ns_lanes, ew_lanes, lane_source = _get_lane_groups(env, tls_id)
                
                if not ns_lanes and not ew_lanes:
                    rows.append(_make_error_row(repeat, tls_id, "skipped", "missing_lane_groups", lane_source))
                    status_counts["skipped"] = status_counts.get("skipped", 0) + 1
                    continue

                baseline = _measure_window(traci, ns_lanes, ew_lanes, steps=baseline_steps, warmup=0)
                vehicle_counts_all.append(baseline["vehicle_count"])

                _set_phase(traci, env, tls_id, ns_idx, hold_steps)
                ns_hold = _measure_window(traci, ns_lanes, ew_lanes, steps=hold_steps, warmup=min(5, hold_steps // 4))

                _set_phase(traci, env, tls_id, ew_idx, hold_steps)
                ew_hold = _measure_window(traci, ns_lanes, ew_lanes, steps=hold_steps, warmup=min(5, hold_steps // 4))

                baseline_queue_proxy = baseline["halting_ns"] + baseline["halting_ew"]
                baseline_vehicle_proxy = baseline["vehicle_count"]

                imp_ns = baseline["halting_ns"] - ns_hold["halting_ns"]
                imp_ew = baseline["halting_ew"] - ew_hold["halting_ew"]
                wrong_ns = baseline["halting_ew"] - ns_hold["halting_ew"]
                wrong_ew = baseline["halting_ns"] - ew_hold["halting_ns"]

                status_val = ""
                reason = ""

                if baseline_queue_proxy < float(min_baseline_queue) and baseline_vehicle_proxy < 1.0:
                    status_val = "skipped"
                    reason = "baseline_queue_below_threshold_no_vehicles"
                elif baseline_queue_proxy < float(min_baseline_queue):
                    status_val = "ambiguous"
                    reason = "low_queue_signal"
                else:
                    consistent = imp_ns > float(threshold) and imp_ew > float(threshold)
                    inverted = wrong_ns > float(threshold) and wrong_ew > float(threshold)
                    if inverted and not consistent:
                        status_val = "inverted"
                        reason = "inverted_response"
                    elif consistent:
                        status_val = "consistent"
                        reason = ""
                    else:
                        status_val = "ambiguous"
                        reason = "low_delta"

                rows.append({
                    "run_id": repeat,
                    "tls_id": tls_id,
                    "status": status_val,
                    "reason": reason,
                    "error_msg": "",
                    "ns_phase_index": int(ns_idx),
                    "ew_phase_index": int(ew_idx),
                    "baseline_vehicle_count_avg": baseline["vehicle_count"],
                    "baseline_halting_ns_avg": baseline["halting_ns"],
                    "baseline_halting_ew_avg": baseline["halting_ew"],
                    "baseline_waiting_ns_avg": baseline["waiting_ns"],
                    "baseline_waiting_ew_avg": baseline["waiting_ew"],
                    "ns_hold_halting_ns_avg": ns_hold["halting_ns"],
                    "ns_hold_halting_ew_avg": ns_hold["halting_ew"],
                    "ew_hold_halting_ns_avg": ew_hold["halting_ns"],
                    "ew_hold_halting_ew_avg": ew_hold["halting_ew"],
                    "imp_ns": imp_ns,
                    "imp_ew": imp_ew,
                    "wrong_ns": wrong_ns,
                    "wrong_ew": wrong_ew,
                    "notes": f"lane_groups_source={lane_source}",
                })
                status_counts[status_val] = status_counts.get(status_val, 0) + 1

        except Exception as exc:
            reason = f"exception:{exc.__class__.__name__}"
            error_msg = str(exc)[:160]
            for tls_id in tls_ids_current:
                rows.append(_make_error_row(repeat, tls_id, "error", reason, error_msg))
                status_counts["error"] = status_counts.get("error", 0) + 1
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    ensure_dir(str(out_dir))
    
    # Create timestamped filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path_timestamped = Path(out_dir) / f"semantic_probe_state_{timestamp_str}.csv"
    csv_path_latest = Path(out_dir) / "semantic_probe_state_latest.csv"
    
    fieldnames = [
        "run_id",
        "tls_id",
        "status",
        "reason",
        "error_msg",
        "ns_phase_index",
        "ew_phase_index",
        "baseline_vehicle_count_avg",
        "baseline_halting_ns_avg",
        "baseline_halting_ew_avg",
        "baseline_waiting_ns_avg",
        "baseline_waiting_ew_avg",
        "ns_hold_halting_ns_avg",
        "ns_hold_halting_ew_avg",
        "ew_hold_halting_ns_avg",
        "ew_hold_halting_ew_avg",
        "imp_ns",
        "imp_ew",
        "wrong_ns",
        "wrong_ew",
        "notes",
    ]
    
    # Write timestamped CSV
    with open(csv_path_timestamped, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    # Copy to latest
    with open(csv_path_latest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    veh_min = min(vehicle_counts_all) if vehicle_counts_all else 0.0
    veh_mean = sum(vehicle_counts_all) / len(vehicle_counts_all) if vehicle_counts_all else 0.0
    veh_max = max(vehicle_counts_all) if vehicle_counts_all else 0.0

    # Get git commit if available
    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]
    except Exception:
        pass
    
    # Get resolved route file path
    route_file_resolved = config.get("env", {}).get("sumo", {}).get("route_file", "unknown")
    
    report_path_timestamped = Path(out_dir) / f"semantic_probe_state_{timestamp_str}.md"
    report_path_latest = Path(out_dir) / "semantic_probe_state_latest.md"
    
    tls_unique = sorted(tls_seen) if tls_seen else tls_ids_config
    tls_count = len(tls_unique)
    expected_rows = int(repeats) * tls_count
    written_rows = len(rows)
    status = "PASS" if status_counts.get("inverted", 0) == 0 and status_counts.get("error", 0) == 0 else "FAIL"
    consistent_tls = sorted({r["tls_id"] for r in rows if r.get("status") == "consistent"})
    inverted_tls = sorted({r["tls_id"] for r in rows if r.get("status") == "inverted"})
    ambiguous_tls = sorted({r["tls_id"] for r in rows if r.get("status") == "ambiguous"})
    skipped_tls = sorted({r["tls_id"] for r in rows if r.get("status") == "skipped"})
    error_tls = sorted({r["tls_id"] for r in rows if r.get("status") == "error"})
    all_skipped = written_rows > 0 and written_rows == status_counts.get("skipped", 0)

    reason_counts: Dict[str, int] = {}
    for r in rows:
        rsn = r.get("reason", "")
        if rsn:
            reason_counts[rsn] = reason_counts.get(rsn, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
    
    # Write report content
    def write_report(f):
        f.write("# Semantic Probe (MDP State) Report\n\n")
        f.write("## Metadata\n")
        f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Config**: `{config_path}`\n")
        f.write(f"- **Route file**: `{route_file_resolved}`\n")
        f.write(f"- **Git commit**: `{git_commit}`\n")
        f.write(f"- **Args**: warmup={warmup_steps}, baseline={baseline_steps}, hold={hold_steps}, min_baseline_queue={min_baseline_queue}\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Status**: {status}\n")
        f.write(f"- **TLS count**: {tls_count}\n")
        f.write(f"- **Repeats**: {repeats}\n")
        f.write(f"- **Expected rows**: {expected_rows}\n")
        f.write(f"- **Rows written**: {written_rows}\n")
        f.write(f"- **Status counts**: {status_counts}\n")
        f.write(f"- **Vehicle count (min/mean/max)**: {veh_min:.2f} / {veh_mean:.2f} / {veh_max:.2f}\n\n")
        
        f.write("## Results\n")
        f.write(f"- **Consistent TLS**: {consistent_tls}\n")
        f.write(f"- **Inverted TLS**: {inverted_tls}\n")
        f.write(f"- **Ambiguous TLS**: {ambiguous_tls}\n")
        f.write(f"- **Skipped TLS**: {skipped_tls}\n")
        f.write(f"- **Error TLS**: {error_tls}\n")
        f.write(f"- **Top reasons**: {top_reasons}\n\n")
        
        if all_skipped:
            if veh_mean < 1.0:
                f.write("**Note**: Semantic not verified (all skipped, vehicle count near zero)\n\n")
            else:
                f.write("**Note**: Semantic not verified (all skipped despite vehicles - possible bug)\n\n")
        
        f.write("## Output Files\n")
        f.write(f"- **CSV (timestamped)**: `{csv_path_timestamped}`\n")
        f.write(f"- **CSV (latest)**: `{csv_path_latest}`\n")
        f.write(f"- **Report (timestamped)**: `{report_path_timestamped}`\n")
        f.write(f"- **Report (latest)**: `{report_path_latest}`\n")
    
    # Write timestamped report
    with open(report_path_timestamped, "w", encoding="utf-8") as f:
        write_report(f)
    
    # Write latest report
    with open(report_path_latest, "w", encoding="utf-8") as f:
        write_report(f)

    return {
        "status_counts": status_counts,
        "expected_rows": expected_rows,
        "written_rows": written_rows,
        "status": status,
        "tls_count": tls_count,
        "tls_ids": tls_unique,
        "csv_path": str(csv_path_latest),
        "csv_path_timestamped": str(csv_path_timestamped),
        "report_path": str(report_path_latest),
        "report_path_timestamped": str(report_path_timestamped),
        "vehicle_count_min": veh_min,
        "vehicle_count_mean": veh_mean,
        "vehicle_count_max": veh_max,
    }


def _make_error_row(repeat: int, tls_id: str, status: str, reason: str, error_msg: str) -> Dict[str, Any]:
    return {
        "run_id": repeat,
        "tls_id": tls_id,
        "status": status,
        "reason": reason,
        "error_msg": error_msg,
        "ns_phase_index": "",
        "ew_phase_index": "",
        "baseline_vehicle_count_avg": float("nan"),
        "baseline_halting_ns_avg": float("nan"),
        "baseline_halting_ew_avg": float("nan"),
        "baseline_waiting_ns_avg": float("nan"),
        "baseline_waiting_ew_avg": float("nan"),
        "ns_hold_halting_ns_avg": float("nan"),
        "ns_hold_halting_ew_avg": float("nan"),
        "ew_hold_halting_ns_avg": float("nan"),
        "ew_hold_halting_ew_avg": float("nan"),
        "imp_ns": float("nan"),
        "imp_ew": float("nan"),
        "wrong_ns": float("nan"),
        "wrong_ew": float("nan"),
        "notes": "",
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    executed = False
    exec_error: Optional[str] = None
    summary: Dict[str, Any] = {}
    try:
        import traci
        summary = run_probe(
            config_path=args.config,
            out_dir=out_dir,
            warmup_steps=int(args.warmup_steps),
            baseline_steps=int(args.baseline_steps),
            hold_steps=int(args.hold_steps),
            min_baseline_queue=float(args.min_baseline_queue),
            threshold=float(args.threshold),
            repeats=int(args.repeats),
        )
        executed = True
    except Exception as exc:
        exec_error = str(exc)
        report_path = Path(out_dir) / "semantic_probe_state.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Semantic Probe (MDP State) Report\n")
            f.write("Status: NOT RUN\n")
            f.write(f"Reason: {exec_error}\n")
    finally:
        if executed:
            expected_rows = summary.get("expected_rows", 0)
            written_rows = summary.get("written_rows", 0)
            status_counts = summary.get("status_counts", {})
            veh_min = summary.get("vehicle_count_min", 0.0)
            veh_mean = summary.get("vehicle_count_mean", 0.0)
            veh_max = summary.get("vehicle_count_max", 0.0)
            print(f"[semantic-probe-state] expected_rows={expected_rows}, written_rows={written_rows}")
            print(f"[semantic-probe-state] vehicle_count_avg(min/mean/max)={veh_min:.2f}/{veh_mean:.2f}/{veh_max:.2f}")
            print(f"[semantic-probe-state] status_counts={status_counts}")
        else:
            print(f"[semantic-probe-state] expected_rows=0, written_rows=0")
            print(f"[semantic-probe-state] vehicle_count_avg(min/mean/max)=0.00/0.00/0.00")
            print(f"[semantic-probe-state] status_counts={{'error': 1}}")
        if executed:
            print("semantic_probe_state: completed")
        else:
            print(f"semantic_probe_state: not executed ({exec_error})")


if __name__ == "__main__":
    main()
