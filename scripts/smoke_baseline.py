from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.kpi import EpisodeKpiTracker
from rl.utils import ensure_dir, generate_run_id, load_yaml_config, set_global_seed


def sec_to_steps(duration_sec: int, step_length: float) -> int:
    dur = float(int(duration_sec))
    step = float(step_length)
    if dur <= 0.0:
        return 0
    steps = int(math.ceil(dur / step))
    if steps <= 0:
        return 1
    return int(steps)


def build_sumo_command(cfg: Dict[str, Any], net_path: Path, route_path: Path, seed: int) -> List[str]:
    cmd: List[str] = [
        str(cfg.get("sumo_binary", "sumo")),
        "-n",
        str(net_path),
        "-r",
        str(route_path),
        "--step-length",
        str(float(cfg.get("step_length_sec", 1.0))),
        "--seed",
        str(int(seed)),
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
    ]

    additional = [str(x) for x in cfg.get("additional_files", [])]
    if len(additional) > 0:
        cmd.extend(["-a", ",".join(additional)])

    extra_args = cfg.get("sumo_extra_args", [])
    for arg in extra_args:
        cmd.append(str(arg))

    return cmd


def read_queue(traci_module: Any, lanes: List[str]) -> float:
    total = 0.0
    for lane_id in lanes:
        total += float(traci_module.lane.getLastStepHaltingNumber(str(lane_id)))
    return float(total)


def run_interval(
    traci_module: Any,
    tls_id: str,
    phase_index: int,
    duration_sec: int,
    step_length: float,
    tracker: Optional[EpisodeKpiTracker],
    tracker_state: Dict[str, bool],
    fallback: Dict[str, float],
    lanes_ns: List[str],
    lanes_ew: List[str],
) -> int:
    steps = sec_to_steps(int(duration_sec), float(step_length))
    if steps <= 0:
        return 0

    traci_module.trafficlight.setPhase(str(tls_id), int(phase_index))
    traci_module.trafficlight.setPhaseDuration(str(tls_id), float(steps * float(step_length)))

    stepped = 0
    dt = float(step_length)

    for _ in range(int(steps)):
        traci_module.simulationStep()

        q_ns = read_queue(traci_module, lanes_ns)
        q_ew = read_queue(traci_module, lanes_ew)
        q_total = float(q_ns + q_ew)

        fallback["queue_sum"] = float(fallback.get("queue_sum", 0.0)) + q_total
        fallback["queue_samples"] = float(fallback.get("queue_samples", 0.0)) + 1.0
        fallback["queue_time"] = float(fallback.get("queue_time", 0.0)) + (q_total * dt)

        try:
            arrived_ids = traci_module.simulation.getArrivedIDList()
            fallback["arrived_vehicles"] = float(fallback.get("arrived_vehicles", 0.0)) + float(len(arrived_ids))
        except Exception:
            pass

        if tracker is not None and bool(tracker_state.get("enabled", False)):
            try:
                try:
                    tracker.on_simulation_step(traci_module, queue_length=q_total)
                except TypeError:
                    tracker.on_simulation_step(traci_module)
            except Exception as exc:
                tracker_state["enabled"] = False
                if not bool(tracker_state.get("warned", False)):
                    print(f"[WARN] KPI tracker disabled due to TraCI error: {exc}")
                    tracker_state["warned"] = True

        stepped += 1

    return int(stepped)


def fixed_split(cycle_length_sec: int, rho_ns: float, rho_min: float) -> Tuple[int, int]:
    cycle_sec = int(cycle_length_sec)
    min_green_sec = int(round(float(rho_min) * float(cycle_sec)))
    if min_green_sec < 0:
        min_green_sec = 0

    g_ns = int(round(float(rho_ns) * float(cycle_sec)))
    g_ns = max(int(min_green_sec), int(g_ns))
    g_ns = min(int(g_ns), max(int(min_green_sec), int(cycle_sec - min_green_sec)))
    g_ew = int(cycle_sec - g_ns)
    return int(g_ns), int(g_ew)


def resolve_output_path(log_dir: str, run_id: str, explicit: Optional[str]) -> str:
    if explicit:
        return str(explicit)

    base = Path(log_dir) / f"{run_id}.csv"
    if not base.exists():
        return str(base)

    for i in range(1, 1000):
        candidate = Path(log_dir) / f"{run_id}_{i}.csv"
        if not candidate.exists():
            return str(candidate)

    return str(base)


def safe_tracker_summary(tracker: EpisodeKpiTracker) -> Dict[str, Any]:
    if hasattr(tracker, "summary_dict"):
        summary = tracker.summary_dict()
        if isinstance(summary, dict):
            return summary
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_sumo.yaml")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sumo-binary", default=None)
    parser.add_argument("--enable-vehicle-kpi", action="store_true")
    args = parser.parse_args()

    os.chdir(repo_root)

    set_global_seed(int(args.seed))

    config = load_yaml_config(args.config)
    env_cfg = config.get("env", {})
    sumo_cfg = env_cfg.get("sumo", {})

    sumo_cfg = dict(sumo_cfg)
    if args.sumo_binary:
        sumo_cfg["sumo_binary"] = str(args.sumo_binary)

    net_path = repo_root / str(sumo_cfg.get("net_file", ""))
    route_path = repo_root / str(sumo_cfg.get("route_file", ""))

    if not net_path.exists():
        sys.exit(f"Network file not found: {net_path}")
    if not route_path.exists():
        sys.exit(f"Route file not found: {route_path}")

    lane_cfg = sumo_cfg.get("lane_groups", {})
    lanes_ns = [str(x) for x in lane_cfg.get("lanes_ns_ctrl", [])]
    lanes_ew = [str(x) for x in lane_cfg.get("lanes_ew_ctrl", [])]

    if len(lanes_ns) == 0 or len(lanes_ew) == 0:
        sys.exit("Lane groups are missing for NS or EW.")

    phase_cfg = sumo_cfg.get("phase_program", {})
    ns_green = int(phase_cfg.get("ns_green", 0))
    ew_green = int(phase_cfg.get("ew_green", 1))
    ns_yellow = phase_cfg.get("ns_yellow")
    ew_yellow = phase_cfg.get("ew_yellow")
    all_red = phase_cfg.get("all_red")

    cycle_length_sec = int(sumo_cfg.get("cycle_length_sec", 60))
    yellow_sec = int(sumo_cfg.get("yellow_sec", 0))
    all_red_sec = int(sumo_cfg.get("all_red_sec", 0))
    step_length_sec = float(sumo_cfg.get("step_length_sec", 1.0))
    rho_min = float(sumo_cfg.get("rho_min", 0.1))

    try:
        import traci
    except Exception as exc:
        sys.exit(f"Failed to import traci: {exc}")

    sumo_seed = int(args.seed)
    command = build_sumo_command(sumo_cfg, net_path, route_path, sumo_seed)

    tls_id = str(sumo_cfg.get("tls_id", "tls0"))

    fallback: Dict[str, float] = {"queue_sum": 0.0, "queue_samples": 0.0, "queue_time": 0.0, "arrived_vehicles": 0.0}
    tracker_state: Dict[str, bool] = {"enabled": bool(args.enable_vehicle_kpi), "warned": False}

    tracker: Optional[EpisodeKpiTracker] = None
    if bool(tracker_state["enabled"]):
        try:
            tracker = EpisodeKpiTracker(stop_speed_threshold=0.1, use_subscription=False)
        except TypeError:
            tracker = EpisodeKpiTracker(stop_speed_threshold=0.1)

    total_steps = 0

    try:
        traci.start(command)

        g_ns, g_ew = fixed_split(cycle_length_sec=cycle_length_sec, rho_ns=0.5, rho_min=rho_min)

        for _ in range(int(args.cycles)):
            total_steps += run_interval(
                traci_module=traci,
                tls_id=tls_id,
                phase_index=int(ns_green),
                duration_sec=int(g_ns),
                step_length=step_length_sec,
                tracker=tracker,
                tracker_state=tracker_state,
                fallback=fallback,
                lanes_ns=lanes_ns,
                lanes_ew=lanes_ew,
            )

            if ns_yellow is not None and int(yellow_sec) > 0:
                total_steps += run_interval(
                    traci_module=traci,
                    tls_id=tls_id,
                    phase_index=int(ns_yellow),
                    duration_sec=int(yellow_sec),
                    step_length=step_length_sec,
                    tracker=tracker,
                    tracker_state=tracker_state,
                    fallback=fallback,
                    lanes_ns=lanes_ns,
                    lanes_ew=lanes_ew,
                )

            if all_red is not None and int(all_red_sec) > 0:
                total_steps += run_interval(
                    traci_module=traci,
                    tls_id=tls_id,
                    phase_index=int(all_red),
                    duration_sec=int(all_red_sec),
                    step_length=step_length_sec,
                    tracker=tracker,
                    tracker_state=tracker_state,
                    fallback=fallback,
                    lanes_ns=lanes_ns,
                    lanes_ew=lanes_ew,
                )

            total_steps += run_interval(
                traci_module=traci,
                tls_id=tls_id,
                phase_index=int(ew_green),
                duration_sec=int(g_ew),
                step_length=step_length_sec,
                tracker=tracker,
                tracker_state=tracker_state,
                fallback=fallback,
                lanes_ns=lanes_ns,
                lanes_ew=lanes_ew,
            )

            if ew_yellow is not None and int(yellow_sec) > 0:
                total_steps += run_interval(
                    traci_module=traci,
                    tls_id=tls_id,
                    phase_index=int(ew_yellow),
                    duration_sec=int(yellow_sec),
                    step_length=step_length_sec,
                    tracker=tracker,
                    tracker_state=tracker_state,
                    fallback=fallback,
                    lanes_ns=lanes_ns,
                    lanes_ew=lanes_ew,
                )

            if all_red is not None and int(all_red_sec) > 0:
                total_steps += run_interval(
                    traci_module=traci,
                    tls_id=tls_id,
                    phase_index=int(all_red),
                    duration_sec=int(all_red_sec),
                    step_length=step_length_sec,
                    tracker=tracker,
                    tracker_state=tracker_state,
                    fallback=fallback,
                    lanes_ns=lanes_ns,
                    lanes_ew=lanes_ew,
                )
    except Exception as exc:
        try:
            traci.close(False)
        except Exception:
            pass
        sys.exit(f"Simulation failed: {exc}")
    finally:
        try:
            traci.close(False)
        except Exception:
            pass

    queue_samples = float(fallback.get("queue_samples", 0.0))
    avg_queue_fallback = float(fallback.get("queue_sum", 0.0)) / queue_samples if queue_samples > 0.0 else float("nan")

    arrived_fallback = int(float(fallback.get("arrived_vehicles", 0.0)))
    avg_wait_fallback = float(fallback.get("queue_time", 0.0)) / float(arrived_fallback) if arrived_fallback > 0 else float(
        "nan"
    )

    summary: Dict[str, Any] = {}
    if tracker is not None and bool(tracker_state.get("enabled", False)):
        try:
            summary = safe_tracker_summary(tracker)
        except Exception as exc:
            tracker_state["enabled"] = False
            if not bool(tracker_state.get("warned", False)):
                print(f"[WARN] KPI summary failed, using fallback KPIs: {exc}")
                tracker_state["warned"] = True

    if bool(tracker_state.get("enabled", False)) and len(summary) > 0:
        avg_wait_time = float(summary.get("avg_wait_time", avg_wait_fallback))
        avg_queue = float(summary.get("avg_queue", avg_queue_fallback))
        arrived_vehicles = int(summary.get("arrived_vehicles", arrived_fallback))
        avg_travel_time = float(summary.get("avg_travel_time", float("nan")))
        avg_stops = float(summary.get("avg_stops", float("nan")))
    else:
        avg_wait_time = float(avg_wait_fallback)
        avg_queue = float(avg_queue_fallback)
        arrived_vehicles = int(arrived_fallback)
        avg_travel_time = float("nan")
        avg_stops = float("nan")

    log_dir = ensure_dir("logs")
    run_id = generate_run_id(prefix="smoke_baseline")
    output_path = resolve_output_path(log_dir=log_dir, run_id=run_id, explicit=args.output)

    fieldnames = [
        "run_id",
        "seed",
        "sumo_seed",
        "cycles",
        "cycle_length_sec",
        "step_length_sec",
        "total_steps",
        "avg_wait_time",
        "avg_travel_time",
        "avg_stops",
        "avg_queue",
        "arrived_vehicles",
    ]

    row = {
        "run_id": run_id,
        "seed": int(args.seed),
        "sumo_seed": int(sumo_seed),
        "cycles": int(args.cycles),
        "cycle_length_sec": int(cycle_length_sec),
        "step_length_sec": float(step_length_sec),
        "total_steps": int(total_steps),
        "avg_wait_time": float(avg_wait_time),
        "avg_travel_time": float(avg_travel_time),
        "avg_stops": float(avg_stops),
        "avg_queue": float(avg_queue),
        "arrived_vehicles": int(arrived_vehicles),
    }

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Baseline run complete. CSV saved to {output_path}")


if __name__ == "__main__":
    main()
