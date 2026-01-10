from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
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


@dataclass
class PhaseSample:
    sim_time: float
    tls_id: str
    phase_index: int
    phase_state: str
    controlled_links_len: int
    controlled_links_hash: str


@dataclass
class PhaseSignature:
    phase_index: int
    total_steps: int
    sample_state: str
    count_g: int
    count_G: int
    count_r: int
    count_y: int
    state_hash: str

    @property
    def green_weight(self) -> float:
        green_total = float(self.count_g + self.count_G)
        return green_total * float(self.total_steps)


def _hash_controlled_links(links: Any) -> Tuple[int, str]:
    try:
        flat = []
        for entry in links:
            for sub in entry:
                flat.append(str(sub))
        payload = "|".join(sorted(flat))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8] if payload else ""
        return len(links), digest
    except Exception:
        return 0, ""


def _collect_phase_samples(env: Any, steps: int) -> List[PhaseSample]:
    traci = getattr(env, "_traci", None)
    if traci is None:
        raise RuntimeError("SUMOEnv is missing _traci handle; ensure reset() was called and SUMO started.")

    tls_ids = sorted(getattr(env, "_tls_ids", []))
    if len(tls_ids) == 0 and hasattr(env, "center_tls_id"):
        tls_ids = [str(getattr(env, "center_tls_id"))]
    samples: List[PhaseSample] = []

    for _ in range(int(steps)):
        traci.simulationStep()
        sim_time = float(traci.simulation.getTime())
        for tls_id in tls_ids:
            phase_idx = int(traci.trafficlight.getPhase(tls_id))
            state = str(traci.trafficlight.getRedYellowGreenState(tls_id))
            ctrl_links_len, ctrl_links_hash = _hash_controlled_links(traci.trafficlight.getControlledLinks(tls_id))
            samples.append(
                PhaseSample(
                    sim_time=sim_time,
                    tls_id=str(tls_id),
                    phase_index=phase_idx,
                    phase_state=state,
                    controlled_links_len=int(ctrl_links_len),
                    controlled_links_hash=ctrl_links_hash,
                )
            )
    return samples


def _build_phase_signatures(samples: List[PhaseSample]) -> Dict[str, List[PhaseSignature]]:
    signatures: Dict[str, Dict[int, PhaseSignature]] = {}
    for sample in samples:
        tls_map = signatures.setdefault(sample.tls_id, {})
        sig = tls_map.get(sample.phase_index)
        if sig is None:
            counts = _count_state(sample.phase_state)
            tls_map[sample.phase_index] = PhaseSignature(
                phase_index=int(sample.phase_index),
                total_steps=1,
                sample_state=sample.phase_state,
                count_g=counts["g"],
                count_G=counts["G"],
                count_r=counts["r"],
                count_y=counts["y"],
                state_hash=_hash_state(sample.phase_state),
            )
        else:
            sig.total_steps += 1
    return {tls: list(sig_map.values()) for tls, sig_map in signatures.items()}


def _count_state(state: str) -> Dict[str, int]:
    return {
        "g": state.count("g"),
        "G": state.count("G"),
        "r": state.count("r"),
        "y": state.count("y") + state.count("Y"),
    }


def _hash_state(state: str) -> str:
    return hashlib.sha1(state.encode("utf-8")).hexdigest()[:8]


def _pick_main_greens(signatures: List[PhaseSignature]) -> List[int]:
    if len(signatures) == 0:
        return []
    ranked = sorted(signatures, key=lambda s: s.green_weight, reverse=True)
    top_two = [sig.phase_index for sig in ranked[:2]]
    return top_two


def _extract_main_green_order(samples_for_tls: List[PhaseSample], main_set: set[int]) -> List[int]:
    if len(samples_for_tls) == 0 or len(main_set) == 0:
        return []

    ordered = sorted(samples_for_tls, key=lambda s: s.sim_time)
    order: List[int] = []
    prev_phase: Optional[int] = None

    for sample in ordered:
        if prev_phase is None:
            prev_phase = sample.phase_index
        if sample.phase_index != prev_phase:
            prev_phase = sample.phase_index
            if sample.phase_index in main_set:
                if len(order) == 0 or order[-1] != sample.phase_index:
                    order.append(sample.phase_index)
                if len(order) >= 2 and len(set(order)) >= 2:
                    break
    return order


def _read_direction_halting(env: Any, tls_id: str) -> Dict[str, float]:
    traci = getattr(env, "_traci", None)
    if traci is None:
        return {k: 0.0 for k in ["N", "E", "S", "W"]}
    dirs = getattr(env, "_direction_lanes_by_tls", {}).get(tls_id, {})
    result: Dict[str, float] = {}
    for key in ["N", "E", "S", "W"]:
        total = 0.0
        for lane_id in dirs.get(key, []):
            try:
                total += float(traci.lane.getLastStepHaltingNumber(str(lane_id)))
            except Exception:
                continue
        result[key] = float(total)
    return result


def _force_phase_and_measure(env: Any, tls_id: str, phase_index: int, hold_steps: int, warmup_steps: int = 0) -> Dict[str, float]:
    traci = getattr(env, "_traci", None)
    if traci is None:
        return {k: 0.0 for k in ["N", "E", "S", "W"]}
    # Hold the phase for a short window to measure directional relief.
    if hasattr(env, "_set_phase"):
        env._set_phase(tls_id=tls_id, phase_index=int(phase_index), hold_steps=int(hold_steps))
    else:
        traci.trafficlight.setPhase(str(tls_id), int(phase_index))
        traci.trafficlight.setPhaseDuration(str(tls_id), float(hold_steps))

    for _ in range(max(0, int(warmup_steps))):
        traci.simulationStep()

    sums = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    for _ in range(int(hold_steps)):
        traci.simulationStep()
        counts = _read_direction_halting(env, tls_id)
        for k in sums:
            sums[k] += float(counts.get(k, 0.0))
    return {k: float(v) / float(max(1, hold_steps)) for k, v in sums.items()}


def _semantic_probe(env: Any, tls_id: str, hold_steps: int, warmup_steps: int = 3) -> Tuple[str, Dict[str, Any]]:
    phases = getattr(env, "_phases", None)
    if phases is None:
        return "unknown", {"reason": "phases_not_available"}

    dirs = getattr(env, "_direction_lanes_by_tls", {}).get(tls_id)
    if not dirs or all(len(v) == 0 for v in dirs.values()):
        return "skipped", {"reason": "direction_lanes_missing"}

    try:
        ns_idx, ew_idx = env.get_ns_ew_phase_indices(tls_id)
    except AttributeError:
        ns_idx = getattr(phases, "ns_green", None)
        ew_idx = getattr(phases, "ew_green", None)

    if ns_idx is None or ew_idx is None:
        return "unknown", {"reason": "phase_indices_missing"}

    traci = getattr(env, "_traci", None)
    if traci is None:
        return "unknown", {"reason": "traci_missing"}

    def _step_and_average(measure_steps: int, warmup: int = 0) -> Dict[str, float]:
        for _ in range(max(0, int(warmup))):
            traci.simulationStep()
        sums = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
        for _ in range(max(1, int(measure_steps))):
            traci.simulationStep()
            counts = _read_direction_halting(env, tls_id)
            for k in sums:
                sums[k] += float(counts.get(k, 0.0))
        return {k: float(v) / float(max(1, measure_steps)) for k, v in sums.items()}

    baseline_counts = _step_and_average(measure_steps=warmup_steps, warmup=max(0, warmup_steps))
    ns_counts = _force_phase_and_measure(env, tls_id, ns_idx, hold_steps=hold_steps, warmup_steps=warmup_steps)
    ew_counts = _force_phase_and_measure(env, tls_id, ew_idx, hold_steps=hold_steps, warmup_steps=warmup_steps)

    baseline_ns = baseline_counts["N"] + baseline_counts["S"]
    baseline_ew = baseline_counts["E"] + baseline_counts["W"]

    ns_phase_ns = ns_counts["N"] + ns_counts["S"]
    ns_phase_ew = ns_counts["E"] + ns_counts["W"]
    ew_phase_ns = ew_counts["N"] + ew_counts["S"]
    ew_phase_ew = ew_counts["E"] + ew_counts["W"]

    relief_ns_in_ns_phase = baseline_ns - ns_phase_ns
    relief_ns_in_ew_phase = baseline_ns - ew_phase_ns
    relief_ew_in_ew_phase = baseline_ew - ew_phase_ew
    relief_ew_in_ns_phase = baseline_ew - ns_phase_ew

    best_ns_phase = "ns" if relief_ns_in_ns_phase >= relief_ns_in_ew_phase else "ew"
    best_ew_phase = "ew" if relief_ew_in_ew_phase >= relief_ew_in_ns_phase else "ns"

    threshold = 0.1
    ns_ok = relief_ns_in_ns_phase > threshold
    ew_ok = relief_ew_in_ew_phase > threshold

    if best_ns_phase == "ns" and best_ew_phase == "ew" and ns_ok and ew_ok:
        status = "consistent"
    elif best_ns_phase == "ew" and best_ew_phase == "ns" and ns_ok and ew_ok:
        status = "inverted"
    else:
        status = "ambiguous"

    return status, {
        "baseline": baseline_counts,
        "ns_counts": ns_counts,
        "ew_counts": ew_counts,
        "relief_ns_in_ns_phase": relief_ns_in_ns_phase,
        "relief_ns_in_ew_phase": relief_ns_in_ew_phase,
        "relief_ew_in_ew_phase": relief_ew_in_ew_phase,
        "relief_ew_in_ns_phase": relief_ew_in_ns_phase,
        "best_ns_phase": best_ns_phase,
        "best_ew_phase": best_ew_phase,
        "ns_ok": ns_ok,
        "ew_ok": ew_ok,
    }


def run_check(config_path: str, steps: int, out_dir: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    config = load_yaml_config(config_path)
    config = apply_calibration_overrides(config, project_root=Path(config_path).resolve().parents[1])
    config = normalize_action_table_schema(config)
    if seed is not None:
        config.setdefault("run", {})["seed"] = int(seed)
    set_global_seed(int(config.get("run", {}).get("seed", 0)))

    project_root = Path(config_path).resolve().parents[1]
    load_route_pool_from_config(config, split="eval", project_root=project_root)
    resolve_route_file_if_manifest(config, project_root)

    env = None
    env = build_env(config)
    samples: List[PhaseSample] = []
    signatures: Dict[str, List[PhaseSignature]] = {}
    reference_tls: Optional[str] = None
    reference_main: List[int] = []
    semantic_results: Dict[str, Dict[str, Any]] = {}
    inverted_tls: List[str] = []
    ambiguous_tls: List[str] = []
    samples_by_tls: Dict[str, List[PhaseSample]] = {}
    ordering_unknown: List[str] = []
    ordering_mismatches: Dict[str, Dict[str, Any]] = {}
    skipped_semantic: Dict[str, str] = {}

    try:
        env.reset()

        samples = _collect_phase_samples(env, steps=steps)
        signatures = _build_phase_signatures(samples)

        for sample in samples:
            samples_by_tls.setdefault(sample.tls_id, []).append(sample)

        tls_ids = sorted(signatures.keys())
        reference_tls = tls_ids[0] if tls_ids else None
        reference_main = _pick_main_greens(signatures.get(reference_tls, [])) if reference_tls else []
        reference_main_set = set(reference_main)
        reference_order = _extract_main_green_order(samples_by_tls.get(reference_tls, []), reference_main_set) if reference_tls else []
        for tls_id in tls_ids:
            cand_main = _pick_main_greens(signatures.get(tls_id, []))
            cand_main_set = set(cand_main)
            cand_order = _extract_main_green_order(samples_by_tls.get(tls_id, []), cand_main_set)

            if len(reference_order) < 2 or len(cand_order) < 2:
                ordering_unknown.append(tls_id)
                continue
            if cand_order != reference_order:
                ordering_mismatches[tls_id] = {
                    "reference_main_set": list(reference_main_set),
                    "reference_order": reference_order,
                    "candidate_main_set": list(cand_main_set),
                    "candidate_order": cand_order,
                }

        for tls_id in tls_ids:
            status, meta = _semantic_probe(env, tls_id, hold_steps=max(5, int(steps * 0.05)), warmup_steps=3)
            semantic_results[tls_id] = {"status": status, **meta}
            if status == "inverted":
                inverted_tls.append(tls_id)
            if status == "ambiguous":
                ambiguous_tls.append(tls_id)
            if status == "skipped":
                skipped_semantic[tls_id] = str(meta.get("reason", "unknown"))
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    # Write CSV log for traceability.
    csv_path = out_dir / "phase_sync_log.csv"
    ensure_dir(str(out_dir))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sim_time",
                "tls_id",
                "phase_index",
                "phase_state",
                "controlled_links_len",
                "controlled_links_hash",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sim_time": sample.sim_time,
                    "tls_id": sample.tls_id,
                    "phase_index": sample.phase_index,
                    "phase_state": sample.phase_state,
                    "controlled_links_len": sample.controlled_links_len,
                    "controlled_links_hash": sample.controlled_links_hash,
                }
            )

    return {
        "signatures": signatures,
        "ordering_mismatches": ordering_mismatches,
        "ordering_unknown": ordering_unknown,
        "semantic_results": semantic_results,
        "inverted_tls": inverted_tls,
        "ambiguous_tls": ambiguous_tls,
        "skipped_semantic": skipped_semantic,
        "reference_tls": reference_tls,
        "reference_main_phases": reference_main,
        "reference_main_order": reference_order,
        "csv_path": str(csv_path),
    }


def _summarize_results(
    result: Dict[str, Any],
    require_semantic: bool = False,
    min_verified_fraction: float = 0.0,
) -> Dict[str, Any]:
    mismatches = result.get("ordering_mismatches", {})
    unknown = result.get("ordering_unknown", [])
    inverted = result.get("inverted_tls", [])
    ambiguous = result.get("ambiguous_tls", [])
    skipped_semantic: Dict[str, str] = result.get("skipped_semantic", {})
    semantic_results: Dict[str, Dict[str, Any]] = result.get("semantic_results", {})

    tls_count = len(semantic_results)
    # Verified = consistent (proven correct) TLS, not including inverted
    consistent_count = sum(1 for r in semantic_results.values() if r.get("status") == "consistent")
    semantic_verified_count = consistent_count
    semantic_verified_fraction = float(semantic_verified_count) / float(tls_count) if tls_count > 0 else 0.0

    # HARD failures = evidence-based issues that MUST be fixed
    hard_failures: List[str] = []
    # SOFT failures = coverage gate (optional gating, not hard evidence)
    soft_failures: List[str] = []
    # Warnings = ambiguous/skipped results that don't prove failure
    warnings: List[str] = []

    if tls_count == 0:
        hard_failures.append("no_tls_detected")
    if len(mismatches) > 0:
        hard_failures.append(f"ordering_mismatches: {sorted(mismatches.keys())}")
    if len(inverted) > 0:
        hard_failures.append(f"inverted_tls: {sorted(inverted)}")

    # Ambiguous/skipped are WARNINGS only, not failures
    if len(ambiguous) > 0:
        warnings.append(f"ambiguous_tls: {sorted(ambiguous)}")
    if len(skipped_semantic) > 0:
        warnings.append(f"skipped_semantic: {sorted(skipped_semantic.keys())}")
    if len(unknown) > 0:
        warnings.append(f"ordering_unknown: {sorted(unknown)}")

    # Determine status
    if len(hard_failures) > 0:
        status = "FAIL"
    elif require_semantic and semantic_verified_fraction < min_verified_fraction:
        # Coverage gate: this is a SOFT fail, not hard evidence
        status = "SOFT FAIL (coverage gate)"
        soft_failures.append(
            f"semantic_verified_fraction ({semantic_verified_fraction:.2f}) < min_verified_fraction ({min_verified_fraction:.2f})"
        )
    else:
        status = "PASS"

    return {
        "status": status,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "warnings": warnings,
        "semantic_verified_count": semantic_verified_count,
        "semantic_verified_fraction": semantic_verified_fraction,
        "consistent_count": consistent_count,
        "tls_count": tls_count,
        "ordering_mismatches": mismatches,
        "inverted_tls": inverted,
        "ambiguous_tls": ambiguous,
        "skipped_tls": list(skipped_semantic.keys()),
        "skipped_semantic": skipped_semantic,
        "ordering_unknown": unknown,
        "require_semantic": require_semantic,
        "min_verified_fraction": min_verified_fraction,
    }


def _write_report(
    out_dir: Path,
    summary: Dict[str, Any],
    result_meta: Dict[str, Any],
    config_path: str,
    executed: bool,
    exec_error: Optional[str],
) -> None:
    ensure_dir(str(out_dir))
    report_path = out_dir / "phase_sync_check_report.md"

    status = summary.get("status", "NOT RUN") if executed else "NOT RUN"
    hard_failures = summary.get("hard_failures", []) if executed else []
    soft_failures = summary.get("soft_failures", []) if executed else []
    warnings = summary.get("warnings", []) if executed else []
    tls_count = summary.get("tls_count", 0) if executed else 0
    semantic_verified_count = summary.get("semantic_verified_count", 0) if executed else 0
    semantic_verified_fraction = summary.get("semantic_verified_fraction", 0.0) if executed else 0.0
    require_semantic = summary.get("require_semantic", False) if executed else False
    min_verified_fraction = summary.get("min_verified_fraction", 0.0) if executed else 0.0

    ordering_mismatches = sorted(result_meta.get("ordering_mismatches", {}).keys()) if executed else []
    ordering_unknown = sorted(result_meta.get("ordering_unknown", [])) if executed else []
    inverted = sorted(result_meta.get("inverted_tls", [])) if executed else []
    ambiguous = sorted(result_meta.get("ambiguous_tls", [])) if executed else []
    skipped_semantic = result_meta.get("skipped_semantic", {}) if executed else {}

    exec_summary = (
        f"Phase sync check {status}."
        if executed
        else "Phase sync check not executed in this environment (SUMO/TraCI unavailable)."
    )

    recommendations = [
        "Swap NS/EW phase blocks in the network XML for affected TLS IDs, or",
        "Add per-TLS phase mapping in controller logic if swapping is not feasible.",
    ]
    recommendation_text = "- " + "\n- ".join(recommendations)

    how_to_run = f"python scripts/check_phase_sync.py --config {config_path} --steps 300 --out_dir reports"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase Sync Check Report\n")
        f.write("## Executive Summary\n")
        f.write(f"{exec_summary}\n")
        f.write("## What was checked\n")
        f.write("Phase ordering consistency across TLS; NS/EW action semantics via forced greens; controlled links shape.\n")
        f.write("## Methodology\n")
        f.write(
            "Collect per-step TLS phase/state via TraCI; build duration-weighted phase signatures; "
            "identify main greens; compare ordering across TLS; force NS/EW greens briefly and "
            "compare halting vehicles by direction to detect inversions.\n"
        )
        f.write("## Results\n")
        if executed:
            f.write(f"Status: {status}\n")
            ref_tls = result_meta.get("reference_tls") or "n/a"
            f.write(f"Reference TLS: {ref_tls}; main greens: {result_meta.get('reference_main_phases', [])}; main order: {result_meta.get('reference_main_order', [])}\n")
        else:
            f.write("Not executed; see Hard Failures for reason.\n")
        f.write("## Hard Failures\n")
        if executed and len(hard_failures) > 0:
            for hf in hard_failures:
                f.write(f"- {hf}\n")
        elif not executed:
            f.write(f"- Execution failed: {exec_error or 'unknown reason'}\n")
        else:
            f.write("None\n")
        f.write("## Soft Failures (Coverage Gate)\n")
        if executed and len(soft_failures) > 0:
            for sf in soft_failures:
                f.write(f"- {sf}\n")
        else:
            f.write("None\n")
        f.write("## Warnings\n")
        if executed and len(warnings) > 0:
            for w in warnings:
                f.write(f"- {w}\n")
        else:
            f.write("None\n")
        f.write("## Semantic Verification Summary\n")
        if executed:
            f.write(f"TLS count: {tls_count}\n")
            f.write(f"Verified count: {semantic_verified_count}\n")
            f.write(f"Verified fraction: {semantic_verified_fraction:.2f}\n")
            f.write(f"Ordering mismatches: {ordering_mismatches}\n")
            f.write(f"Ordering unknown: {ordering_unknown}\n")
            f.write(f"Inverted TLS: {inverted}\n")
            f.write(f"Ambiguous TLS: {ambiguous}\n")
            f.write(f"Skipped semantic: {sorted(skipped_semantic.items())}\n")
            f.write(f"require_semantic: {require_semantic}\n")
            f.write(f"min_verified_fraction: {min_verified_fraction}\n")
            f.write(f"CSV log: {result_meta.get('csv_path', 'n/a')}\n")
        else:
            f.write("Not executed.\n")
        f.write("## Recommendations\n")
        if len(inverted) > 0 or len(ordering_mismatches) > 0:
            f.write(f"{recommendation_text}\n")
        else:
            f.write("No action required.\n")
        f.write("## How to run\n")
        f.write(f"{how_to_run}\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TLS phase semantics consistency across the network.")
    parser.add_argument("--config", type=str, default="configs/train_1.yaml", help="Path to YAML config.")
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps to sample phases.")
    parser.add_argument("--out_dir", type=str, default="reports", help="Output directory for report and CSV.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--require_semantic",
        action="store_true",
        default=False,
        help="If set, FAIL when semantic_verified_fraction < min_verified_fraction.",
    )
    parser.add_argument(
        "--min_verified_fraction",
        type=float,
        default=0.0,
        help="Minimum verified fraction threshold (used only when --require_semantic is set).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    executed = False
    exec_error: Optional[str] = None
    summary: Dict[str, Any] = {"status": "NOT RUN", "hard_failures": [], "soft_failures": [], "warnings": []}
    result_meta: Dict[str, Any] = {}

    try:
        try:
            import traci  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency guard
            exec_error = f"TraCI/SUMO not available: {exc}"
            raise

        result_meta = run_check(config_path=args.config, steps=int(args.steps), out_dir=out_dir, seed=args.seed)
        summary = _summarize_results(
            result_meta,
            require_semantic=bool(args.require_semantic),
            min_verified_fraction=float(args.min_verified_fraction),
        )
        executed = True
    except Exception as exc:
        exec_error = str(exc)
        summary["hard_failures"].append(f"execution_failed: {exec_error}")
        summary["status"] = "FAIL"
    finally:
        _write_report(
            out_dir=out_dir,
            summary=summary,
            result_meta=result_meta,
            config_path=args.config,
            executed=executed,
            exec_error=exec_error,
        )

    status = summary.get("status", "NOT RUN")
    hard_count = len(summary.get("hard_failures", []))
    soft_count = len(summary.get("soft_failures", []))
    warn_count = len(summary.get("warnings", []))
    verified_frac = summary.get("semantic_verified_fraction", 0.0)

    if executed:
        print(f"[phase-sync] status={status} hard_failures={hard_count} soft_failures={soft_count} warnings={warn_count} verified_fraction={verified_frac:.2f}")
    else:
        print(f"[phase-sync] Not executed: {exec_error}")

    # Exit code: PASS=0, any FAIL (hard or soft)=1
    if status == "PASS":
        return 0
    return 1



if __name__ == "__main__":
    sys.exit(main())
