#!/usr/bin/env python
"""
Run Gating - Wrapper CLI for Feasibility Gating

Usage:
    python scripts/run_gating.py --mode quick      # 3 demands × 3 seeds
    python scripts/run_gating.py --mode final      # 5 demands × 5 seeds
    python scripts/run_gating.py --demands 600,800,1000 --seeds 5

Output:
    gating_results/gating_runs.csv       - All individual runs
    gating_results/gating_summary.json   - Aggregated mean±std per demand
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_CONFIG = "configs/train_1.yaml"
DEFAULT_HORIZON = 1500
DEFAULT_WARMUP = 300
DEFAULT_WORKERS = 4

QUICK_DEMANDS = [600, 800, 1000]
QUICK_SEEDS = [42, 43, 44]

FINAL_DEMANDS = [600, 700, 800, 900, 1000]
FINAL_SEEDS = [42, 43, 44, 45, 46]

OUTPUT_DIR = project_root / "gating_results"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feasibility gating sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_gating.py --mode quick
  python scripts/run_gating.py --mode final --workers 8
  python scripts/run_gating.py --demands 600,800 --seeds 3
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "final", "custom"],
        default="quick",
        help="Preset mode: quick (3×3) or final (5×5)"
    )
    parser.add_argument(
        "--demands",
        type=str,
        default=None,
        help="Comma-separated demand levels (e.g., 600,800,1000)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to run (starting from 42)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Simulation horizon in seconds (default: {DEFAULT_HORIZON})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup period in seconds (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Base config file (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="max_pressure,fixed",
        help="Comma-separated controllers (default: max_pressure,fixed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    return parser.parse_args(argv)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results by demand level, computing mean±std.
    """
    from collections import defaultdict
    
    by_demand: Dict[int, List[Dict]] = defaultdict(list)
    
    for r in results:
        if r.get("status") == "ERROR":
            continue
        demand = int(r.get("demand", 0))
        by_demand[demand].append(r)
    
    summary = {}
    
    for demand, runs in sorted(by_demand.items()):
        n = len(runs)
        if n == 0:
            continue
        
        # Extract key metrics
        metrics = {
            "completion_rate": [r.get("completion_rate", 0) for r in runs],
            "teleport_rate": [r.get("teleport_rate", 0) for r in runs],
            "throughput_end_ratio": [r.get("throughput_end_ratio", 0) for r in runs],
            "avg_wait_time": [r.get("avg_wait_time", 0) for r in runs],
            "n_present_end": [r.get("n_present_end", 0) for r in runs],
        }
        
        summary[demand] = {
            "n_runs": n,
            "seeds": [r.get("seed") for r in runs],
            "controllers": list(set(r.get("controller") for r in runs)),
        }
        
        for metric, values in metrics.items():
            arr = np.array(values, dtype=np.float64)
            summary[demand][f"{metric}_mean"] = float(np.mean(arr))
            summary[demand][f"{metric}_std"] = float(np.std(arr))
        
        # Pass/fail counts
        statuses = [r.get("status", "UNKNOWN") for r in runs]
        summary[demand]["strict_pass_count"] = statuses.count("STRICT_PASS")
        summary[demand]["relaxed_pass_count"] = statuses.count("RELAXED_PASS")
        summary[demand]["fail_count"] = statuses.count("FAIL")
        
        # Recommend: demand passes if all runs are STRICT_PASS or RELAXED_PASS
        all_pass = all(s in ("STRICT_PASS", "RELAXED_PASS") for s in statuses)
        strict_pass = all(s == "STRICT_PASS" for s in statuses)
        summary[demand]["recommendation"] = (
            "TRAIN_SAFE" if strict_pass else
            "TRAIN_MARGINAL" if all_pass else
            "EVAL_ONLY"
        )
    
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Resolve demands and seeds based on mode
    if args.demands:
        demands = [int(d.strip()) for d in args.demands.split(",")]
    elif args.mode == "quick":
        demands = QUICK_DEMANDS
    else:
        demands = FINAL_DEMANDS
    
    if args.seeds:
        seeds = list(range(42, 42 + args.seeds))
    elif args.mode == "quick":
        seeds = QUICK_SEEDS
    else:
        seeds = FINAL_SEEDS
    
    controllers = [c.strip() for c in args.controllers.split(",")]
    
    print("=" * 70)
    print("FEASIBILITY GATING")
    print("=" * 70)
    print(f"Mode:        {args.mode}")
    print(f"Demands:     {demands}")
    print(f"Seeds:       {seeds}")
    print(f"Controllers: {controllers}")
    print(f"Horizon:     {args.horizon}s")
    print(f"Warmup:      {args.warmup}s")
    print(f"Workers:     {args.workers}")
    print(f"Config:      {args.config}")
    print("=" * 70)
    
    # Import and run gating
    try:
        from scripts.feasibility_gating import run_demand_sweep
        from rl.utils import load_yaml_config
    except ImportError as e:
        print(f"[ERROR] Failed to import: {e}")
        return 1
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run sweep
    config_path = str(project_root / args.config)
    
    results = run_demand_sweep(
        base_config_path=config_path,
        demands=demands,
        horizons=[args.horizon],
        seeds=seeds,
        controllers=controllers,
        output_dir=output_dir,
        warmup_sec=args.warmup,
        num_workers=args.workers,
    )
    
    # Save raw results
    csv_path = output_dir / "gating_runs.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OUTPUT] Raw results: {csv_path}")
    
    # Aggregate and save summary
    summary = aggregate_results(results)
    
    json_path = output_dir / "gating_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OUTPUT] Summary: {json_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("GATING SUMMARY")
    print("=" * 70)
    print(f"{'Demand':<8} {'Runs':<5} {'Completion':<18} {'Teleport':<15} {'Recommend':<12}")
    print("-" * 70)
    
    for demand in sorted(summary.keys()):
        s = summary[demand]
        comp = f"{s['completion_rate_mean']:.2%} ± {s['completion_rate_std']:.2%}"
        tele = f"{s['teleport_rate_mean']:.2%} ± {s['teleport_rate_std']:.2%}"
        rec = s['recommendation']
        print(f"{demand:<8} {s['n_runs']:<5} {comp:<18} {tele:<15} {rec:<12}")
    
    print("=" * 70)
    
    # Find recommended training demand
    train_candidates = [d for d, s in summary.items() if s['recommendation'] == 'TRAIN_SAFE']
    if train_candidates:
        print(f"\n✅ Recommended train demand: {max(train_candidates)}")
    else:
        marginal = [d for d, s in summary.items() if s['recommendation'] == 'TRAIN_MARGINAL']
        if marginal:
            print(f"\n⚠️ Marginal train demand (use with caution): {min(marginal)}")
        else:
            print("\n❌ No demand level passed gating. Check network/routes.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
