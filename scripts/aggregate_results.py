#!/usr/bin/env python
"""
Aggregate Results - Compute mean±std and generate reports

Usage:
    python scripts/aggregate_results.py results/eval_matrix.csv
    python scripts/aggregate_results.py results/eval_matrix.csv --output results/summary

Outputs:
    - summary.md: Markdown table for reports
    - summary.csv: Clean CSV with aggregated stats
    - learning_curves.png: RL training curves (if train logs provided)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results and generate reports",
    )
    
    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV from eval_matrix.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base path (default: same dir as input)"
    )
    parser.add_argument(
        "--train-logs",
        type=str,
        default=None,
        help="Comma-separated paths to training log CSVs for learning curves"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "both"],
        default="markdown",
        help="Table format (default: markdown)"
    )
    
    return parser.parse_args(argv)


def load_results(csv_path: str) -> List[Dict[str, Any]]:
    """Load evaluation results from CSV."""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["demand", "seed", "horizon_sec", "warmup_sec",
                       "avg_wait_time_corr", "avg_travel_time_corr",
                       "throughput_corr", "completion_rate", "teleport_rate",
                       "arrived_vehicles", "n_present_end", "avg_queue",
                       "total_reward", "episode_steps"]:
                if key in row and row[key]:
                    try:
                        if key in ["demand", "seed", "horizon_sec", "warmup_sec",
                                  "arrived_vehicles", "n_present_end", "episode_steps"]:
                            row[key] = int(row[key])
                        else:
                            row[key] = float(row[key])
                    except ValueError:
                        pass
            results.append(row)
    return results


def aggregate_by_policy_demand(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Aggregate results by (policy, demand), computing mean±std.
    
    Returns:
        {policy: {demand: {metric_mean, metric_std, n_seeds, ...}}}
    """
    grouped: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r.get("status") == "ERROR":
            continue
        policy = r.get("policy", "unknown")
        demand = int(r.get("demand", 0))
        grouped[policy][demand].append(r)
    
    aggregated = {}
    
    for policy, by_demand in grouped.items():
        aggregated[policy] = {}
        for demand, runs in by_demand.items():
            n = len(runs)
            
            metrics_to_agg = [
                "avg_wait_time_corr",
                "avg_travel_time_corr", 
                "throughput_corr",
                "completion_rate",
                "teleport_rate",
                "arrived_vehicles",
                "n_present_end",
                "avg_queue",
                "total_reward",
            ]
            
            agg = {"n_seeds": n, "seeds": [r.get("seed") for r in runs]}
            
            for metric in metrics_to_agg:
                values = [float(r.get(metric, 0)) for r in runs]
                arr = np.array(values, dtype=np.float64)
                agg[f"{metric}_mean"] = float(np.mean(arr))
                agg[f"{metric}_std"] = float(np.std(arr))
            
            aggregated[policy][demand] = agg
    
    return aggregated


def format_metric(mean: float, std: float, is_percent: bool = False) -> str:
    """Format metric as mean ± std."""
    if is_percent:
        return f"{mean*100:.1f}% ± {std*100:.1f}%"
    else:
        return f"{mean:.2f} ± {std:.2f}"


def generate_markdown_table(
    aggregated: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate markdown table."""
    
    # Collect all demands
    all_demands = set()
    for policy_data in aggregated.values():
        all_demands.update(policy_data.keys())
    all_demands = sorted(all_demands)
    
    # Collect all policies
    all_policies = sorted(aggregated.keys())
    
    lines = [
        "# Evaluation Results Summary\n",
        f"Generated from {len(all_policies)} policies × {len(all_demands)} demands\n",
        "",
        "## Main Results Table\n",
        "| Policy | Demand | Seeds | Avg Wait (s) | Avg Travel (s) | Throughput | Completion | Teleport |",
        "|--------|--------|-------|--------------|----------------|------------|------------|----------|",
    ]
    
    for policy in all_policies:
        for demand in all_demands:
            if demand not in aggregated[policy]:
                continue
            
            data = aggregated[policy][demand]
            n = data["n_seeds"]
            
            wait = format_metric(
                data["avg_wait_time_corr_mean"],
                data["avg_wait_time_corr_std"]
            )
            travel = format_metric(
                data["avg_travel_time_corr_mean"],
                data["avg_travel_time_corr_std"]
            )
            thru = format_metric(
                data["throughput_corr_mean"],
                data["throughput_corr_std"]
            )
            comp = format_metric(
                data["completion_rate_mean"],
                data["completion_rate_std"],
                is_percent=True
            )
            tele = format_metric(
                data["teleport_rate_mean"],
                data["teleport_rate_std"],
                is_percent=True
            )
            
            lines.append(f"| {policy} | {demand} | {n} | {wait} | {travel} | {thru} | {comp} | {tele} |")
    
    lines.extend([
        "",
        "## Notes\n",
        "- **Avg Wait**: Average waiting time per vehicle (corrected for teleports)",
        "- **Avg Travel**: Average travel time per vehicle (corrected)",
        "- **Throughput**: Vehicles arrived per simulation step",
        "- **Completion**: Fraction of inserted vehicles that arrived",
        "- **Teleport**: Fraction of vehicles that teleported (indicates congestion)",
        "",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[OUTPUT] Markdown table: {output_path}")


def generate_summary_csv(
    aggregated: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate clean summary CSV."""
    
    fieldnames = [
        "policy", "demand", "n_seeds",
        "avg_wait_time_corr_mean", "avg_wait_time_corr_std",
        "avg_travel_time_corr_mean", "avg_travel_time_corr_std",
        "throughput_corr_mean", "throughput_corr_std",
        "completion_rate_mean", "completion_rate_std",
        "teleport_rate_mean", "teleport_rate_std",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for policy in sorted(aggregated.keys()):
            for demand in sorted(aggregated[policy].keys()):
                data = aggregated[policy][demand]
                row = {
                    "policy": policy,
                    "demand": demand,
                    "n_seeds": data["n_seeds"],
                }
                for key in fieldnames[3:]:
                    row[key] = data.get(key, 0)
                writer.writerow(row)
    
    print(f"[OUTPUT] Summary CSV: {output_path}")


def generate_learning_curves(
    train_log_paths: List[str],
    output_path: Path,
) -> None:
    """Generate learning curves from training logs."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping learning curves")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10.colors
    
    for idx, log_path in enumerate(train_log_paths):
        if not Path(log_path).exists():
            continue
        
        # Load training log
        episodes = []
        rewards = []
        losses = []
        
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "episode" in row:
                    episodes.append(int(row["episode"]))
                if "total_reward" in row or "reward" in row:
                    r = row.get("total_reward", row.get("reward", 0))
                    rewards.append(float(r) if r else 0)
                if "loss" in row or "avg_loss" in row:
                    l = row.get("loss", row.get("avg_loss", 0))
                    losses.append(float(l) if l else 0)
        
        label = Path(log_path).stem
        color = colors[idx % len(colors)]
        
        # Episode rewards
        if episodes and rewards:
            ax = axes[0, 0]
            ax.plot(episodes[:len(rewards)], rewards, label=label, color=color, alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Episode Rewards")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Smoothed rewards (rolling mean)
        if len(rewards) > 10:
            ax = axes[0, 1]
            window = min(50, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(len(smoothed)), smoothed, label=label, color=color)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward (Smoothed)")
            ax.set_title(f"Smoothed Rewards (window={window})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Loss
        if losses and any(l > 0 for l in losses):
            ax = axes[1, 0]
            ax.plot(range(len(losses)), losses, label=label, color=color, alpha=0.7)
            ax.set_xlabel("Update Step")
            ax.set_ylabel("TD Loss")
            ax.set_title("Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Placeholder for 4th subplot
    axes[1, 1].text(0.5, 0.5, "Reserved for\nablation comparison", 
                     ha='center', va='center', fontsize=12)
    axes[1, 1].set_title("Ablation")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OUTPUT] Learning curves: {output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    # Determine output paths
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = input_path.parent / input_path.stem
    
    output_base.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_base}.*")
    
    # Load and aggregate
    results = load_results(str(input_path))
    print(f"Loaded {len(results)} evaluation runs")
    
    aggregated = aggregate_by_policy_demand(results)
    
    # Generate outputs
    generate_markdown_table(aggregated, Path(f"{output_base}.md"))
    generate_summary_csv(aggregated, Path(f"{output_base}_stats.csv"))
    
    # Generate learning curves if train logs provided
    if args.train_logs:
        log_paths = [p.strip() for p in args.train_logs.split(",")]
        generate_learning_curves(log_paths, Path(f"{output_base}_curves.png"))
    
    print("=" * 60)
    print("DONE")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
