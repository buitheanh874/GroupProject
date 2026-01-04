from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def load_controller_data(path: str, controller_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['controller_label'] = controller_name
    return df


def compute_stats(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    stats_dict = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna()
        stats_dict[metric] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'count': int(len(values)),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    
    return stats_dict


def compute_improvement(baseline_val: float, treatment_val: float, metric: str) -> float:
    if baseline_val == 0:
        return 0.0
    
    lower_is_better = ['avg_wait_time', 'avg_travel_time', 'avg_queue', 'max_wait_time', 'p95_wait_time']
    
    if metric in lower_is_better:
        improvement = (baseline_val - treatment_val) / baseline_val * 100
    else:
        improvement = (treatment_val - baseline_val) / abs(baseline_val) * 100
    
    return improvement


def t_test_comparison(df1: pd.DataFrame, df2: pd.DataFrame, metric: str) -> tuple[float, float]:
    if metric not in df1.columns or metric not in df2.columns:
        return float('nan'), float('nan')
    
    values1 = df1[metric].dropna()
    values2 = df2[metric].dropna()
    
    if len(values1) < 2 or len(values2) < 2:
        return float('nan'), float('nan')
    
    t_stat, p_value = stats.ttest_ind(values1, values2)
    return float(t_stat), float(p_value)


def format_markdown_report(comparisons: Dict[str, Any], baseline_name: str) -> str:
    lines = []
    
    lines.append("# Controller Performance Comparison")
    lines.append("")
    lines.append(f"**Baseline:** {baseline_name}")
    lines.append("")
    
    lines.append("## Summary")
    lines.append("")
    lines.append("| Controller | Avg Wait (s) | Improvement | Arrived Veh | P-value |")
    lines.append("|------------|--------------|-------------|-------------|---------|")
    
    for controller, data in comparisons.items():
        if controller == baseline_name:
            continue
        
        avg_wait = data['stats'].get('avg_wait_time', {}).get('mean', 0)
        improvement = data['improvements'].get('avg_wait_time', 0)
        arrived = data['stats'].get('arrived_vehicles', {}).get('mean', 0)
        p_value = data['p_values'].get('avg_wait_time', float('nan'))
        
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        lines.append(f"| {controller} | {avg_wait:.2f} | {improvement:+.1f}% | {arrived:.0f} | {p_value:.4f} {sig_marker} |")
    
    lines.append("")
    lines.append("*Significance: *** p<0.001, ** p<0.01, * p<0.05*")
    lines.append("")
    
    lines.append("## Detailed Metrics")
    lines.append("")
    
    metrics = ['avg_wait_time', 'avg_travel_time', 'avg_queue', 'avg_stops', 'arrived_vehicles', 'total_reward']
    
    lines.append("| Metric | " + " | ".join(comparisons.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(comparisons)) + "|")
    
    for metric in metrics:
        metric_display = metric.replace('_', ' ').title()
        row_values = []
        
        for controller in comparisons.keys():
            val = comparisons[controller]['stats'].get(metric, {}).get('mean', 0)
            std = comparisons[controller]['stats'].get(metric, {}).get('std', 0)
            
            if controller == baseline_name:
                row_values.append(f"{val:.2f} ± {std:.2f}")
            else:
                improvement = comparisons[controller]['improvements'].get(metric, 0)
                row_values.append(f"{val:.2f} ({improvement:+.1f}%)")
        
        lines.append(f"| {metric_display} | " + " | ".join(row_values) + " |")
    
    lines.append("")
    
    lines.append("## Interpretation")
    lines.append("")
    
    best_controller = None
    best_improvement = -float('inf')
    
    for controller, data in comparisons.items():
        if controller == baseline_name:
            continue
        
        improvement = data['improvements'].get('avg_wait_time', -float('inf'))
        if improvement > best_improvement:
            best_improvement = improvement
            best_controller = controller
    
    if best_controller:
        lines.append(f"**Best performer:** {best_controller} with {best_improvement:+.1f}% improvement in average wait time")
        
        p_val = comparisons[best_controller]['p_values'].get('avg_wait_time', 1.0)
        if p_val < 0.05:
            lines.append(f"- This improvement is **statistically significant** (p={p_val:.4f})")
        else:
            lines.append(f"- This improvement is **not statistically significant** (p={p_val:.4f})")
    
    lines.append("")
    
    return '\n'.join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare performance between different controllers.")
    parser.add_argument("--fixed", required=True, help="Fixed controller CSV")
    parser.add_argument("--rl", help="RL controller CSV (optional)")
    parser.add_argument("--max-pressure", help="Max-pressure controller CSV (optional)")
    parser.add_argument("--output", required=True, help="Output markdown file")
    parser.add_argument("--baseline", default="fixed", help="Baseline controller name")
    args = parser.parse_args(argv)
    
    print("Loading controller data...")
    
    data = {
        'fixed': load_controller_data(args.fixed, 'fixed')
    }
    
    if args.rl:
        data['rl'] = load_controller_data(args.rl, 'rl')
    
    if args.max_pressure:
        data['max_pressure'] = load_controller_data(args.max_pressure, 'max_pressure')
    
    print(f"  Loaded {len(data)} controllers")
    
    metrics = [
        'total_reward',
        'episode_steps',
        'arrived_vehicles',
        'avg_wait_time',
        'avg_travel_time',
        'avg_stops',
        'avg_queue',
    ]
    
    for df in data.values():
        if 'max_wait_time' in df.columns and 'max_wait_time' not in metrics:
            metrics.append('max_wait_time')
        if 'p95_wait_time' in df.columns and 'p95_wait_time' not in metrics:
            metrics.append('p95_wait_time')
    
    print("\nComputing statistics...")
    
    comparisons = {}
    baseline_stats = compute_stats(data[args.baseline], metrics)
    
    for controller_name, df in data.items():
        stats_dict = compute_stats(df, metrics)
        
        improvements = {}
        if controller_name != args.baseline:
            for metric in metrics:
                if metric in baseline_stats and metric in stats_dict:
                    baseline_val = baseline_stats[metric]['mean']
                    treatment_val = stats_dict[metric]['mean']
                    improvements[metric] = compute_improvement(baseline_val, treatment_val, metric)
        
        p_values = {}
        if controller_name != args.baseline:
            for metric in metrics:
                t_stat, p_val = t_test_comparison(data[args.baseline], df, metric)
                p_values[metric] = p_val
        
        comparisons[controller_name] = {
            'stats': stats_dict,
            'improvements': improvements,
            'p_values': p_values,
        }
    
    print("\nGenerating report...")
    
    report = format_markdown_report(comparisons, args.baseline)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding='utf-8')
    
    print(f"\nSaved comparison report to: {output_path}")


if __name__ == "__main__":
    try:
        from scipy import stats
    except ImportError:
        print("\n[ERROR] scipy is required for statistical tests")
        print("Install: pip install scipy")
        sys.exit(1)
    
    main()