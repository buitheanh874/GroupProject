from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def load_csvs(pattern: str) -> pd.DataFrame:
    """Load and concatenate all CSV files matching pattern."""
    files = glob.glob(pattern)
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Loading {len(files)} CSV files...")
    dfs = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  ✓ {Path(file).name} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ {Path(file).name}: {e}")
    
    if len(dfs) == 0:
        raise ValueError("No valid CSV files could be loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows from {len(dfs)} files\n")
    
    return combined


def aggregate_stats(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Compute mean and std for each controller/scenario combination."""
    metrics = [
        'total_reward',
        'episode_steps',
        'arrived_vehicles',
        'avg_wait_time',
        'avg_travel_time',
        'avg_stops',
        'avg_queue',
    ]

    if 'max_wait_time' in df.columns:
        metrics.append('max_wait_time')
    if 'p95_wait_time' in df.columns:
        metrics.append('p95_wait_time')
    metrics = [m for m in metrics if m in df.columns]
    grouped = df.groupby(group_by)[metrics].agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    return grouped.reset_index()


def format_markdown(df: pd.DataFrame) -> str:
    """Format aggregated stats as Markdown table."""
    lines = []
    lines.append("# KPI Summary")
    lines.append("")

    if 'controller' in df.columns:
        for controller in df['controller'].unique():
            lines.append(f"## {controller.upper()} Controller")
            lines.append("")
            subset = df[df['controller'] == controller]
            lines.append("| Metric | Mean | Std | N |")
            lines.append("|--------|------|-----|---|")
            
            for _, row in subset.iterrows():
                for col in subset.columns:
                    if col.endswith('_mean'):
                        metric = col.replace('_mean', '')
                        mean_val = row[f'{metric}_mean']
                        std_val = row.get(f'{metric}_std', 0)
                        count_val = row.get(f'{metric}_count', 0)
                        
                        lines.append(f"| {metric} | {mean_val:.3f} | {std_val:.3f} | {int(count_val)} |")
            
            lines.append("")
    else:
        lines.append("| Metric | Mean | Std | N |")
        lines.append("|--------|------|-----|---|")
        
        for col in df.columns:
            if col.endswith('_mean'):
                metric = col.replace('_mean', '')
                mean_val = df[f'{metric}_mean'].iloc[0]
                std_val = df.get(f'{metric}_std', pd.Series([0])).iloc[0]
                count_val = df.get(f'{metric}_count', pd.Series([0])).iloc[0]
                
                lines.append(f"| {metric} | {mean_val:.3f} | {std_val:.3f} | {int(count_val)} |")
    
    return '\n'.join(lines)


def format_latex(df: pd.DataFrame) -> str:
    """Format aggregated stats as LaTeX table."""
    lines = []
    
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{KPI Summary}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Metric & Mean & Std & N \\\\")
    lines.append("\\midrule")
    
    for col in df.columns:
        if col.endswith('_mean'):
            metric = col.replace('_mean', '').replace('_', ' ').title()
            mean_val = df[f'{col}'].iloc[0]
            std_col = col.replace('_mean', '_std')
            std_val = df.get(std_col, pd.Series([0])).iloc[0]
            count_col = col.replace('_mean', '_count')
            count_val = df.get(count_col, pd.Series([0])).iloc[0]
            
            lines.append(f"{metric} & {mean_val:.3f} & {std_val:.3f} & {int(count_val)} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return '\n'.join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Glob pattern for input CSV files")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["csv", "markdown", "latex"], default="markdown", help="Output format")
    parser.add_argument("--group-by", nargs="+", default=["controller"], help="Columns to group by")
    args = parser.parse_args(argv)
    
    try:
        df = load_csvs(args.input)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    
    missing = [col for col in args.group_by if col not in df.columns]
    if missing:
        print(f"\n[WARNING] Grouping columns not found: {missing}")
        print(f"Available columns: {list(df.columns)}")
        args.group_by = []

    if args.group_by:
        print(f"Aggregating by: {args.group_by}")
        aggregated = aggregate_stats(df, args.group_by)
    else:
        print("Computing overall statistics (no grouping)")
        metrics = [col for col in df.columns if col not in ['run_id', 'episode', 'scenario']]
        aggregated = df[metrics].agg(['mean', 'std', 'count']).T
        aggregated.columns = ['mean', 'std', 'count']
        aggregated = aggregated.reset_index()
        aggregated.columns = ['metric', 'value_mean', 'value_std', 'value_count']
    
    print(f"\nAggregated {len(aggregated)} rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "csv":
        aggregated.to_csv(output_path, index=False)
        print(f"\n✓ Saved CSV to: {output_path}")
    
    elif args.format == "markdown":
        markdown_text = format_markdown(aggregated)
        output_path.write_text(markdown_text, encoding='utf-8')
        print(f"\n✓ Saved Markdown to: {output_path}")
        print("\nPreview:")
        print("-" * 80)
        print(markdown_text[:500])
        if len(markdown_text) > 500:
            print("...")
    
    elif args.format == "latex":
        latex_text = format_latex(aggregated)
        output_path.write_text(latex_text, encoding='utf-8')
        print(f"\n✓ Saved LaTeX to: {output_path}")
        print("\nPreview:")
        print("-" * 80)
        print(latex_text)
    
    print(f"\nOutput ready for report: {output_path}")


if __name__ == "__main__":
    main()