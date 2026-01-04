from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 150,
})


def load_data(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    
    if len(files) == 0:
        raise ValueError(f"No files found: {pattern}")
    
    print(f"Loading {len(files)} files...")
    dfs = [pd.read_csv(f) for f in files]
    
    return pd.concat(dfs, ignore_index=True)


def plot_bar_comparison(df: pd.DataFrame, metrics: List[str], output_dir: Path) -> None:
    if 'controller' not in df.columns:
        print("[WARN] No controller column, skipping bar plots")
        return
    
    controllers = df['controller'].unique()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        means = []
        stds = []
        labels = []
        
        for controller in controllers:
            subset = df[df['controller'] == controller][metric].dropna()
            if len(subset) > 0:
                means.append(subset.mean())
                stds.append(subset.std())
                labels.append(controller)
        
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Controller')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_file = output_dir / f'bar_{metric}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  {output_file.name}")


def plot_box_comparison(df: pd.DataFrame, metrics: List[str], output_dir: Path) -> None:
    if 'controller' not in df.columns:
        print("[WARN] No controller column, skipping box plots")
        return
    
    controllers = df['controller'].unique()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        data_to_plot = []
        labels = []
        
        for controller in controllers:
            subset = df[df['controller'] == controller][metric].dropna()
            if len(subset) > 0:
                data_to_plot.append(subset.values)
                labels.append(controller)
        
        if len(data_to_plot) == 0:
            continue
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f'box_{metric}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  {output_file.name}")


def plot_improvement_heatmap(df: pd.DataFrame, baseline: str, output_dir: Path) -> None:
    if 'controller' not in df.columns:
        return
    
    controllers = [c for c in df['controller'].unique() if c != baseline]
    
    if len(controllers) == 0:
        return
    
    metrics = ['avg_wait_time', 'avg_queue', 'arrived_vehicles', 'total_reward']
    metrics = [m for m in metrics if m in df.columns]
    
    baseline_data = df[df['controller'] == baseline]
    
    improvements = []
    
    for controller in controllers:
        row = []
        controller_data = df[df['controller'] == controller]
        
        for metric in metrics:
            baseline_mean = baseline_data[metric].mean()
            controller_mean = controller_data[metric].mean()
            
            if 'wait' in metric or 'queue' in metric:
                improvement = (baseline_mean - controller_mean) / baseline_mean * 100
            else:
                improvement = (controller_mean - baseline_mean) / abs(baseline_mean) * 100
            
            row.append(improvement)
        
        improvements.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(controllers)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.set_yticklabels(controllers)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=15)
    
    for i in range(len(controllers)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{improvements[i][j]:.1f}%',
                           ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(f'Performance Improvement vs {baseline.title()}')
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_improvements.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  {output_file.name}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate KPI comparison plots from evaluation CSVs.")
    parser.add_argument("--input", required=True, help="Glob pattern for CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--baseline", default="fixed", help="Baseline controller for improvement plots")
    args = parser.parse_args(argv)
    
    print("\n" + "="*80)
    print("KPI PLOTTING")
    print("="*80)
    
    try:
        df = load_data(args.input)
        print(f"Total: {len(df)} rows\n")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = [
        'avg_wait_time',
        'avg_travel_time',
        'avg_queue',
        'arrived_vehicles',
        'total_reward',
    ]
    
    if 'max_wait_time' in df.columns:
        metrics.append('max_wait_time')
    if 'p95_wait_time' in df.columns:
        metrics.append('p95_wait_time')
    
    print("\nGenerating bar charts...")
    plot_bar_comparison(df, metrics, output_dir)
    
    print("\nGenerating box plots...")
    plot_box_comparison(df, metrics, output_dir)
    
    print("\nGenerating improvement heatmap...")
    plot_improvement_heatmap(df, args.baseline, output_dir)
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)
    
    plot_count = len(list(output_dir.glob('*.png')))
    print(f"\nGenerated {plot_count} plots")
    print("\nFiles:")
    for plot_file in sorted(output_dir.glob('*.png')):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[ERROR] matplotlib is required for plotting")
        print("Install: pip install matplotlib")
        sys.exit(1)
    
    main()