"""
Script to evaluate multiple checkpoints from a training run to verify 
the "best" model is truly superior and not just a lucky run.

Usage:
    python scripts/verify_best_model.py --model-dir models/bignet_9tls_long
    
This will:
1. Evaluate ALL checkpoints (episode_50, 100, 150, 200, ..., best)
2. Compare performance across checkpoints
3. Determine if "best" is statistically better or just variance
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config


def find_all_checkpoints(model_dir: Path):
    """Find all .pt checkpoints in directory."""
    checkpoints = []
    for pt_file in model_dir.glob("*.pt"):
        checkpoints.append(pt_file)
    return sorted(checkpoints)


def extract_episode_number(checkpoint_path: Path) -> int:
    """Extract episode number from checkpoint filename."""
    name = checkpoint_path.stem
    if "_best" in name:
        return 999999  # Sort "best" to end
    elif "episode_" in name:
        try:
            return int(name.split("episode_")[1].split("_")[0])
        except:
            return 0
    return 0


def evaluate_checkpoint(checkpoint_path: Path, eval_config_path: Path, output_dir: Path):
    """Evaluate a single checkpoint."""
    import subprocess
    
    output_file = output_dir / f"eval_{checkpoint_path.stem}.csv"
    
    cmd = [
        "python", "scripts/eval.py",
        "--config", str(eval_config_path),
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_file)
    ]
    
    print(f"\n[Evaluating] {checkpoint_path.name}")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Complete: {output_file}")
        return output_file
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return None


def compare_results(eval_results_dir: Path):
    """Compare all evaluation results."""
    all_results = []
    
    for csv_file in eval_results_dir.glob("eval_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            checkpoint_name = csv_file.stem.replace("eval_", "")
            
            # Extract summary stats
            summary = {
                "checkpoint": checkpoint_name,
                "avg_wait_time": df["avg_wait_time_corr"].mean(),
                "avg_wait_time_std": df["avg_wait_time_corr"].std(),
                "throughput": df["throughput_corr"].mean(),
                "completion_rate": df["completion_rate"].mean(),
                "num_seeds": len(df)
            }
            all_results.append(summary)
        except Exception as e:
            print(f"Warning: Could not parse {csv_file}: {e}")
    
    if not all_results:
        print("No evaluation results found!")
        return
    
    # Create comparison dataframe
    df_compare = pd.DataFrame(all_results)
    df_compare = df_compare.sort_values("avg_wait_time")
    
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON (sorted by avg_wait_time)")
    print("="*80)
    print(df_compare.to_string(index=False))
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    best_row = df_compare.iloc[0]
    print(f"\nBest checkpoint: {best_row['checkpoint']}")
    print(f"  Avg wait time: {best_row['avg_wait_time']:.2f} ± {best_row['avg_wait_time_std']:.2f}")
    print(f"  Throughput: {best_row['throughput']:.2f}")
    print(f"  Completion rate: {best_row['completion_rate']:.3f}")
    
    # Check if "best" is actually best
    if "_best" in best_row['checkpoint']:
        print("\n✅ The '_best.pt' checkpoint IS the best performer!")
    else:
        print(f"\n⚠️  WARNING: The '_best.pt' checkpoint is NOT the best!")
        print(f"   Best performer is: {best_row['checkpoint']}")
        print(f"   This suggests 'best' was selected by training reward, not eval performance.")
    
    # Check variance
    wait_times = df_compare["avg_wait_time"].values
    variance = np.std(wait_times)
    print(f"\nVariance across checkpoints: {variance:.2f}")
    if variance > 5.0:
        print("⚠️  HIGH VARIANCE: Checkpoints have very different performance!")
        print("   This suggests training was unstable or 'best' may be lucky.")
    else:
        print("✓ Low variance: Checkpoints are relatively consistent.")
    
    # Save comparison
    output_path = eval_results_dir / "checkpoint_comparison.csv"
    df_compare.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify if 'best' checkpoint is truly best")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--eval-config", type=str, default="configs/eval.yaml", help="Eval config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for eval results")
    args = parser.parse_args()
    
    model_dir = project_root / args.model_dir
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Find checkpoints
    checkpoints = find_all_checkpoints(model_dir)
    if not checkpoints:
        print(f"Error: No checkpoints found in {model_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints in {model_dir}")
    
    # Sort by episode number
    checkpoints = sorted(checkpoints, key=extract_episode_number)
    
    # Create output directory
    if args.output_dir:
        output_dir = project_root / args.output_dir
    else:
        output_dir = project_root / "results" / "verify_best" / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_config = project_root / args.eval_config
    if not eval_config.exists():
        print(f"Error: Eval config not found: {eval_config}")
        sys.exit(1)
    
    # Evaluate all checkpoints
    print(f"\nEvaluating {len(checkpoints)} checkpoints...")
    print(f"Results will be saved to: {output_dir}")
    
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] {checkpoint.name}")
        evaluate_checkpoint(checkpoint, eval_config, output_dir)
    
    # Compare results
    print("\n" + "="*80)
    print("EVALUATIONS COMPLETE - COMPARING RESULTS")
    print("="*80)
    compare_results(output_dir)


if __name__ == "__main__":
    main()
