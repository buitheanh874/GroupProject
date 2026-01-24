"""
Merge multiple normalization JSON files into a single stats file.
Usage: python scripts/merge_norm_stats.py --inputs norm_500.json norm_750.json norm_1000.json --out configs/norm_turn801010.json
"""
import argparse
import json
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', required=True, help="List of JSON files to merge")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    all_means = []
    all_stds = []
    all_counts = []
    
    total_samples = 0
    state_dim = None

    print(f"Merging {len(args.inputs)} files...")

    for fpath in args.inputs:
        path = Path(fpath)
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        n = data['num_samples']
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        
        # Validate data
        if state_dim is None:
            state_dim = data['state_dim']
        elif data['state_dim'] != state_dim:
            print(f"[ERROR] State dim mismatch in {path}: {data['state_dim']} vs {state_dim}")
            continue
            
        all_means.append(mean)
        all_stds.append(std)
        all_counts.append(n)
        total_samples += n
        
        print(f"  Loaded {path.name}: {n} samples (d={data.get('demand', '?')})")

    if total_samples == 0:
        print("No samples merged.")
        return

    # Weighted merger of means and variances
    all_means = np.array(all_means)
    all_vars = np.array(all_stds) ** 2
    all_counts = np.array(all_counts).reshape(-1, 1)
    
    # 1. Combined Mean
    weighted_sum_means = np.sum(all_means * all_counts, axis=0)
    combined_mean = weighted_sum_means / total_samples
    
    # 2. Combined Variance
    term1 = np.sum(all_vars * all_counts, axis=0)
    mean_diff_sq = (all_means - combined_mean) ** 2
    term2 = np.sum(mean_diff_sq * all_counts, axis=0)
    
    combined_var = (term1 + term2) / total_samples
    combined_std = np.sqrt(combined_var)
    combined_std = np.maximum(combined_std, 1e-6)

    # Save
    stats = {
        "mean": combined_mean.tolist(),
        "std": combined_std.tolist(),
        "state_dim": state_dim,
        "num_samples": int(total_samples),
        "source_files": [str(p) for p in args.inputs]
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nSuccessfully merged to {out_path}")
    print(f"Total Samples: {total_samples}")

if __name__ == "__main__":
    main()
