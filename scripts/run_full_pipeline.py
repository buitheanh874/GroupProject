from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


TRAIN_CONFIGS = {
    "fair_low": "configs/train_fair_low.yaml",
    "fair_high": "configs/train_fair_high.yaml",
    "nofair_low": "configs/train_nofair_low.yaml",
    "nofair_high": "configs/train_nofair_high.yaml",
}

EVAL_CONFIGS = {
    "low": "configs/eval_low.yaml",
    "high": "configs/eval_high.yaml",
}

NORM_STATS_CONFIGS = {
    "low": ("configs/train_fair_low.yaml", "configs/norm_stats_low.json"),
    "high": ("configs/train_fair_high.yaml", "configs/norm_stats_high.json"),
}


def run_command(cmd: List[str], description: str) -> bool:
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=repo_root)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} FAILED with exit code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] {description} completed")
    return True


def collect_normalization_stats() -> bool:
    print("\n" + "="*80)
    print("PHASE 1: COLLECT NORMALIZATION STATISTICS")
    print("="*80)
    
    import json
    
    for traffic_level, (config_path, output_path) in NORM_STATS_CONFIGS.items():
        if Path(output_path).exists():
            print(f"\n[SKIP] {output_path} already exists")
            continue

        dummy_stats = {
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "episodes": 0,
            "seed": 0,
            "num_samples": 0,
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dummy_stats, f, indent=2)
        
        print(f"\n[INFO] Created dummy {output_path} (will be overwritten)")
        
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "collect_norm_stats.py"),
            "--config", config_path,
            "--episodes", "5",
            "--seed", "0",
            "--out", output_path,
        ]
        
        if not run_command(cmd, f"Collect norm stats for {traffic_level} traffic"):
            return False
    
    return True


def train_all_models(episodes: int = 300) -> Dict[str, str]:
    print("\n" + "="*80)
    print("PHASE 2: TRAIN ALL MODELS")
    print("="*80)
    
    trained_models = {}
    
    for model_name, config_path in TRAIN_CONFIGS.items():
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--config", config_path,
            "--episodes", str(episodes),
        ]
        
        if not run_command(cmd, f"Train model: {model_name}"):
            print(f"[WARNING] Training {model_name} failed, continuing with others...")
            continue
        
        model_dir = Path(f"models/{model_name}")
        best_model = model_dir / f"train_{model_name}_*_best.pt"
        
        import glob
        matches = glob.glob(str(best_model))
        
        if matches:
            trained_models[model_name] = matches[0]
            print(f"[INFO] Model saved: {trained_models[model_name]}")
        else:
            print(f"[WARNING] Best model not found for {model_name}")
    
    return trained_models


def evaluate_all_combinations(trained_models: Dict[str, str], runs: int = 10) -> bool:
    print("\n" + "="*80)
    print("PHASE 3: EVALUATE ALL COMBINATIONS")
    print("="*80)
    
    all_success = True
    
    for model_name, model_path in trained_models.items():
        for traffic_level, eval_config in EVAL_CONFIGS.items():
            cmd = [
                sys.executable,
                str(repo_root / "scripts" / "eval.py"),
                "--config", eval_config,
                "--controller", "all",
                "--model_path", model_path,
                "--runs", str(runs),
            ]
            
            description = f"Evaluate {model_name} on {traffic_level} traffic"
            
            if not run_command(cmd, description):
                print(f"[WARNING] {description} failed, continuing...")
                all_success = False
    
    return all_success


def generate_summary_report(trained_models: Dict[str, str]) -> None:
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    print("\nTrained Models:")
    for model_name, model_path in trained_models.items():
        fairness_type = "WITH fairness" if "fair_" in model_name and "nofair" not in model_name else "NO fairness"
        traffic_type = "LOW (50 veh/h)" if "_low" in model_name else "HIGH (150 veh/h)"
        print(f"  - {model_name:15s}: {fairness_type:15s} | {traffic_type:15s} | {model_path}")
    
    print("\nEvaluation Matrix:")
    print("  Each model evaluated on:")
    print("    - LOW traffic (BI_50_test.rou.xml)")
    print("    - HIGH traffic (BI_150_test.rou.xml)")
    
    print("\nResults Location:")
    print("  Training logs:  logs/")
    print("  Trained models: models/")
    print("  Eval results:   results/")
    
    print("\nComparison Analysis:")
    print("  1. Fairness impact: Compare fair_* vs nofair_* models")
    print("  2. Traffic robustness: Compare model performance across low/high traffic")
    print("  3. Generalization: Models trained on low tested on high (and vice versa)")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Master training and evaluation pipeline")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per model")
    parser.add_argument("--eval-runs", type=int, default=10, help="Evaluation runs per setting")
    parser.add_argument("--skip-norm", action="store_true", help="Skip normalization stats collection")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing models)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MASTER TRAINING & EVALUATION PIPELINE")
    print("="*80)
    print(f"Training episodes: {args.episodes}")
    print(f"Evaluation runs: {args.eval_runs}")
    
    if not args.skip_norm:
        if not collect_normalization_stats():
            sys.exit(1)
    
    trained_models = {}
    
    if not args.skip_train:
        trained_models = train_all_models(episodes=args.episodes)
        
        if len(trained_models) == 0:
            print("\n[ERROR] No models were successfully trained!")
            sys.exit(1)
    else:
        print("\n[SKIP] Training phase skipped")
        import glob
        for model_name in TRAIN_CONFIGS.keys():
            model_dir = Path(f"models/{model_name}")
            best_model = model_dir / f"train_{model_name}_*_best.pt"
            matches = glob.glob(str(best_model))
            if matches:
                trained_models[model_name] = matches[0]
        
        if len(trained_models) == 0:
            print("\n[ERROR] No existing models found!")
            sys.exit(1)
    
    if not args.skip_eval:
        evaluate_all_combinations(trained_models, runs=args.eval_runs)
    
    generate_summary_report(trained_models)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()