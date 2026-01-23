"""
Fine-tuning script for the BIGNET 9-TLS traffic signal control model.

This script fine-tunes a pre-trained model (train_bignet_300ep_20260118_092429_best.pt)
to improve performance by:
1. Loading the pre-trained weights
2. Using lower learning rate for stable refinement
3. Focusing on harder scenarios in the curriculum
4. Running for fewer episodes (100 vs original 200)

Usage:
    python scripts/finetune.py

    Or with custom settings:
    python scripts/finetune.py --episodes 50 --lr 0.00005
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from rl.utils import load_yaml_config, save_yaml_config


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the pre-trained BIGNET traffic signal control model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default fine-tuning (100 episodes, lr=0.0001)
    python scripts/finetune.py

    # Shorter fine-tuning run
    python scripts/finetune.py --episodes 50

    # Custom learning rate
    python scripts/finetune.py --lr 0.00005

    # Custom checkpoint
    python scripts/finetune.py --checkpoint models/other_model.pt
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/BEST/train_bignet_300ep_20260118_092429_best.pt",
        help="Path to the pre-trained checkpoint to fine-tune from"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_1.yaml",
        help="Path to fine-tuning config YAML"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of fine-tuning episodes (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=None,
        help="Override starting epsilon (default: 0.15)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name (default: finetune_bignet)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=1,
        help="Starting episode number (useful for resuming fine-tuning)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = project_root / args.checkpoint
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints in models/BEST:")
        best_dir = project_root / "models" / "BEST"
        if best_dir.exists():
            for f in best_dir.glob("*.pt"):
                print(f"  - {f.name}")
        sys.exit(1)
    
    # Load config
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    
    config = load_yaml_config(str(config_path))
    
    # Apply overrides
    if args.episodes is not None:
        # Update total episodes and redistribute across phases
        total_episodes = args.episodes
        config.setdefault("train", {})["episodes"] = total_episodes
        
        # Redistribute curriculum phases proportionally
        if config.get("curriculum", {}).get("enabled", False):
            phases = config["curriculum"]["phases"]
            ratios = [0.20, 0.50, 0.30]  # Default distribution
            for i, phase in enumerate(phases):
                if i < len(ratios):
                    phase["episodes"] = max(1, int(total_episodes * ratios[i]))
        
        print(f"[Config Override] Episodes: {total_episodes}")
    
    if args.lr is not None:
        config.setdefault("agent", {})["learning_rate"] = args.lr
        print(f"[Config Override] Learning rate: {args.lr}")
    
    if args.eps_start is not None:
        config.setdefault("exploration", {})["eps_start"] = args.eps_start
        print(f"[Config Override] Epsilon start: {args.eps_start}")
    
    if args.run_name is not None:
        config.setdefault("run", {})["run_name"] = args.run_name
        print(f"[Config Override] Run name: {args.run_name}")
    
    if args.seed is not None:
        config.setdefault("run", {})["seed"] = args.seed
        print(f"[Config Override] Seed: {args.seed}")
    
    # Print fine-tuning info
    print("\n" + "=" * 60)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 60)
    print(f"Pre-trained checkpoint: {checkpoint_path}")
    print(f"Config file: {config_path}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {config.get('agent', {}).get('learning_rate', 0.0001)}")
    print(f"  Epsilon start: {config.get('exploration', {}).get('eps_start', 0.15)}")
    print(f"  Epsilon end: {config.get('exploration', {}).get('eps_end', 0.02)}")
    print(f"  Epsilon decay steps: {config.get('exploration', {}).get('eps_decay_steps', 5000)}")
    
    if config.get("curriculum", {}).get("enabled", False):
        print(f"\nCurriculum phases:")
        for i, phase in enumerate(config["curriculum"]["phases"]):
            print(f"  Phase {i+1}: {phase['name']} - {phase['episodes']} episodes")
    else:
        print(f"\nTotal episodes: {config.get('train', {}).get('episodes', 100)}")
    
    print(f"\nOutput directories:")
    print(f"  Logs: {config.get('logging', {}).get('log_dir', 'logs/finetune_bignet')}")
    print(f"  Models: {config.get('logging', {}).get('model_dir', 'models/finetune_bignet')}")
    print("=" * 60 + "\n")
    
    # Import training function
    from scripts.train import run_training
    
    # Run fine-tuning
    print(f"[Fine-tune] Starting from: {checkpoint_path}")
    print(f"[Fine-tune] Start episode: {args.start_episode}")
    
    try:
        metrics_path = run_training(
            config=config,
            resume_path=str(checkpoint_path),
            start_episode_override=args.start_episode
        )
        print(f"\n[Fine-tune] Complete! Metrics saved to: {metrics_path}")
        print(f"[Fine-tune] Check models/finetune_bignet for the fine-tuned model")
        
    except KeyboardInterrupt:
        print("\n[Fine-tune] Interrupted by user. Check model_dir for crash checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Fine-tune] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
