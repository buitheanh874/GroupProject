#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import copy
import yaml

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config


def distribute_episodes(total_episodes: int, num_workers: int) -> List[tuple]:
    base_per_worker = total_episodes // num_workers
    remainder = total_episodes % num_workers
    
    assignments = []
    current_start = 1
    
    for i in range(num_workers):
        worker_episodes = base_per_worker + (1 if i < remainder else 0)
        end_episode = current_start + worker_episodes - 1
        assignments.append((current_start, end_episode, worker_episodes))
        current_start = end_episode + 1
    
    return assignments


def get_phase_for_episode(episode: int, phases: List[Dict]) -> Dict:
    cumulative = 0
    for phase in phases:
        cumulative += phase.get("episodes", 0)
        if episode <= cumulative:
            return phase
    return phases[-1]  


def generate_worker_config(
    base_config: Dict[str, Any],
    worker_id: int,
    start_ep: int,
    end_ep: int,
    num_episodes: int,
    output_dir: Path,
) -> Path:
    config = copy.deepcopy(base_config)
    config["run"]["run_name"] = f"train_worker{worker_id:03d}"
    config["train"]["episodes"] = num_episodes
    config["logging"] = {
        "log_dir": f"logs/parallel/worker{worker_id:03d}",
        "model_dir": f"models/parallel/worker{worker_id:03d}",
        "results_dir": f"results/parallel/worker{worker_id:03d}",
    }

    phases = config.get("curriculum", {}).get("phases", [])
    if phases:
        cumulative = 0
        worker_phases = []
        remaining_episodes = num_episodes
        
        for phase in phases:
            phase_start = cumulative + 1
            phase_end = cumulative + phase.get("episodes", 0)
            
            if phase_end >= start_ep and phase_start <= end_ep:
                overlap_start = max(phase_start, start_ep)
                overlap_end = min(phase_end, end_ep)
                overlap_count = overlap_end - overlap_start + 1
                
                if overlap_count > 0:
                    worker_phase = copy.deepcopy(phase)
                    worker_phase["episodes"] = overlap_count
                    worker_phases.append(worker_phase)
            
            cumulative = phase_end
        
        if worker_phases:
            config["curriculum"]["phases"] = worker_phases
    
    config_path = output_dir / f"train_worker{worker_id:03d}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return config_path


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/train_1.yaml")
    parser.add_argument("--output-dir", type=str, default="configs/parallel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    base_config = load_yaml_config(args.config)
    curriculum = base_config.get("curriculum", {})
    phases = curriculum.get("phases", [])
    
    if phases:
        total_episodes = sum(p.get("episodes", 0) for p in phases)
    else:
        total_episodes = base_config.get("train", {}).get("episodes", 1000)
    
    print(f"Total episodes: {total_episodes}")
    print(f"Workers: {args.workers}")
    print()

    assignments = distribute_episodes(total_episodes, args.workers)
    
    for i, (start, end, count) in enumerate(assignments):
        worker_id = i + 1
        print(f"Worker {worker_id:<3} {start:>5} - {end:<5} ({count:>3} eps)")
    
    print()
    
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, end, count) in enumerate(assignments):
        worker_id = i + 1
        generate_worker_config(
            base_config, worker_id, start, end, count, output_dir
        )
    
    merge_script = output_dir / "merge_checkpoints.md"
    with open(merge_script, "w", encoding="utf-8") as f:
        for i in range(args.workers):
            worker_id = i + 1
            f.write(f"copy models\\parallel\\worker{worker_id:03d}\\*_best.pt models\\parallel\\merged\\\n")
        
        f.write("\n")
        
        for i in range(1, args.workers):
            prev_id = i
            curr_id = i + 1
            f.write(f"python scripts/train.py --config configs/parallel/train_worker{curr_id:03d}.yaml --resume models/parallel/worker{prev_id:03d}/*_best.pt\n")


if __name__ == "__main__":
    main()
