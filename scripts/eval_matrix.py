#!/usr/bin/env python
"""
Eval Matrix - Run all policies × demands × seeds for systematic comparison

Usage:
    python scripts/eval_matrix.py --mode quick      # 3 demands × 3 seeds
    python scripts/eval_matrix.py --mode final      # 3 demands × 5 seeds
    python scripts/eval_matrix.py --policies fixed,max_pressure,rl_full --demands 600,800

Policies available:
    - fixed: Fixed-time controller (cycle=90, split=50/50)
    - max_pressure: MaxPressure controller
    - actuated: Actuated controller (gap-out based)
    - webster: Webster formula controller
    - rl_plain: RL without advanced components
    - rl_full: Full RL with all components

Output:
    results/eval_matrix.csv     - All evaluation results
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_HORIZON = 1500
DEFAULT_WARMUP = 300

QUICK_DEMANDS = [600, 800, 1000]
QUICK_SEEDS = [42, 43, 44]

FINAL_DEMANDS = [600, 800, 1000]
FINAL_SEEDS = [42, 43, 44, 45, 46]

ALL_POLICIES = ["fixed", "max_pressure", "actuated", "webster", "rl_plain", "rl_full"]
DEFAULT_POLICIES = ["fixed", "max_pressure", "actuated", "rl_full"]

CONFIGS = {
    "rl_full": "configs/train_1.yaml",
    "rl_plain": "configs/train_1_plain.yaml",
    "baseline": "configs/train_1.yaml",  # Used for non-RL policies
}

MODELS = {
    "rl_full": "models/1/best_model.pt",
    "rl_plain": "models/1_plain/best_model.pt",
}


@dataclass
class EvalResult:
    """Single evaluation run result."""
    policy: str
    demand: int
    seed: int
    horizon_sec: int
    warmup_sec: int
    
    # Core metrics (corrected for teleport/missing)
    avg_wait_time_corr: float
    avg_travel_time_corr: float
    throughput_corr: float
    completion_rate: float
    teleport_rate: float
    
    # Additional metrics
    arrived_vehicles: int
    n_present_end: int
    avg_queue: float
    total_reward: float
    episode_steps: int
    
    # Meta
    status: str
    error_msg: str = ""
    route_file: str = ""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation matrix: policies × demands × seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "final", "custom"],
        default="quick",
        help="Preset mode: quick (3 seeds) or final (5 seeds)"
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help=f"Comma-separated policies (available: {', '.join(ALL_POLICIES)})"
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
        help=f"Simulation horizon (default: {DEFAULT_HORIZON})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup period (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_matrix.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--unseen",
        action="store_true",
        help="Use hold-out route manifest for generalization test"
    )
    parser.add_argument(
        "--rl-full-model",
        type=str,
        default=None,
        help="Override path to RL-Full model checkpoint"
    )
    parser.add_argument(
        "--rl-plain-model",
        type=str,
        default=None,
        help="Override path to RL-Plain model checkpoint"
    )
    
    return parser.parse_args(argv)


def run_single_eval(
    policy: str,
    demand: int,
    seed: int,
    horizon: int,
    warmup: int,
    config_path: str,
    model_path: Optional[str] = None,
    unseen: bool = False,
) -> EvalResult:
    """
    Run a single evaluation episode.
    """
    from rl.utils import load_yaml_config, set_global_seed
    from scripts.common import build_env, build_agent
    from controllers import (
        FixedTimeController, FixedTimeControllerConfig,
        ActuatedController, ActuatedControllerConfig,
        WebsterController, WebsterControllerConfig,
        select_action_from_defs,
    )
    
    try:
        # Load config
        config = load_yaml_config(config_path)
        config = copy.deepcopy(config)
        
        # Override horizon
        config['env']['sumo']['max_sim_seconds'] = horizon
        config['run']['seed'] = seed
        
        # Set route based on demand and seed
        manifest_name = "manifest_holdout.txt" if unseen else f"manifest_d{demand}.txt"
        manifest_dir = "networks/variants/eval" if unseen else "networks/variants/train_1000s"
        manifest_path = project_root / manifest_dir / manifest_name
        
        route_file = ""
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                routes = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            if routes:
                route_file = routes[seed % len(routes)]
                full_route_path = project_root / manifest_dir / route_file
                config['env']['sumo']['route_file'] = str(full_route_path)
        
        # Build env
        set_global_seed(seed)
        env = build_env(config)
        action_defs = getattr(env, '_action_defs', [])
        
        # Setup controller/agent
        agent = None
        controller = None
        
        if policy == "fixed":
            fc = FixedTimeControllerConfig(target_split=(0.5, 0.5), target_cycle_sec=90)
            controller = FixedTimeController(action_defs, fc)
        elif policy == "max_pressure":
            pass  # Use select_action_from_defs inline
        elif policy == "actuated":
            controller = ActuatedController(action_defs)
        elif policy == "webster":
            controller = WebsterController(action_defs)
        elif policy in ("rl_full", "rl_plain"):
            if model_path and Path(model_path).exists():
                agent = build_agent(config, env)
                import torch
                ckpt = torch.load(model_path, map_location="cpu")
                if 'model_state_dict' in ckpt:
                    agent._policy_net.load_state_dict(ckpt['model_state_dict'])
                else:
                    agent._policy_net.load_state_dict(ckpt)
                agent._policy_net.eval()
            else:
                # Fallback to random if no model
                controller = FixedTimeController(action_defs, FixedTimeControllerConfig())
        
        # Reset controller state
        if hasattr(controller, 'reset'):
            controller.reset()
        
        # Run episode
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            # Select action
            if isinstance(state, dict):
                tls_ids = sorted(state.keys())
                actions = {}
                
                for tls in tls_ids:
                    tls_state = state.get(tls, np.zeros(4))
                    
                    if agent is not None:
                        # RL agent
                        import torch
                        with torch.no_grad():
                            s = torch.as_tensor(tls_state, dtype=torch.float32).unsqueeze(0)
                            q = agent._policy_net(s)
                            actions[tls] = int(q.argmax(dim=1).item())
                    elif controller is not None:
                        if hasattr(controller, 'act'):
                            sim_time = getattr(env, '_current_time', step_count * 90)
                            actions[tls] = controller.act(tls_state, tls, sim_time)
                        else:
                            actions[tls] = controller.act()
                    else:
                        # MaxPressure
                        actions[tls] = select_action_from_defs(
                            tls_state, action_defs,
                            default_action_id=7
                        )
                
                next_state, rewards, done, info = env.step(actions)
                reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
                total_reward += float(np.mean(reward_values))
            else:
                if agent is not None:
                    import torch
                    with torch.no_grad():
                        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                        q = agent._policy_net(s)
                        action = int(q.argmax(dim=1).item())
                elif controller is not None:
                    if hasattr(controller, 'act') and policy in ("actuated", "webster"):
                        action = controller.act(state)
                    else:
                        action = controller.act()
                else:
                    action = select_action_from_defs(state, action_defs, default_action_id=7)
                
                next_state, reward, done, info = env.step(action)
                total_reward += float(reward)
            
            state = next_state
            step_count += 1
        
        # Get KPIs
        kpi = {}
        if hasattr(env, '_kpi_tracker') and env._kpi_tracker is not None:
            kpi = env._kpi_tracker.summary_dict()
        elif isinstance(info, dict):
            kpi = info.get('episode_kpi', {})
        
        # Get n_present
        n_present_end = 0
        if hasattr(env, '_traci') and env._traci is not None:
            try:
                n_present_end = int(env._traci.vehicle.getIDCount())
            except:
                pass
        
        env.close()
        
        return EvalResult(
            policy=policy,
            demand=demand,
            seed=seed,
            horizon_sec=horizon,
            warmup_sec=warmup,
            avg_wait_time_corr=float(kpi.get('avg_wait_time_corr', 0)),
            avg_travel_time_corr=float(kpi.get('avg_travel_time_corr', 0)),
            throughput_corr=float(kpi.get('throughput_corr', 0)),
            completion_rate=float(kpi.get('completion_rate', 0)),
            teleport_rate=float(kpi.get('teleport_rate', 0)),
            arrived_vehicles=int(kpi.get('arrived_vehicles', 0)),
            n_present_end=n_present_end,
            avg_queue=float(kpi.get('avg_queue', 0)),
            total_reward=total_reward,
            episode_steps=step_count,
            status="OK",
            route_file=route_file,
        )
        
    except Exception as e:
        import traceback
        return EvalResult(
            policy=policy,
            demand=demand,
            seed=seed,
            horizon_sec=horizon,
            warmup_sec=warmup,
            avg_wait_time_corr=0,
            avg_travel_time_corr=0,
            throughput_corr=0,
            completion_rate=0,
            teleport_rate=0,
            arrived_vehicles=0,
            n_present_end=0,
            avg_queue=0,
            total_reward=0,
            episode_steps=0,
            status="ERROR",
            error_msg=str(e)[:200],
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Resolve parameters
    if args.policies:
        policies = [p.strip() for p in args.policies.split(",")]
    else:
        policies = DEFAULT_POLICIES
    
    if args.demands:
        demands = [int(d.strip()) for d in args.demands.split(",")]
    else:
        demands = QUICK_DEMANDS if args.mode == "quick" else FINAL_DEMANDS
    
    if args.seeds:
        seeds = list(range(42, 42 + args.seeds))
    else:
        seeds = QUICK_SEEDS if args.mode == "quick" else FINAL_SEEDS
    
    # Resolve model paths
    models = {
        "rl_full": args.rl_full_model or str(project_root / MODELS["rl_full"]),
        "rl_plain": args.rl_plain_model or str(project_root / MODELS["rl_plain"]),
    }
    
    print("=" * 70)
    print("EVAL MATRIX")
    print("=" * 70)
    print(f"Policies: {policies}")
    print(f"Demands:  {demands}")
    print(f"Seeds:    {seeds}")
    print(f"Horizon:  {args.horizon}s")
    print(f"Unseen:   {args.unseen}")
    print("=" * 70)
    
    # Build task list
    total_tasks = len(policies) * len(demands) * len(seeds)
    print(f"Total evaluations: {total_tasks}")
    
    results: List[EvalResult] = []
    completed = 0
    start_time = time.time()
    
    for policy in policies:
        # Determine config and model
        if policy in ("rl_full", "rl_plain"):
            config_path = str(project_root / CONFIGS[policy])
            model_path = models[policy]
        else:
            config_path = str(project_root / CONFIGS["baseline"])
            model_path = None
        
        for demand in demands:
            for seed in seeds:
                completed += 1
                print(f"[{completed}/{total_tasks}] {policy} d={demand} s={seed}...", end=" ", flush=True)
                
                result = run_single_eval(
                    policy=policy,
                    demand=demand,
                    seed=seed,
                    horizon=args.horizon,
                    warmup=args.warmup,
                    config_path=config_path,
                    model_path=model_path,
                    unseen=args.unseen,
                )
                
                results.append(result)
                
                status_icon = "✅" if result.status == "OK" else "❌"
                print(f"{status_icon} wait={result.avg_wait_time_corr:.1f}s comp={result.completion_rate:.2%}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "policy", "demand", "seed", "horizon_sec", "warmup_sec",
        "avg_wait_time_corr", "avg_travel_time_corr", "throughput_corr",
        "completion_rate", "teleport_rate", "arrived_vehicles",
        "n_present_end", "avg_queue", "total_reward", "episode_steps",
        "status", "error_msg", "route_file",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "policy": r.policy,
                "demand": r.demand,
                "seed": r.seed,
                "horizon_sec": r.horizon_sec,
                "warmup_sec": r.warmup_sec,
                "avg_wait_time_corr": r.avg_wait_time_corr,
                "avg_travel_time_corr": r.avg_travel_time_corr,
                "throughput_corr": r.throughput_corr,
                "completion_rate": r.completion_rate,
                "teleport_rate": r.teleport_rate,
                "arrived_vehicles": r.arrived_vehicles,
                "n_present_end": r.n_present_end,
                "avg_queue": r.avg_queue,
                "total_reward": r.total_reward,
                "episode_steps": r.episode_steps,
                "status": r.status,
                "error_msg": r.error_msg,
                "route_file": r.route_file,
            })
    
    print(f"\n[OUTPUT] Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
