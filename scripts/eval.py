#!/usr/bin/env python
"""
Unified Evaluation Script - Combines eval.py and eval_matrix.py functionality

Usage:
    # Using config file (recommended)
    python scripts/eval_unified.py --config configs/eval.yaml
    
    # Override config parameters via CLI
    python scripts/eval_unified.py --config configs/eval.yaml --policies rl --demands 1000
    
    # Quick preset modes
    python scripts/eval_unified.py --config configs/eval.yaml --mode quick   # 3 demands × 3 seeds
    python scripts/eval_unified.py --config configs/eval.yaml --mode final   # 3 demands × 5 seeds

Policies available:
    - fixed: Fixed-time controller (cycle=90, split=50/50)
    - max_pressure: MaxPressure controller
    - actuated: Actuated controller (gap-out based)
    - webster: Webster formula controller
    - rl_plain: RL without advanced components
    - rl: Trained RL model

Output:
    results/eval_results.csv - All evaluation results
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

ALL_POLICIES = ["fixed", "max_pressure", "actuated", "webster", "random", "rl"]
DEFAULT_POLICIES = ["fixed", "max_pressure", "actuated", "rl"]


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
        description="Unified evaluation: policies × demands × seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config YAML file"
    )
    parser.add_argument(
        "--mode", 
        choices=["quick", "final", "custom"],
        default=None,
        help="Preset mode (overrides config): quick (3 seeds) or final (5 seeds)"
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
        default=None,
        help=f"Simulation horizon (default: {DEFAULT_HORIZON})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=f"Warmup period (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (overrides config)"
    )
    parser.add_argument(
        "--unseen",
        action="store_true",
        default=None,
        help="Use hold-out route manifest for generalization test"
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Override path to RL model checkpoint"
    )
    parser.add_argument(
        "--route-manifest",
        type=str,
        default=None,
        help="Override route manifest path (applies to all demands/seeds)"
    )
    
    return parser.parse_args(argv)


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Load YAML config and apply CLI overrides."""
    from rl.utils import load_yaml_config
    
    config = load_yaml_config(args.config)
    eval_cfg = config.get("eval_matrix", config.get("eval", {}))
    
    # Apply mode preset first
    mode = args.mode or eval_cfg.get("mode", "custom")
    
    # Resolve policies
    if args.policies:
        policies = [p.strip() for p in args.policies.split(",")]
    elif "policies" in eval_cfg:
        policies = eval_cfg["policies"]
    else:
        policies = QUICK_DEMANDS if mode == "quick" else DEFAULT_POLICIES
    
    # Resolve demands
    if args.demands:
        demands = [int(d.strip()) for d in args.demands.split(",")]
    elif "demands" in eval_cfg:
        demands = eval_cfg["demands"]
    else:
        demands = QUICK_DEMANDS if mode == "quick" else FINAL_DEMANDS
    
    # Resolve seeds
    if args.seeds:
        seeds = list(range(42, 42 + args.seeds))
    elif "seeds" in eval_cfg:
        n_seeds = eval_cfg["seeds"]
        seeds = list(range(42, 42 + n_seeds)) if isinstance(n_seeds, int) else n_seeds
    else:
        seeds = QUICK_SEEDS if mode == "quick" else FINAL_SEEDS
    
    # Resolve other parameters
    horizon = args.horizon or eval_cfg.get("horizon", DEFAULT_HORIZON)
    warmup = args.warmup or eval_cfg.get("warmup", DEFAULT_WARMUP)
    
    # Unseen flag (CLI takes precedence, then config, default False)
    unseen = args.unseen if args.unseen is not None else eval_cfg.get("unseen", False)
    
    # Output path
    output = args.output or eval_cfg.get("output", "results/eval_results.csv")
    
    # Model paths
    models_cfg = config.get("models", {})
    rl_model = args.rl_model or models_cfg.get("rl", "models/final_design/parallel_final.pt")

    # Optional route manifest override
    route_manifest_cfg = None
    if "route_manifest" in eval_cfg:
        route_manifest_cfg = eval_cfg.get("route_manifest")
    route_manifest = args.route_manifest or route_manifest_cfg
    
    return {
        "base_config": config,
        "policies": policies,
        "demands": demands,
        "seeds": seeds,
        "horizon": horizon,
        "warmup": warmup,
        "unseen": unseen,
        "output": output,
        "models": {
            "rl": rl_model,
        },
        "route_manifest": route_manifest,
    }


def _select_route_from_manifest(manifest_path: Path, demand: int, seed: int) -> Tuple[str, Path]:
    """Load manifest, filter by demand suffix if available, and pick deterministic route."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        routes = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    if not routes:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    demand_routes = [r for r in routes if f"_d{demand}" in r]
    if demand_routes:
        routes = demand_routes

    route_file = routes[seed % len(routes)]
    full_route_path = (manifest_path.parent / route_file).resolve()
    if not full_route_path.exists():
        raise FileNotFoundError(f"Route file from manifest missing: {full_route_path}")

    return route_file, full_route_path


def run_single_eval(
    policy: str,
    demand: int,
    seed: int,
    horizon: int,
    warmup: int,
    base_config: Dict[str, Any],
    model_path: Optional[str] = None,
    unseen: bool = False,
    route_manifest: Optional[str] = None,
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
        FlexibleMaxPressureController,
        select_action_from_defs,
    )
    
    try:
        # Deep copy config
        config = copy.deepcopy(base_config)
        
        # Override horizon and seed
        config['env']['sumo']['max_sim_seconds'] = horizon
        config['run']['seed'] = seed
        
        # Set route based on demand and seed
        route_file = ""
        if route_manifest:
            manifest_path = Path(route_manifest)
            if not manifest_path.is_absolute():
                manifest_path = project_root / manifest_path
            route_file, full_route_path = _select_route_from_manifest(manifest_path, demand, seed)
            config['env']['sumo']['route_file'] = str(full_route_path)
            print(f"    [MANIFEST] Using route: {Path(route_file).name}")
        elif unseen:
            unseen_manifest = project_root / "networks/variants/train_turn801010/manifest_eval_unseen.txt"
            route_file, full_route_path = _select_route_from_manifest(unseen_manifest, demand, seed)
            # Validate no overlap with training manifests
            train_manifest = project_root / f"networks/variants/train_turn801010/{demand}/manifest.txt"
            if train_manifest.exists():
                with open(train_manifest, 'r') as f:
                    train_routes = set(l.strip() for l in f if l.strip() and not l.startswith('#'))
                overlap = train_routes & {Path(route_file).name}
                if overlap:
                    raise ValueError(f"Route overlap detected between train and unseen: {overlap}")
            
            config['env']['sumo']['route_file'] = str(full_route_path)
            print(f"    [UNSEEN] Using route: {Path(route_file).name}")
        else:
            # Standard training routes
            manifest_dir = f"networks/variants/train_turn801010/{demand}"
            manifest_path = project_root / manifest_dir / "manifest.txt"
            
            if manifest_path.exists():
                route_file, full_route_path = _select_route_from_manifest(manifest_path, demand, seed)
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
            controller = FlexibleMaxPressureController(
                action_defs=action_defs,
                min_green_sec=10.0,
                max_green_sec=60.0,
            )
        elif policy == "actuated":
            controller = ActuatedController(action_defs)
        elif policy == "webster":
            controller = WebsterController(action_defs)
        elif policy == "random":
            from controllers.random_controller import RandomController
            controller = RandomController(action_defs, seed=seed)
        elif policy == "rl":
            if model_path and Path(model_path).exists():
                agent = build_agent(config, env)
                if isinstance(agent, tuple):
                    agent, _device = agent
                policy_net = getattr(agent, "_policy_net", None)
                if policy_net is None and hasattr(agent, "online_net"):
                    policy_net = agent.online_net
                if policy_net is None:
                    raise AttributeError("Agent has no policy network attribute")

                import torch
                ckpt = torch.load(model_path, map_location="cpu")
                if 'online_state_dict' in ckpt:
                    state_dict = ckpt['online_state_dict']
                else:
                    state_dict = ckpt.get('model_state_dict', ckpt)
                policy_net.load_state_dict(state_dict, strict=False)
                policy_net.eval()
            else:
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
            if isinstance(state, dict):
                tls_ids = sorted(state.keys())
                actions = {}
                
                for tls in tls_ids:
                    tls_state = state.get(tls, np.zeros(4))
                    
                    if agent is not None:
                        import torch
                        with torch.no_grad():
                            s = torch.as_tensor(tls_state, dtype=torch.float32).unsqueeze(0)
                            policy_net = getattr(agent, "_policy_net", None)
                            if policy_net is None and hasattr(agent, "online_net"):
                                policy_net = agent.online_net
                            q = policy_net(s)
                            actions[tls] = int(q.argmax(dim=1).item())
                    elif controller is not None:
                        if hasattr(controller, 'act'):
                            sim_time = getattr(env, '_current_time', step_count * 90)
                            actions[tls] = controller.act(tls_state, tls, sim_time)
                        else:
                            actions[tls] = controller.act()
                    else:
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
                        policy_net = getattr(agent, "_policy_net", None)
                        if policy_net is None and hasattr(agent, "online_net"):
                            policy_net = agent.online_net
                        q = policy_net(s)
                        action = int(q.argmax(dim=1).item())
                elif controller is not None:
                    if hasattr(controller, 'act') and policy in ("actuated", "webster", "max_pressure"):
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
        traceback.print_exc()
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
    eval_params = load_config_with_overrides(args)
    
    policies = eval_params["policies"]
    demands = eval_params["demands"]
    seeds = eval_params["seeds"]
    horizon = eval_params["horizon"]
    warmup = eval_params["warmup"]
    unseen = eval_params["unseen"]
    output = eval_params["output"]
    models = eval_params["models"]
    base_config = eval_params["base_config"]
    route_manifest = eval_params["route_manifest"]
    
    print("=" * 70)
    print("UNIFIED EVALUATION")
    print("=" * 70)
    print(f"Config:   {args.config}")
    print(f"Policies: {policies}")
    print(f"Demands:  {demands}")
    print(f"Seeds:    {seeds}")
    print(f"Horizon:  {horizon}s")
    print(f"Warmup:   {warmup}s")
    print(f"Unseen:   {unseen}")
    print(f"Output:   {output}")
    print("=" * 70)
    
    # Build task list
    total_tasks = len(policies) * len(demands) * len(seeds)
    print(f"Total evaluations: {total_tasks}")
    
    results: List[EvalResult] = []
    completed = 0
    start_time = time.time()
    
    for policy in policies:
        # Determine model path
        if policy == "rl":
            model_path = models[policy]
        else:
            model_path = None
        
        for demand in demands:
            for seed in seeds:
                completed += 1
                print(f"[{completed}/{total_tasks}] {policy} d={demand} s={seed}...", end=" ", flush=True)
                
                result = run_single_eval(
                    policy=policy,
                    demand=demand,
                    seed=seed,
                    horizon=horizon,
                    warmup=warmup,
                    base_config=base_config,
                    model_path=model_path,
                    unseen=unseen,
                    route_manifest=route_manifest,
                )
                
                results.append(result)
                
                status_icon = "[OK]" if result.status == "OK" else "[FAIL]"
                print(f"{status_icon} wait={result.avg_wait_time_corr:.1f}s comp={result.completion_rate:.2%}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Save results
    output_path = Path(output)
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
