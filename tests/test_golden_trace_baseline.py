"""
Golden-trace baseline test for performance optimization invariance verification.

Usage:
    python tests/test_golden_trace_baseline.py --seed 42 --workers 1 --episodes 2 --save-baseline
    python tests/test_golden_trace_baseline.py --seed 42 --workers 1 --episodes 2 --compare-baseline

This script:
1. Runs a short training session with fixed seed
2. Records decision-step data: (action, state, reward, done, gamma)
3. Saves/compares golden trace for invariance testing

Acceptance criteria:
- Actions: exact match
- Floats: np.allclose(atol=1e-7, rtol=1e-6)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure single-threaded for determinism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.perf_utils import (
    GoldenTraceRecorder, 
    TransitionCounters, 
    TimingBreakdown,
    compute_throughput,
)
from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env, build_agent


BASELINE_DIR = project_root / "tests" / "golden_trace_baselines"


def run_baseline_test(
    config_path: str,
    seed: int,
    episodes: int,
    save_baseline: bool = False,
    compare_baseline: bool = False,
) -> dict:
    """
    Run baseline test and optionally save/compare golden trace.
    
    Returns dict with:
    - passed: bool
    - message: str
    - counters: TransitionCounters
    - throughput: dict
    - timing: dict
    """
    print(f"[GoldenTrace] Config: {config_path}")
    print(f"[GoldenTrace] Seed: {seed}, Episodes: {episodes}")
    
    # Load config and override for baseline test
    config = load_yaml_config(config_path)
    config["run"] = config.get("run", {})
    config["run"]["seed"] = seed
    
    # Force single worker, no parallel
    config["parallel"] = config.get("parallel", {})
    config["parallel"]["num_actors"] = 1
    
    # Set deterministic exploration (fixed epsilon for reproducibility)
    config["exploration"] = config.get("exploration", {})
    config["exploration"]["eps_start"] = 0.5
    config["exploration"]["eps_end"] = 0.5
    config["exploration"]["warmup_global_steps"] = 0
    config["exploration"]["eps_decay_steps"] = 1
    
    # Override route_file to use an existing file for baseline test
    existing_route = "networks/variants/train/800/bignet_train_seed00042_d800.rou.xml"
    config.setdefault("env", {}).setdefault("sumo", {})["route_file"] = existing_route
    
    # Disable curriculum for simple baseline test
    config["curriculum"] = {"enabled": False}
    
    set_global_seed(seed)
    
    # Build env and agent
    env = build_env(config)
    agent, device = build_agent(config, env)
    
    # Initialize tracking
    recorder = GoldenTraceRecorder()
    counters = TransitionCounters()
    timing = TimingBreakdown()
    
    total_transitions = 0
    start_time = time.time()
    
    print(f"[GoldenTrace] Starting {episodes} episodes...")
    
    for ep in range(episodes):
        timing.start("episode_total")
        
        timing.start("env_reset")
        state = env.reset()
        timing.stop("env_reset")
        
        done = False
        ep_steps = 0
        
        while not done:
            # Handle multi-TLS state (use first TLS for action selection)
            if isinstance(state, dict):
                state_for_action = list(state.values())[0]
            else:
                state_for_action = state
            
            timing.start("action_select")
            action = agent.select_action(state_for_action, epsilon=0.5)
            timing.stop("action_select")
            
            # Prepare action for env
            if isinstance(state, dict):
                actions = {tls_id: action for tls_id in state.keys()}
            else:
                actions = action
            
            timing.start("env_step")
            next_state, reward, done, info = env.step(actions)
            timing.stop("env_step")
            
            # Get scalar reward and gamma
            if isinstance(reward, dict):
                reward_scalar = sum(reward.values())
            else:
                reward_scalar = float(reward)
            
            gamma = info.get("gamma", 0.98)
            
            # Record for golden trace
            recorder.record(
                action=action,
                state=state_for_action if isinstance(state_for_action, np.ndarray) else np.array(state_for_action),
                reward=reward_scalar,
                done=done,
                gamma=gamma,
            )
            
            # Count transitions
            total_transitions += 1
            ep_steps += 1
            
            state = next_state
        
        timing.stop("episode_total")
        print(f"[GoldenTrace] Episode {ep+1}/{episodes}: {ep_steps} steps")
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Update counters (simulating producer side)
    counters.produced_transitions = total_transitions
    counters.consumed_transitions = total_transitions  # Same for single process
    
    env.close()
    
    # Compute throughput
    throughput = compute_throughput(total_transitions, wall_time)
    
    print(f"\n[GoldenTrace] Results:")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  decision_steps/sec: {throughput['decision_steps_per_sec']:.2f}")
    print(timing.summary_str())
    
    # No-drop verification
    no_drop_passed, no_drop_msg = counters.verify_no_drop()
    print(f"  No-drop check: {'PASS' if no_drop_passed else 'FAIL'} ({no_drop_msg})")
    
    # Save/compare baseline
    baseline_path = BASELINE_DIR / f"golden_trace_seed{seed}_ep{episodes}.json"
    
    result = {
        "passed": True,
        "message": "",
        "counters": counters.to_dict(),
        "throughput": throughput,
        "timing": timing.get_stats(),
        "trace_hash": recorder.compute_hash(),
    }
    
    if save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        recorder.save(baseline_path)
        
        # Also save metadata
        meta_path = BASELINE_DIR / f"golden_trace_seed{seed}_ep{episodes}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "seed": seed,
                "episodes": episodes,
                "config_path": config_path,
                "counters": counters.to_dict(),
                "throughput": throughput,
                "trace_hash": recorder.compute_hash(),
            }, f, indent=2)
        
        print(f"\n[GoldenTrace] Baseline saved to: {baseline_path}")
        result["message"] = f"Baseline saved to {baseline_path}"
    
    if compare_baseline:
        if not baseline_path.exists():
            result["passed"] = False
            result["message"] = f"Baseline not found: {baseline_path}"
            print(f"\n[GoldenTrace] ERROR: {result['message']}")
        else:
            baseline_recorder = GoldenTraceRecorder.load(baseline_path)
            passed, msg = recorder.compare(baseline_recorder)
            result["passed"] = passed
            result["message"] = msg
            
            if passed:
                print(f"\n[GoldenTrace] PASS: {msg}")
            else:
                print(f"\n[GoldenTrace] FAIL: {msg}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Golden-trace baseline test")
    parser.add_argument("--config", default="configs/train_final_design.yaml",
                        help="Config file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save golden trace as baseline")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare against saved baseline")
    
    args = parser.parse_args()
    
    result = run_baseline_test(
        config_path=args.config,
        seed=args.seed,
        episodes=args.episodes,
        save_baseline=args.save_baseline,
        compare_baseline=args.compare_baseline,
    )
    
    # Exit with appropriate code
    if args.compare_baseline:
        sys.exit(0 if result["passed"] else 1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
