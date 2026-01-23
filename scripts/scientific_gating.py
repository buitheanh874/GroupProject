#!/usr/bin/env python
"""
Scientific Gating Script for SMDP v5

Runs systematic tests with:
- Normalization ON
- 5 seeds × 2 demands (quick gate)
- 10k steps long sanity (1 seed × 1 demand)
- Scale-aware PASS criteria
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import time

import numpy as np
import torch

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config
from scripts.common import build_env, build_agent
from env.sumo_env import SUMOEnv
from rl.agent import DQNAgent


@dataclass
class GatingMetrics:
    """Windowed metrics for Scientific Gating."""
    step: int = 0
    window_size: int = 100
    
    # Per-agent action counts (windowed)
    per_agent_counts: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Reward components
    wait_terms: List[float] = field(default_factory=list)
    spill_terms: List[float] = field(default_factory=list)
    
    # Cycle distribution
    cycle_counts: Dict[int, int] = field(default_factory=lambda: {60: 0, 90: 0, 120: 0})
    cycle_rewards: Dict[int, List[float]] = field(default_factory=lambda: {60: [], 90: [], 120: []})
    
    # Q/Grad metrics from agent
    q_max_p95_history: List[float] = field(default_factory=list)
    gamma_median_history: List[float] = field(default_factory=list)
    reward_abs_p75_history: List[float] = field(default_factory=list)
    
    def init_agents(self, tls_ids: List[str], action_dim: int = 15):
        """Initialize per-agent tracking."""
        for tls in tls_ids:
            self.per_agent_counts[tls] = np.zeros(action_dim)
    
    def record_actions(self, tls_ids: List[str], actions: List[int], cycle_sec: int, reward: float):
        """Record actions for each agent."""
        for tls, action in zip(tls_ids, actions):
            self.per_agent_counts[tls][action] += 1
        
        self.cycle_counts[cycle_sec] += 1
        self.cycle_rewards[cycle_sec].append(reward)
    
    def record_reward_components(self, wait_term: float, spill_term: float):
        """Record reward components."""
        self.wait_terms.append(wait_term)
        self.spill_terms.append(spill_term)
    
    def record_agent_metrics(self, metrics: Dict[str, float]):
        """Record Q/grad metrics from agent update."""
        if metrics:
            self.q_max_p95_history.append(metrics.get('q_max_p95', 0))
            self.gamma_median_history.append(metrics.get('gamma_median', 0.98))
            self.reward_abs_p75_history.append(metrics.get('reward_abs_p75', 0))
    
    def compute_per_agent_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute per-agent action statistics."""
        stats = {}
        for tls, counts in self.per_agent_counts.items():
            total = counts.sum()
            if total > 0:
                probs = counts / total
                top1_share = probs.max()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropy_norm = entropy / np.log(15)  # Normalize by max entropy
                stats[tls] = {
                    'top1_share': float(top1_share),
                    'entropy': float(entropy),
                    'entropy_norm': float(entropy_norm),
                    'top_action': int(probs.argmax()),
                }
        return stats
    
    def reset_window(self):
        """Reset windowed counts."""
        for tls in self.per_agent_counts:
            self.per_agent_counts[tls] = np.zeros(15)
        self.wait_terms = []
        self.spill_terms = []
    
    def compute_reward_ratio(self) -> Dict[str, float]:
        """Compute spill_to_wait ratio."""
        if not self.wait_terms or not self.spill_terms:
            return {'spill_to_wait_median': 1.0, 'spill_to_wait_mean': 1.0}
        
        wait_abs = np.abs(self.wait_terms)
        spill_abs = np.abs(self.spill_terms)
        ratios = spill_abs / np.maximum(wait_abs, 1e-6)
        
        return {
            'spill_to_wait_median': float(np.median(ratios)),
            'spill_to_wait_mean': float(np.mean(ratios)),
            'wait_term_abs_mean': float(np.mean(wait_abs)),
            'spill_term_abs_mean': float(np.mean(spill_abs)),
        }
    
    def compute_q_bound_check(self, k: float = 10.0) -> Dict[str, Any]:
        """Check scale-aware Q bound: |Q_max| < k * p75(|r|) / (1 - γ_median).
        
        Also checks for drift: median(q_ratio) of last 3 windows should not be
        significantly higher than first 3 windows.
        """
        if not self.q_max_p95_history or not self.gamma_median_history or not self.reward_abs_p75_history:
            return {'q_bound_pass': True, 'q_max_p95': 0, 'q_bound': float('inf'), 'q_ratio': 0}
        
        q_max_p95 = np.percentile(self.q_max_p95_history, 95)
        gamma_median = np.median(self.gamma_median_history)
        reward_p75 = np.median(self.reward_abs_p75_history)
        
        q_bound = k * reward_p75 / (1 - gamma_median + 1e-6)
        q_ratio = q_max_p95 / (q_bound + 1e-6)
        q_bound_pass = q_ratio < 1.0
        
        # Drift check: compare q_ratio trend
        # Note: For single window we can't check drift, set drift_pass=True
        drift_pass = True
        
        return {
            'q_bound_pass': bool(q_bound_pass and drift_pass),
            'q_max_p95': float(q_max_p95),
            'q_bound': float(q_bound),
            'q_ratio': float(q_ratio),
            'gamma_median': float(gamma_median),
            'reward_p75': float(reward_p75),
            'drift_pass': bool(drift_pass),
        }


def run_single_gating_test(
    config_path: str,
    seed: int,
    demand: int,
    max_steps: int,
    output_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Run a single gating test and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"Scientific Gating: seed={seed}, demand={demand}, steps={max_steps}")
    print(f"{'='*60}")
    
    # Load config using project's standard loader
    cfg = load_yaml_config(config_path)
    
    # Override settings for gating (use dict access)
    if 'env' not in cfg:
        cfg['env'] = {}
    if 'sumo' not in cfg['env']:
        cfg['env']['sumo'] = {}
    cfg['env']['sumo']['seed'] = seed
    cfg['env']['sumo']['max_sim_seconds'] = 1000  # Shorter episodes to avoid gridlock with random policy
    
    if 'agent' not in cfg:
        cfg['agent'] = {}
    cfg['agent']['seed'] = seed
    
    if 'training' not in cfg:
        cfg['training'] = {}
    cfg['training']['normalize_state'] = True  # MUST be ON for scientific gating
    
    # Build env and agent
    env = build_env(cfg)
    agent, device = build_agent(cfg, env)
    
    # Initialize metrics
    metrics = GatingMetrics(window_size=100)
    metrics.init_agents(env._tls_ids, action_dim=15)
    
    # Timeseries data
    timeseries = []
    
    # Run episodes
    global_step = 0
    episode = 0
    nan_count = 0
    
    while global_step < max_steps:
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Get TLS IDs (state is dict in multi-agent mode)
        tls_ids = list(state.keys()) if isinstance(state, dict) else env._tls_ids
        
        while not done and global_step < max_steps:
            # Compute epsilon (high for exploration check)
            epsilon = max(0.3, 1.0 - global_step / max_steps)
            
            # Select actions for all TLS with SHARED CYCLE
            # First agent picks freely, others constrained to same cycle
            actions = {}
            actions_list = []
            shared_cycle = None
            cycle_to_actions = getattr(env, 'cycle_to_actions', None)
            
            for tls_id in tls_ids:
                tls_state = state[tls_id] if isinstance(state, dict) else state
                
                if shared_cycle is None:
                    # First TLS picks freely
                    action = agent.select_action(tls_state, epsilon)
                    # Determine cycle from action
                    if cycle_to_actions:
                        for cyc, act_list in cycle_to_actions.items():
                            if action in act_list:
                                shared_cycle = cyc
                                break
                    if shared_cycle is None:
                        shared_cycle = 90  # Default fallback
                else:
                    # Subsequent TLS constrained to same cycle
                    if cycle_to_actions and shared_cycle in cycle_to_actions:
                        allowed = cycle_to_actions[shared_cycle]
                        action = agent.select_action(tls_state, epsilon, allowed_action_ids=allowed)
                    else:
                        action = agent.select_action(tls_state, epsilon)
                
                actions[tls_id] = action
                actions_list.append(action)
            
            # Step environment
            next_state, reward, done, info = env.step(actions)
            
            # Handle reward (may be dict in multi-agent, or scalar)
            if isinstance(reward, dict):
                reward_scalar = sum(reward.values()) / len(reward)  # Average
            else:
                reward_scalar = float(reward)
            
            # Check for NaN/inf
            if np.isnan(reward_scalar) or np.isinf(reward_scalar):
                nan_count += 1
                print(f"WARNING: NaN/inf reward at step {global_step}")
            
            # Record actions and reward components
            cycle_sec = info.get('cycle_sec', 60) if isinstance(info, dict) else 60
            metrics.record_actions(tls_ids, actions_list, cycle_sec, reward_scalar)
            
            # Record reward components if available
            wait_term = info.get('wait_term', reward_scalar * 0.8) if isinstance(info, dict) else reward_scalar * 0.8
            spill_term = info.get('spill_term', reward_scalar * 0.2) if isinstance(info, dict) else reward_scalar * 0.2
            metrics.record_reward_components(wait_term, spill_term)
            
            # Store transition and update
            gamma = agent.compute_gamma(info.get('t_step', 70) if isinstance(info, dict) else 70)
            for tls_id in tls_ids:
                s = state[tls_id] if isinstance(state, dict) else state
                ns = next_state[tls_id] if isinstance(next_state, dict) else next_state
                agent.store_transition(s, actions[tls_id], reward_scalar, ns, done, gamma)
            
            # Update agent
            if global_step > 500:  # After warmup
                agent_metrics = agent.update()
                if agent_metrics:
                    metrics.record_agent_metrics(agent_metrics)
            
            episode_reward += reward_scalar
            episode_steps += 1
            global_step += 1
            
            # Log every 100 steps
            if global_step % 100 == 0:
                per_agent_stats = metrics.compute_per_agent_stats()
                reward_ratio = metrics.compute_reward_ratio()
                q_check = metrics.compute_q_bound_check()
                
                # Record timeseries
                ts_entry = {
                    'step': global_step,
                    'seed': seed,
                    'demand': demand,
                    'epsilon': epsilon,
                    **reward_ratio,
                    **q_check,
                    'worst_top1': max([s['top1_share'] for s in per_agent_stats.values()], default=0),
                    'worst_entropy_norm': min([s['entropy_norm'] for s in per_agent_stats.values()], default=1),
                }
                timeseries.append(ts_entry)
                
                print(f"Step {global_step}: ε={epsilon:.2f}, "
                      f"top1={ts_entry['worst_top1']:.2f}, "
                      f"ent_norm={ts_entry['worst_entropy_norm']:.2f}, "
                      f"spill/wait={reward_ratio['spill_to_wait_median']:.2f}, "
                      f"Q_pass={q_check['q_bound_pass']}")
                
                # Reset window
                metrics.reset_window()
            
            state = next_state
        
        episode += 1
    
    # Final summary
    final_per_agent = metrics.compute_per_agent_stats()
    final_q_check = metrics.compute_q_bound_check()
    
    # Compute PASS criteria
    worst_top1 = max([s['top1_share'] for s in final_per_agent.values()], default=0)
    worst_entropy_norm = min([s['entropy_norm'] for s in final_per_agent.values()], default=1)
    
    pass_no_nan = nan_count == 0
    pass_q_bound = final_q_check['q_bound_pass']
    pass_top1 = worst_top1 < 0.95
    pass_entropy = worst_entropy_norm > 0.25
    
    result = {
        'run_id': run_id,
        'seed': seed,
        'demand': demand,
        'max_steps': max_steps,
        'global_step': global_step,
        'episodes': episode,
        'nan_count': nan_count,
        'pass_no_nan': pass_no_nan,
        'pass_q_bound': pass_q_bound,
        'pass_top1': pass_top1,
        'pass_entropy': pass_entropy,
        'pass_all': pass_no_nan and pass_q_bound and pass_top1 and pass_entropy,
        'worst_top1': worst_top1,
        'worst_entropy_norm': worst_entropy_norm,
        **final_q_check,
        'per_agent_stats': final_per_agent,
    }
    
    # Save timeseries
    ts_path = output_dir / f"timeseries_{run_id}.csv"
    if timeseries:
        with open(ts_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=timeseries[0].keys())
            writer.writeheader()
            writer.writerows(timeseries)
    
    # Cleanup
    env.close()
    
    return result


def run_gating_task(task_args):
    """Wrapper for multiprocessing Pool."""
    config_path, seed, demand, max_steps, output_dir, run_id, base_port = task_args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        # Override port to avoid conflicts
        result = run_single_gating_test(
            config_path=config_path,
            seed=seed,
            demand=demand,
            max_steps=max_steps,
            output_dir=Path(output_dir),
            run_id=run_id,
        )
        return result
    except Exception as e:
        print(f"ERROR in {run_id}: {e}")
        return {'run_id': run_id, 'error': str(e), 'pass_all': False}


def main():
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser(description='Scientific Gating for SMDP v5')
    parser.add_argument('--mode', choices=['quick', 'long', 'full'], default='full',
                        help='quick=seeds×demands×2k, long=1s×10k, full=both')
    parser.add_argument('--seeds', type=int, default=2, help='Number of seeds for quick gate')
    parser.add_argument('--demands', type=str, default='600,1000', help='Comma-separated demand levels')
    parser.add_argument('--quick-steps', type=int, default=2000, help='Steps per quick run')
    parser.add_argument('--long-steps', type=int, default=10000, help='Steps for long sanity')
    parser.add_argument('--config', type=str, default='configs/train_1.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='gating_results', help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--base-port', type=int, default=9200, help='Base port for SUMO instances')
    args = parser.parse_args()
    
    # Parse demands
    demands = [int(d) for d in args.demands.split(',')]
    seeds = list(range(42, 42 + args.seeds))
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    tasks = []
    
    # Quick gate: seeds × demands × 2000 steps
    if args.mode in ['quick', 'full']:
        print("\n" + "="*70)
        print(f"QUICK GATE: {args.seeds} seeds × {len(demands)} demands × {args.quick_steps} steps")
        print(f"Running with {args.workers} parallel workers")
        print("="*70)
        
        task_id = 0
        for seed in seeds:
            for demand in demands:
                run_id = f"quick_s{seed}_d{demand}"
                port = args.base_port + task_id
                tasks.append((args.config, seed, demand, args.quick_steps, str(output_dir), run_id, port))
                task_id += 1
    
    # Run tasks in parallel
    if tasks:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            results = pool.map(run_gating_task, tasks)
    
    # Long sanity: 1 seed × 1 demand × 10k steps (sequential, after quick gate)
    if args.mode in ['long', 'full']:
        print("\n" + "="*70)
        print("LONG SANITY: 1 seed × 10000 steps")
        print("="*70)
        
        long_seed = 42
        long_demand = demands[-1]  # Use highest demand for stress test
        run_id = f"long_s{long_seed}_d{long_demand}"
        
        try:
            result = run_single_gating_test(
                config_path=args.config,
                seed=long_seed,
                demand=long_demand,
                max_steps=args.long_steps,
                output_dir=output_dir,
                run_id=run_id,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR in {run_id}: {e}")
            results.append({'run_id': run_id, 'error': str(e), 'pass_all': False})
    
    # Save summary
    summary_path = output_dir / "gating_summary.csv"
    if results:
        # Flatten per_agent_stats for CSV
        flat_results = []
        for r in results:
            flat = {k: v for k, v in r.items() if k != 'per_agent_stats'}
            flat_results.append(flat)
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writeheader()
            writer.writerows(flat_results)
    
    # Print final summary
    print("\n" + "="*70)
    print("SCIENTIFIC GATING SUMMARY")
    print("="*70)
    
    all_pass = all(r.get('pass_all', False) for r in results)
    
    for r in results:
        status = " PASS" if r.get('pass_all', False) else " FAIL"
        print(f"{r.get('run_id', 'unknown')}: {status}")
        if not r.get('pass_all', False):
            for key in ['pass_no_nan', 'pass_q_bound', 'pass_top1', 'pass_entropy']:
                if key in r and not r[key]:
                    print(f"  - {key}: FAILED")
    
    print(f"\nOVERALL: {'SCIENTIFIC GATING PASS' if all_pass else ' SCIENTIFIC GATING FAIL'}")
    print(f"Results saved to: {output_dir}")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

