from __future__ import annotations

import os
import sys
import argparse
import multiprocessing as mp
from multiprocessing import Queue, Event, Process
from pathlib import Path
from typing import Any, Dict, List
from queue import Empty, Full
import time

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    
    config = load_yaml_config(args.config)
    parallel_cfg = config.get("parallel", {})
    
    if not parallel_cfg.get("enabled", False):
        print("parallel.enabled is false or missing. Use scripts/train.py for standard training.")
        return 0
    
    num_actors = int(parallel_cfg.get("num_actors", 4))
    base_port = int(parallel_cfg.get("base_port", 8813))
    base_seed = int(parallel_cfg.get("base_seed", 42))
    chunk_size = int(parallel_cfg.get("chunk_size", 256))
    queue_max_chunks = int(parallel_cfg.get("queue_max_chunks", 200))
    sync_every_updates = int(parallel_cfg.get("sync_every_updates", 100))
    epsilon_base = float(parallel_cfg.get("epsilon_base", 0.2))
    epsilon_delta = float(parallel_cfg.get("epsilon_worker_delta", 0.02))
    
    print("Parallel Training Plan")
    print(f"  num_actors: {num_actors}")
    print(f"  base_port: {base_port}")
    print(f"  base_seed: {base_seed}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  queue_max_chunks: {queue_max_chunks}")
    print(f"  sync_every_updates: {sync_every_updates}")
    print()
    print("Worker assignments:")
    for i in range(num_actors):
        port = base_port + i
        seed = base_seed + i
        eps = min(1.0, epsilon_base + epsilon_delta * i)
        print(f"  Worker {i}: port={port}, seed={seed}, epsilon={eps:.3f}")
    print()
    
    if args.dry_run:
        print("Dry run complete. No processes spawned.")
        return 0
    
    mp.set_start_method("spawn", force=True)
    
    experience_queue = Queue(maxsize=queue_max_chunks)
    weight_queues = [Queue(maxsize=1) for _ in range(num_actors)]
    stop_event = Event()
    
    from rl.parallel_collector_1 import collector_process
    
    actors: List[Process] = []
    for i in range(num_actors):
        p = Process(
            target=collector_process,
            args=(i, config, experience_queue, weight_queues[i], stop_event),
            daemon=True,
        )
        p.start()
        actors.append(p)
        print(f"Started collector {i}")
    
    print("Starting learner...")
    
    _run_learner(
        config=config,
        experience_queue=experience_queue,
        weight_queues=weight_queues,
        stop_event=stop_event,
        sync_every_updates=sync_every_updates,
    )
    
    stop_event.set()
    for i, p in enumerate(actors):
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
    
    print("Done")
    return 0


def _run_learner(
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queues: List[Queue],
    stop_event: Event,
    sync_every_updates: int,
) -> None:
    from rl.agent import AgentConfig, DQNAgent
    
    agent_cfg = config.get("agent", {})
    parallel_cfg = config.get("parallel", {})
    sumo_cfg = config.get("env", {}).get("sumo", {})
    
    state_dim = sumo_cfg.get("state_dim", 12)
    tls_ids = sumo_cfg.get("tls_ids", [])
    num_tls = len(tls_ids) if tls_ids else 1
    action_dim = 15
    
    hidden_dims = agent_cfg.get("hidden_dims", [192, 192])
    gamma = agent_cfg.get("gamma", 0.99)
    lr = agent_cfg.get("learning_rate", 0.0003)
    batch_size = agent_cfg.get("batch_size", 256)
    buffer_size = agent_cfg.get("replay_buffer_size", 200000)
    target_update_freq = agent_cfg.get("target_update_freq", 3000)
    clip_grad_norm = agent_cfg.get("clip_grad_norm", 10.0)
    seed = config.get("run", {}).get("seed", 42)
    use_huber_loss = agent_cfg.get("use_huber_loss", True)
    learning_starts = int(agent_cfg.get("learning_starts", 5000))
    train_freq = int(agent_cfg.get("train_freq", 4))
    max_update_time_ms = float(parallel_cfg.get("max_update_time_ms", 50.0))
    
    agent_config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        gamma=gamma,
        learning_rate=lr,
        batch_size=batch_size,
        replay_buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        clip_grad_norm=clip_grad_norm,
        seed=seed,
        use_huber_loss=use_huber_loss,
    )
    
    device = torch.device("cpu")
    agent = DQNAgent(agent_config, device=device)
    agent.to_train_mode()
    
    model_dir = config.get("logging", {}).get("model_dir", "models/parallel")
    os.makedirs(model_dir, exist_ok=True)
    
    learner_updates = 0
    agent_transitions_total = 0
    global_env_steps_total = 0
    pending_transitions = 0
    learning_started = False
    last_log_time = time.time()
    
    print(f"Learner config: learning_starts={learning_starts}, train_freq={train_freq}, "
          f"max_update_time_ms={max_update_time_ms}, num_tls={num_tls}")
    print("Learner waiting for experiences...")
    
    try:
        while not stop_event.is_set():
            chunk_data = _receive_chunk(experience_queue, timeout=0.1)
            
            if chunk_data is not None:
                if isinstance(chunk_data, dict):
                    transitions = chunk_data.get("transitions", [])
                    chunk_global_steps = chunk_data.get("global_steps", 0)
                else:
                    transitions = chunk_data
                    chunk_global_steps = len(chunk_data) // num_tls
                
                chunk_len = len(transitions)
                expected_len = chunk_global_steps * num_tls
                if chunk_len != expected_len and chunk_global_steps > 0:
                    print(f"[WARN] Chunk invariant violation: len={chunk_len} != {chunk_global_steps}*{num_tls}={expected_len}")
                
                for transition in transitions:
                    s, a, r, ns, d = transition
                    agent.replay_buffer.push(
                        state=np.asarray(s, dtype=np.float32),
                        action=int(a),
                        reward=float(r),
                        next_state=np.asarray(ns, dtype=np.float32),
                        done=bool(d),
                    )
                
                agent_transitions_total += chunk_len
                global_env_steps_total += chunk_global_steps
                pending_transitions += chunk_len

            if not learning_started:
                if agent_transitions_total >= learning_starts:
                    learning_started = True
                    print(f"Learning started at {agent_transitions_total} transitions "
                          f"(buffer: {len(agent.replay_buffer)})")
                else:
                    if agent_transitions_total % 1000 == 0 and agent_transitions_total > 0:
                        print(f"Warmup: {agent_transitions_total}/{learning_starts} transitions...")
                    continue

            update_start = time.perf_counter()
            updates_this_iter = 0
            
            while pending_transitions >= train_freq:
                elapsed_ms = (time.perf_counter() - update_start) * 1000
                if elapsed_ms > max_update_time_ms:
                    break
                    
                loss = agent.update()
                if loss is not None:
                    learner_updates += 1
                    updates_this_iter += 1
                pending_transitions -= train_freq
                
                if learner_updates % sync_every_updates == 0:
                    _broadcast_weights(agent, weight_queues)
            
            now = time.time()
            if now - last_log_time >= 10.0 and learner_updates > 0:
                utd_agent = learner_updates / max(1, agent_transitions_total)
                utd_global = learner_updates / max(1, global_env_steps_total)
                print(f"Step {learner_updates} | Trans: {agent_transitions_total} | "
                      f"Global: {global_env_steps_total} | Pending: {pending_transitions} | "
                      f"UTD_agent: {utd_agent:.4f} | UTD_global: {utd_global:.2f}")
                last_log_time = now
                    
    except KeyboardInterrupt:
        pass
    finally:
        final_path = os.path.join(model_dir, f"parallel_final_step{learner_updates}.pt")
        agent.save_checkpoint(final_path, {
            "learner_updates": learner_updates,
            "agent_transitions_total": agent_transitions_total,
            "global_env_steps_total": global_env_steps_total,
        })
        print(f"Saved: {final_path}")
        utd_agent = learner_updates / max(1, agent_transitions_total)
        print(f"Final UTD_agent: {utd_agent:.4f} (target: 0.25)")






def _receive_chunk(queue: Queue, timeout: float):
    try:
        return queue.get(timeout=timeout)
    except Empty:
        return None


def _broadcast_weights(agent: DQNAgent, weight_queues: List[Queue]) -> None:
    weights = {k: v.cpu().clone() for k, v in agent.online_net.state_dict().items()}
    
    for wq in weight_queues:
        while True:
            try:
                wq.get_nowait()
            except Empty:
                break
        
        try:
            wq.put_nowait(weights)
        except Full:
            pass


if __name__ == "__main__":
    sys.exit(main())
