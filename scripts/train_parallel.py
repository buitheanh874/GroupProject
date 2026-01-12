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
    from rl.replay_buffer import ReplayBuffer
    
    agent_cfg = config.get("agent", {})
    state_dim = config.get("env", {}).get("sumo", {}).get("state_dim", 12)
    action_dim = 15
    
    hidden_dims = agent_cfg.get("hidden_dims", [192, 192])
    gamma = agent_cfg.get("gamma", 0.99)
    lr = agent_cfg.get("learning_rate", 0.0003)
    batch_size = agent_cfg.get("batch_size", 256)
    buffer_size = agent_cfg.get("replay_buffer_size", 200000)
    target_update_freq = agent_cfg.get("target_update_freq", 3000)
    seed = config.get("run", {}).get("seed", 42)
    
    agent_config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        gamma=gamma,
        learning_rate=lr,
        batch_size=batch_size,
        replay_buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        seed=seed,
    )
    
    device = torch.device("cpu")
    agent = DQNAgent(agent_config, device=device)
    agent.to_train_mode()
    
    replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed, state_dim=state_dim)
    
    model_dir = config.get("train", {}).get("model_dir", "models/parallel")
    os.makedirs(model_dir, exist_ok=True)
    
    global_step = 0
    total_transitions = 0
    
    print("Learner waiting for experiences...")
    
    try:
        while not stop_event.is_set():
            chunk = _receive_chunk(experience_queue, timeout=0.1)
            
            if chunk is not None:
                for transition in chunk:
                    s, a, r, ns, d = transition
                    replay_buffer.push(
                        state=np.asarray(s, dtype=np.float32),
                        action=int(a),
                        reward=float(r),
                        next_state=np.asarray(ns, dtype=np.float32),
                        done=bool(d),
                    )
                    total_transitions += 1
            
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size, device)
                loss = agent.update(batch)
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"Step {global_step} | Transitions: {total_transitions} | Buffer: {len(replay_buffer)}")
                
                if global_step % sync_every_updates == 0:
                    _broadcast_weights(agent, weight_queues)
                    
    except KeyboardInterrupt:
        pass
    finally:
        final_path = os.path.join(model_dir, f"parallel_final_step{global_step}.pt")
        agent.save_checkpoint(final_path, {"global_step": global_step, "total_transitions": total_transitions})
        print(f"Saved: {final_path}")


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
