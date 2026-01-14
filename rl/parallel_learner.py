from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Queue, Event
from queue import Empty, Full
import time
import csv

import numpy as np
import torch
torch.set_num_threads(4)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.agent import DQNAgent
from rl.replay_buffer import ReplayBuffer, TransitionBatch


def learner_loop(
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queues: List[Queue],
    stop_event: Event,
    total_episodes: int = 1000,
    checkpoint_dir: str = "models/parallel",
    log_dir: str = "logs/parallel",
) -> None:
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
    
    from rl.agent import AgentConfig
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
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    global_step = 0
    total_transitions = 0
    weight_version = 0
    last_weight_sync = time.time()
    weight_sync_interval = 2.0
    
    last_checkpoint = time.time()
    checkpoint_interval = 300.0
    
    losses: List[float] = []
    best_buffer_size = 0
    
    log_path = os.path.join(log_dir, "learner_metrics.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "timestamp", "global_step", "total_transitions", "buffer_size",
        "avg_loss", "weight_version"
    ])
    log_writer.writeheader()
    
    print(f"[Learner] Started. Checkpoint dir: {checkpoint_dir}")
    print(f"[Learner] Waiting for experiences from {len(weight_queues)} actors...")
    
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
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)
                if loss is not None:
                    losses.append(float(loss))
                global_step += 1
                
                if global_step % 100 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else 0.0
                    print(f"[Learner] Step {global_step} | Transitions: {total_transitions} | "
                          f"Buffer: {len(replay_buffer)} | Loss: {avg_loss:.6f}")
                    
                    log_writer.writerow({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "global_step": global_step,
                        "total_transitions": total_transitions,
                        "buffer_size": len(replay_buffer),
                        "avg_loss": avg_loss,
                        "weight_version": weight_version,
                    })
                    log_file.flush()
            
            now = time.time()
            if now - last_weight_sync >= weight_sync_interval:
                weight_version += 1
                _broadcast_weights(agent, weight_queues, weight_version)
                last_weight_sync = now
            
            if now - last_checkpoint >= checkpoint_interval:
                ckpt_path = os.path.join(checkpoint_dir, f"learner_step{global_step}.pt")
                agent.save_checkpoint(ckpt_path, {
                    "global_step": global_step,
                    "total_transitions": total_transitions,
                    "weight_version": weight_version,
                })
                print(f"[Learner] Checkpoint saved: {ckpt_path}")
                last_checkpoint = now
                
    except KeyboardInterrupt:
        print("[Learner] Interrupted")
    finally:
        final_path = os.path.join(checkpoint_dir, f"learner_final_step{global_step}.pt")
        agent.save_checkpoint(final_path, {
            "global_step": global_step,
            "total_transitions": total_transitions,
            "weight_version": weight_version,
        })
        print(f"[Learner] Final checkpoint: {final_path}")
        log_file.close()
        print(f"[Learner] Stopped. Total steps: {global_step}, transitions: {total_transitions}")


def _receive_chunk(queue: Queue, timeout: float) -> Optional[List[Tuple]]:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        return None


def _broadcast_weights(agent: DQNAgent, weight_queues: List[Queue], version: int) -> None:
    weights = {k: v.cpu().clone() for k, v in agent.online_net.state_dict().items()}
    
    for wq in weight_queues:
        while True:
            try:
                wq.get_nowait()
            except Empty:
                break
        
        try:
            wq.put_nowait((weights, version))
        except Full:
            pass
