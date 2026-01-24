from __future__ import annotations

import os
import sys
import argparse
import json
import multiprocessing as mp
from multiprocessing import Queue, Event, Process, Value
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-episode", type=int, default=0, help="Starting episode for curriculum resume")
    parser.add_argument("--episode-offset", type=int, default=0, help="Offset for episode numbering (log only)")
    args = parser.parse_args(argv)
    
    config = load_yaml_config(args.config)
    parallel_cfg = config.get("parallel", {})
    
    if args.resume_episode > 0:
        parallel_cfg["resume_episode"] = args.resume_episode
    
    if args.episode_offset > 0:
        parallel_cfg["episode_offset"] = args.episode_offset
        
    config["parallel"] = parallel_cfg
    
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
    
    # Shared counter for epsilon decay (joint decision steps across all workers)
    # Definition: 1 global_step = 1 joint decision step = 1 env.step() call
    # With N TLS agents, this counts the shared decision, NOT individual agent actions
    # NOTE: Use 'l' (long int) instead of 'i' (int) for safety with large step counts
    global_step_counter = Value('l', 0)
    
    from rl.parallel_collector_1 import collector_process
    
    actors: List[Process] = []
    for i in range(num_actors):
        p = Process(
            target=collector_process,
            args=(i, config, experience_queue, weight_queues[i], stop_event, global_step_counter),
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
        resume_path=args.resume,
        global_step_counter=global_step_counter,  # Pass counter to restore on resume
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
    resume_path: str = None,
    global_step_counter: Value = None,
) -> None:
    from rl.agent import AgentConfig, DQNAgent
    from scripts.train import run_smoke_eval_episode  # Import smoke eval capability
    
    agent_cfg = config.get("agent", {})
    parallel_cfg = config.get("parallel", {})
    train_cfg = config.get("train", {})  # Get train config for smoke eval settings
    sumo_cfg = config.get("env", {}).get("sumo", {})
    logging_cfg = config.get("logging", {})
    
    state_dim = sumo_cfg.get("state_dim", 12)
    tls_ids = sumo_cfg.get("tls_ids", [])
    num_tls = len(tls_ids) if tls_ids else 1
    # Infer action_dim from config: prefer action_table, else action_splits × cycle_options
    action_table = sumo_cfg.get("action_table", [])
    if action_table:
        action_dim = len(action_table)
    else:
        action_splits = sumo_cfg.get("action_splits", [])
        cycle_opts = sumo_cfg.get("cycle_options_sec", sumo_cfg.get("cycle_options", [sumo_cfg.get("green_cycle_sec", 60)]))
        action_dim = max(1, len(action_splits) * len(cycle_opts) if action_splits else 1)
    
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
    
    # Smoke eval settings
    smoke_eval_every = int(train_cfg.get("smoke_eval_every", 0))
    smoke_eval_demand = int(train_cfg.get("smoke_eval_demand", 750))
    smoke_eval_horizon = int(train_cfg.get("smoke_eval_horizon_sec", 750))
    log_dir = str(logging_cfg.get("log_dir", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    run_name = str(config.get("run", {}).get("run_name", "train"))
    smoke_eval_log_path = os.path.join(log_dir, f"{run_name}_smoke_eval.csv")
    
    # Curriculum histogram settings (Gate 4)
    curriculum_hist_every = int(train_cfg.get("curriculum_hist_every", 0))
    curriculum_stats_path = os.path.join(log_dir, f"{run_name}_curriculum_stats.jsonl")
    last_curriculum_hist_episode = 0
    
    # We need to track episodes for smoke eval and curriculum hist
    # In parallel, episodes are distributed. We can approximate 'every N episodes' 
    # by tracking total episodes reported by workers.
    total_episodes_completed = 0
    last_smoke_eval_episode = 0

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
    last_logged_updates = -1
    recent_losses = []  # Track losses between log intervals
    
    checkpoint_interval_sec = float(parallel_cfg.get("checkpoint_interval_sec", 300.0))
    last_checkpoint_time = time.time()
    last_ckpt_step = -1
    
    if resume_path and os.path.exists(resume_path):
        # RESUME SEMANTICS (IMPORTANT FOR ACADEMIC CLAIMS):
        # - Restores: network weights, optimizer state, global_step_counter (epsilon decay)
        # - NOT restored: replay buffer (starts empty)
        # - This is "FINE-TUNING with epsilon continuity", NOT true continuation
        # - For true continuation: would need to restore full replay buffer state
        print(f"[Fine-tuning] Loading checkpoint: {resume_path}")
        print(f"[Fine-tuning] Note: Replay buffer NOT restored (training dynamics partially restart)")
        
        checkpoint = torch.load(resume_path, map_location=device)
        agent.online_net.load_state_dict(checkpoint["online_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        learner_updates = checkpoint.get("learner_updates", 0)
        agent_transitions_total = checkpoint.get("agent_transitions_total", 0)
        global_env_steps_total = checkpoint.get("global_env_steps_total", 0)
        
        # CRITICAL: Restore global_step_counter for epsilon decay continuity
        resumed_global_steps = checkpoint.get("global_decision_steps", global_env_steps_total)
        if global_step_counter is not None:
            with global_step_counter.get_lock():
                global_step_counter.value = resumed_global_steps
            print(f"[Fine-tuning] Restored epsilon clock: global_step={resumed_global_steps}")
        
        last_ckpt_step = learner_updates
        last_logged_updates = learner_updates
        
        if agent_transitions_total >= learning_starts:
            learning_started = True
        
        print(f"[Fine-tuning] Restored: learner_updates={learner_updates}, prev_transitions={agent_transitions_total}")
    
    print(f"Learner config: learning_starts={learning_starts}, train_freq={train_freq}, "
          f"max_update_time_ms={max_update_time_ms}, num_tls={num_tls}")
    print("Learner waiting for experiences...")
    
    # Curriculum tracking: episode_uid -> phase_idx mapping
    episode_to_phase: Dict[int, int] = {}
    sampled_batch_phase_counts: Dict[int, int] = {}  # Rolling histogram of sampled batches
    last_histogram_log_updates = 0
    HISTOGRAM_LOG_INTERVAL = 1000  # Log histogram every N updates
    
    # Calculate total target episodes for stop condition
    curriculum_cfg = config.get("curriculum", {})
    phase_idx_to_name: Dict[int, str] = {}  # Map phase_idx -> phase_name for Gate 4 logs
    if curriculum_cfg.get("enabled", False) and curriculum_cfg.get("phases"):
        total_target_episodes = sum(p.get("episodes", 0) for p in curriculum_cfg["phases"])
        for idx, p in enumerate(curriculum_cfg["phases"]):
            phase_idx_to_name[idx] = p.get("name", f"phase_{idx}")
    else:
        total_target_episodes = int(train_cfg.get("episodes", 100))
    print(f"Learner expects ~{total_target_episodes} total episodes")
    if phase_idx_to_name:
        print(f"Curriculum phases: {phase_idx_to_name}")
    
    # Track queue empty time for graceful shutdown
    queue_empty_start_time = None
    QUEUE_EMPTY_TIMEOUT_SEC = 30.0  # Stop if queue empty for 30s after target reached
    
    try:
        while not stop_event.is_set():
            chunk_data = _receive_chunk(experience_queue, timeout=0.1)
            
            # Check for graceful shutdown when training complete
            if chunk_data is None:
                if int(total_episodes_completed) >= total_target_episodes:
                    if queue_empty_start_time is None:
                        queue_empty_start_time = time.time()
                        print(f"[Learner] Training complete: {int(total_episodes_completed)}/{total_target_episodes} episodes. Waiting for workers to finish...")
                    elif time.time() - queue_empty_start_time > QUEUE_EMPTY_TIMEOUT_SEC:
                        print(f"[Learner] All workers finished. Stopping learner.")
                        break
            else:
                queue_empty_start_time = None  # Reset timer if we get data

            if chunk_data is not None:
                if isinstance(chunk_data, dict):
                    transitions = chunk_data.get("transitions", [])
                    chunk_global_steps = chunk_data.get("global_steps", 0)
                    chunk_phase_idx = chunk_data.get("phase_idx", -1)
                    chunk_episode_uid = chunk_data.get("episode_uid", -1)
                    
                    # Track completed episodes (approximate via unique new UIDs seen or provided explicitly?)
                    # A better proxy for parallel might be transitions / average_steps, but we want exact count
                    # Ideally workers send a "episode_done" signal. 
                    # Assuming we just infer from chunk: if chunk contains 'done', we might count.
                    # But chunk_data doesn't explicitly aggregate episode counts. 
                    # We'll use a simple heuristic or checking for 'done' in transitions.
                    
                    # Update episode_to_phase mapping
                    if chunk_episode_uid >= 0 and chunk_phase_idx >= 0:
                        episode_to_phase[chunk_episode_uid] = chunk_phase_idx
                        
                    # Check for done flags to count episodes
                    dones_in_chunk = sum(1 for t in transitions if (len(t)==6 and t[4]) or (len(t)==5 and t[4]))
                    # Note: one episode produces done signal for EACH agent (num_tls).
                    # So roughly dones / num_tls = episodes.
                    if num_tls > 0:
                        total_episodes_completed += (dones_in_chunk / num_tls)
                        
                else:
                    transitions = chunk_data
                    chunk_global_steps = len(chunk_data) // num_tls
                    # Fallback for legacy format
                    dones_in_chunk = sum(1 for t in transitions if t[4])
                    if num_tls > 0:
                        total_episodes_completed += (dones_in_chunk / num_tls)
                
                chunk_len = len(transitions)
                expected_len = chunk_global_steps * num_tls
                if chunk_len != expected_len and chunk_global_steps > 0:
                    print(f"[WARN] Chunk invariant violation: len={chunk_len} != {chunk_global_steps}*{num_tls}={expected_len}")
                
                for transition in transitions:
                    # Handle both 5-tuple (legacy) and 6-tuple (with episode_uid)
                    if len(transition) == 6:
                        s, a, r, ns, d, ep_uid = transition
                    else:
                        s, a, r, ns, d = transition
                        ep_uid = -1  # Unknown (backward compat)
                    
                    agent.replay_buffer.push(
                        state=np.asarray(s, dtype=np.float32),
                        action=int(a),
                        reward=float(r),
                        next_state=np.asarray(ns, dtype=np.float32),
                        done=bool(d),
                        episode_uid=int(ep_uid),
                    )
                
                agent_transitions_total += chunk_len
                global_env_steps_total += chunk_global_steps
                pending_transitions += chunk_len

            # Check for Smoke Eval trigger
            if smoke_eval_every > 0 and int(total_episodes_completed) > last_smoke_eval_episode:
                # Check if we crossed a multiple of smoke_eval_every
                # Example: last=0, current=10, every=10 -> run.
                # Example: last=10, current=19 -> no.
                # Example: last=19, current=21 -> run (crossed 20).
                
                # Logic: Find the highest multiple of every <= current.
                target_multiple = (int(total_episodes_completed) // smoke_eval_every) * smoke_eval_every
                
                if target_multiple > 0 and target_multiple > last_smoke_eval_episode:
                    print(f"[SmokeEval] Triggered at ~{int(total_episodes_completed)} episodes (target {target_multiple})")
                    warmup_eval = int(config.get("env", {}).get("sumo", {}).get("warmup_sec", 300))
                    try:
                        run_smoke_eval_episode(
                            agent=agent,
                            base_config=config,
                            demand=smoke_eval_demand,
                            seed=int(seed + target_multiple + 123456), # Distinct seed
                            horizon_sec=smoke_eval_horizon,
                            warmup_sec=warmup_eval,
                            log_path=smoke_eval_log_path,
                            episode_idx=int(target_multiple),
                            phase_name="parallel_mix", # Hard to get exact phase in parallel learner
                        )
                        last_smoke_eval_episode = target_multiple # Update marker
                    except Exception as smoke_err:
                        print(f"[SmokeEval] Failed: {smoke_err}")

            # Check for Curriculum Histogram trigger (Gate 4)
            if curriculum_hist_every > 0 and int(total_episodes_completed) > last_curriculum_hist_episode:
                target_hist = (int(total_episodes_completed) // curriculum_hist_every) * curriculum_hist_every
                
                if target_hist > 0 and target_hist > last_curriculum_hist_episode and len(agent.replay_buffer) > 0:
                    print(f"[CurriculumHist] Logging at ~{int(total_episodes_completed)} episodes (target {target_hist})")
                    try:
                        buffer_hist = agent.replay_buffer.get_phase_histogram(episode_to_phase)
                        sampled_hist: Dict[int, int] = {}
                        sample_size = min(256, len(agent.replay_buffer))
                        batch = agent.replay_buffer.sample(batch_size=sample_size, device=torch.device("cpu"))
                        ep_uids = batch.episode_uids.cpu().numpy().reshape(-1)
                        for uid in ep_uids:
                            phase_val = episode_to_phase.get(int(uid), -1)
                            sampled_hist[phase_val] = sampled_hist.get(phase_val, 0) + 1
                        
                        # Create named histograms (phase_name -> count) for readability
                        buffer_hist_named = {phase_idx_to_name.get(k, f"unknown_{k}"): v for k, v in buffer_hist.items()}
                        sampled_hist_named = {phase_idx_to_name.get(k, f"unknown_{k}"): v for k, v in sampled_hist.items()}
                        
                        hist_entry = {
                            "timestamp": time.time(),
                            "episode": int(target_hist),
                            "global_step": int(global_env_steps_total),
                            "buffer_size": len(agent.replay_buffer),
                            "buffer_phase_histogram": buffer_hist,
                            "buffer_phase_histogram_named": buffer_hist_named,
                            "sampled_batch_phase_histogram": sampled_hist,
                            "sampled_batch_phase_histogram_named": sampled_hist_named,
                            "phase_idx_to_name": phase_idx_to_name,
                            "learner_updates": int(learner_updates),
                        }
                        with open(curriculum_stats_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(hist_entry) + "\n")
                        last_curriculum_hist_episode = target_hist
                        print(f"[CurriculumHist] Saved snapshot: buffer_size={len(agent.replay_buffer)}, sampled_hist_named={sampled_hist_named}")
                    except Exception as hist_err:
                        print(f"[CurriculumHist] Failed: {hist_err}")

            if not learning_started:
                if agent_transitions_total >= learning_starts:
                    learning_started = True
                    print(f"Learning started at {agent_transitions_total} transitions "
                          f"(buffer: {len(agent.replay_buffer)})")
                else:
                    if agent_transitions_total % 25000 == 0 and agent_transitions_total > 0:
                        print(f"Warmup: {agent_transitions_total}/{learning_starts} transitions...")
                    continue

            update_start = time.perf_counter()
            updates_this_iter = 0
            
            while pending_transitions >= train_freq:
                elapsed_ms = (time.perf_counter() - update_start) * 1000
                if elapsed_ms > max_update_time_ms:
                    break
                    
                metrics = agent.update()
                if metrics is not None:
                    learner_updates += 1
                    updates_this_iter += 1
                    # Extract loss from metrics dict (agent.update now returns dict)
                    loss_value = metrics.get('loss', 0.0) if isinstance(metrics, dict) else float(metrics)
                    recent_losses.append(loss_value)
                pending_transitions -= train_freq
                
                if learner_updates % sync_every_updates == 0:
                    _broadcast_weights(agent, weight_queues)
            
            now = time.time()
            should_log_step = learner_updates > 0 and learner_updates % 100 == 0 and learner_updates != last_logged_updates
            should_log_time = learner_updates > 0 and now - last_log_time >= 30.0 and learner_updates != last_logged_updates
            
            if should_log_step or should_log_time:
                utd_agent = learner_updates / max(1, agent_transitions_total)
                utd_global = learner_updates / max(1, global_env_steps_total)
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                print(f"Step {learner_updates} | Trans: {agent_transitions_total} | "
                      f"Global: {global_env_steps_total} | Pending: {pending_transitions} | "
                      f"UTD_agent: {utd_agent:.4f} | UTD_global: {utd_global:.2f} | Loss: {avg_loss:.4f}")
                recent_losses.clear()  # Reset for next interval
                last_log_time = now
                last_logged_updates = learner_updates
            
            if now - last_checkpoint_time >= checkpoint_interval_sec and learner_updates > 0 and learner_updates != last_ckpt_step:
                # Get current global_step for epsilon decay continuity
                current_global_steps = global_step_counter.value if global_step_counter else global_env_steps_total
                ckpt_path = os.path.join(model_dir, f"parallel_ckpt_step{learner_updates}.pt")
                agent.save_checkpoint(ckpt_path, {
                    "learner_updates": learner_updates,
                    "agent_transitions_total": agent_transitions_total,
                    "global_env_steps_total": global_env_steps_total,
                    "global_decision_steps": current_global_steps,  # For epsilon decay resume
                })
                print(f"[Checkpoint] Saved: {ckpt_path} (global_decision_steps={current_global_steps})")
                last_checkpoint_time = now
                last_ckpt_step = learner_updates
                    
    except KeyboardInterrupt:
        pass
    finally:
        # Get final global_step for epsilon decay continuity
        final_global_steps = global_step_counter.value if global_step_counter else global_env_steps_total
        final_path = os.path.join(model_dir, f"parallel_final_step{learner_updates}.pt")
        agent.save_checkpoint(final_path, {
            "learner_updates": learner_updates,
            "agent_transitions_total": agent_transitions_total,
            "global_env_steps_total": global_env_steps_total,
            "global_decision_steps": final_global_steps,  # For epsilon decay resume
        })
        print(f"Saved: {final_path} (global_decision_steps={final_global_steps})")
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
