from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from env.base_env import BaseEnv
from env.normalization import StateNormalizer
from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram
from env.toy_queue_env import ToyQueueEnv, ToyQueueEnvConfig
from rl.agent import AgentConfig, DQNAgent
from rl.utils import resolve_device


def build_env(config: Dict[str, Any]) -> BaseEnv:
    env_config = config.get("env", {})
    env_type = str(env_config.get("type", "")).strip().lower()

    if env_type == "toy":
        toy_cfg = env_config.get("toy", {})
        toy_env_config = ToyQueueEnvConfig(
            max_steps=int(toy_cfg.get("max_steps", 200)),
            arrival_prob=float(toy_cfg.get("arrival_prob", 0.7)),
            serve_slow_rate=int(toy_cfg.get("serve_slow_rate", 1)),
            serve_fast_rate=int(toy_cfg.get("serve_fast_rate", 3)),
            seed=int(config.get("run", {}).get("seed", 0)),
        )
        return ToyQueueEnv(toy_env_config)

    if env_type == "sumo":
        sumo_cfg = env_config.get("sumo", {})
        lane_cfg = sumo_cfg.get("lane_groups", {})
        phase_cfg = sumo_cfg.get("phase_program", {})

        lanes = SumoLaneGroups(
            lanes_ns_ctrl=[str(x) for x in lane_cfg.get("lanes_ns_ctrl", [])],
            lanes_ew_ctrl=[str(x) for x in lane_cfg.get("lanes_ew_ctrl", [])],
            lanes_rt_ns=[str(x) for x in lane_cfg.get("lanes_rt_ns", [])],
            lanes_rt_ew=[str(x) for x in lane_cfg.get("lanes_rt_ew", [])],
        )

        phases = SumoPhaseProgram(
            ns_green=int(phase_cfg.get("ns_green", 0)),
            ew_green=int(phase_cfg.get("ew_green", 1)),
            ns_yellow=phase_cfg.get("ns_yellow"),
            ew_yellow=phase_cfg.get("ew_yellow"),
            all_red=phase_cfg.get("all_red"),
        )

        action_splits_raw = sumo_cfg.get("action_splits", [])
        action_splits = [(float(x[0]), float(x[1])) for x in action_splits_raw] if len(action_splits_raw) > 0 else []

        sumo_env_config = SumoEnvConfig(
            sumo_binary=str(sumo_cfg.get("sumo_binary", "sumo")),
            net_file=str(sumo_cfg.get("net_file", "")),
            route_file=str(sumo_cfg.get("route_file", "")),
            additional_files=[str(x) for x in sumo_cfg.get("additional_files", [])],
            tls_id=str(sumo_cfg.get("tls_id", "tls0")),
            step_length_sec=float(sumo_cfg.get("step_length_sec", 1.0)),
            cycle_length_sec=int(sumo_cfg.get("cycle_length_sec", 60)),
            yellow_sec=int(sumo_cfg.get("yellow_sec", 0)),
            all_red_sec=int(sumo_cfg.get("all_red_sec", 0)),
            max_cycles=int(sumo_cfg.get("max_cycles", 60)),
            seed=int(config.get("run", {}).get("seed", 0)),
            rho_min=float(sumo_cfg.get("rho_min", 0.1)),
            action_splits=action_splits,
            include_transition_in_waiting=bool(sumo_cfg.get("include_transition_in_waiting", False)),
            terminate_on_empty=bool(sumo_cfg.get("terminate_on_empty", True)),
            sumo_extra_args=[str(x) for x in sumo_cfg.get("sumo_extra_args", [])],
        )

        normalization_cfg = config.get("normalization", {})
        mean = normalization_cfg.get("mean", [0.0, 0.0, 0.0, 0.0])
        std = normalization_cfg.get("std", [1.0, 1.0, 1.0, 1.0])

        normalizer = StateNormalizer(mean=mean, std=std)

        return SUMOEnv(config=sumo_env_config, lanes=lanes, phases=phases, normalizer=normalizer)

    raise ValueError(f"Unsupported env type: {env_type}")


def build_agent(config: Dict[str, Any], env: BaseEnv) -> Tuple[DQNAgent, torch.device]:
    run_cfg = config.get("run", {})
    device = resolve_device(str(run_cfg.get("device", "cpu")))

    agent_cfg = config.get("agent", {})
    hidden_dims = agent_cfg.get("hidden_dims", [128, 128])

    agent_config = AgentConfig(
        state_dim=int(env.state_dim),
        action_dim=int(env.action_dim),
        hidden_dims=[int(x) for x in hidden_dims],
        gamma=float(agent_cfg.get("gamma", 0.98)),
        learning_rate=float(agent_cfg.get("learning_rate", 1e-3)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        replay_buffer_size=int(agent_cfg.get("replay_buffer_size", 100000)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 1000)),
        seed=int(run_cfg.get("seed", 0)),
    )

    agent = DQNAgent(config=agent_config, device=device)
    return agent, device


def format_state(state: np.ndarray) -> str:
    values = [float(x) for x in np.asarray(state).reshape(-1).tolist()]
    return "[" + ", ".join([f"{v:.3f}" for v in values]) + "]"
